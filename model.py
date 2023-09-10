import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from torchsummary import summary


class MeanShift(nn.Module):
    def __init__(self, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = 0.4488 * sign
        g = 0.4371 * sign
        b = 0.4040 * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

# channel attention


class CA(nn.Module):
    def __init__(self, channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_weight = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(1, channel, 1),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_weight(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size),
            act,
            conv(n_feat, n_feat, kernel_size),
            CA(n_feat)
        )

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()

        n_feat = 16
        kernel_size = 3
        act = nn.ReLU(True)

        body = []
        body.append(conv(3, n_feat, kernel_size))

        for _ in range(16):
            body.append(RCAB(conv, n_feat, kernel_size, act))

        body.append(conv(n_feat, 27, 3))

        self.body = nn.Sequential(*body)
        self.ps = nn.PixelShuffle(3)

        self.sub_mean = MeanShift(True)
        self.add_mean = MeanShift(False)

    def forward(self, x):
        # b = F.interpolate(x, scale_factor=3, mode='bicubic')
        x = self.sub_mean(x)
        x = self.body(x)
        x = self.ps(x)
        x = self.add_mean(x)
        # x += b
        return x


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SuperResolution().to(device)

    # torchsummary report
    summary(model, input_size=(3, 224, 224))

    # thop report
    img = torch.randn(1, 3, 256, 256).to(device)
    macs, params = profile(model, inputs=(img,), verbose=False)
    flops = macs * 2 / 1e9  # G
    params = params / 1e6  # M
    print()
    print("======================================")
    print(f"FLOPs : { flops } G")
    print(f"PARAMS : { params } M ")
    print("======================================")
    if flops < 11.1:
        print("Your FLOPs is smaller than 11.1 G.")
    else:
        print("Your FLOPs is larger than 11.1 G.")
    print("======================================")
    print()


if __name__ == "__main__":
    main()

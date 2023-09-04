import torch
import torch.nn as nn
# import torch.nn.functional as F
from thop import profile
from torchsummary import summary


# channel attention
class CA(nn.Module):
    def __init__(self, channel):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_weight = nn.Sequential(
            nn.Conv2d(channel, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(1, channel, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_weight(y)
        return x * y


class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bn=False, act=nn.ReLU(True)):
        super(RCAB, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size))
            if bn:
                m.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                m.append(act)
        m.append(CA(n_feat))
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act,
                 n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(conv, n_feat, kernel_size,
                 bn=False, act=nn.ReLU(True))
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

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

        n_resgroups = 2
        n_resblocks = 4
        n_feats = 16
        kernel_size = 3
        act = nn.ReLU(True)

        head = [conv(3, n_feats, kernel_size)]

        body = [
            ResidualGroup(conv, n_feats, kernel_size, act=act,
                          n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]

        body.append(conv(n_feats, n_feats, kernel_size))

        tail = [
            conv(n_feats, 27, 3),
            nn.PixelShuffle(3),
            act
        ]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
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

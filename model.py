import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary


class SuperResolution(nn.Module):
    def __init__(self):
        super(SuperResolution, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 27, 3, 1, 1)
        self.PS = nn.PixelShuffle(3)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.PS(x)
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

import torch
import torch.nn as nn
from thop import profile


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
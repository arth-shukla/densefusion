import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

def get_unet_cls(bilinear=False):
    def make_unet_bilinear():
        return UNet(bilinear=True)
    def make_unet():
        return UNet()
    return make_unet_bilinear if bilinear else make_unet


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        return F.relu(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.mp = nn.MaxPool2d(kernel_size=2)
        self.dc = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x):
        x = self.mp(x)
        return self.dc(x)


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, int(in_channels / 2))
        else:
            self.up = nn.ConvTranspose2d(in_channels, int(in_channels / 2), kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=82, bilinear=False):
        super().__init__()

        factor = bilinear + 1

        self.in_tile = DoubleConv(n_channels, 64, 64)
        self.d1 = DownConv(64, 128)
        self.d2 = DownConv(128, 256)
        self.d3 = DownConv(256, 512)
        self.d4 = DownConv(512, int(1024 / factor))

        self.u1 = UpConv(1024, int(512 / factor), bilinear)
        self.u2 = UpConv(512, int(256 / factor), bilinear)
        self.u3 = UpConv(256, int(128 / factor), bilinear)
        self.u4 = UpConv(128, 64, bilinear)
        self.out_tile = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.in_tile(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)

        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        return self.out_tile(x)

    def use_checkpointing(self):
        self.in_tile = checkpoint(self.in_tile)
        self.d1 = checkpoint(self.d1)
        self.d2 = checkpoint(self.d2)
        self.d3 = checkpoint(self.d3)
        self.d4 = checkpoint(self.d4)
        self.u1 = checkpoint(self.u1)
        self.u2 = checkpoint(self.u2)
        self.u3 = checkpoint(self.u3)
        self.u4 = checkpoint(self.u4)
        self.out_tile = checkpoint(self.out_tile)

import torch
import torch.nn as nn
import torch.nn.functional as F
from learning.utils import OBJ_NAMES

class SegNet(nn.Module):
    def __init__(self, in_chn=3, out_chn=len(OBJ_NAMES), bn_momentum=0.1):
        super().__init__()

        self.e_conv11 = nn.Conv2d(in_chn, 64, kernel_size=3, padding=1)
        self.e_bn11 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.e_conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.e_bn12 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.e_conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.e_bn21 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.e_conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.e_bn22 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.e_conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e_bn31 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.e_conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.e_bn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.e_conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.e_bn33 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.e_conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.e_bn41 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.e_conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e_bn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.e_conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e_bn43 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.e_conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e_bn51 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.e_conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e_bn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.e_conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.e_bn53 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.d_conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d_bn53 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.d_conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d_bn52 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.d_conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d_bn51 = nn.BatchNorm2d(512, momentum=bn_momentum)

        self.d_conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d_bn43 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.d_conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.d_bn42 = nn.BatchNorm2d(512, momentum=bn_momentum)
        self.d_conv41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d_bn41 = nn.BatchNorm2d(256, momentum=bn_momentum)

        self.d_conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d_bn33 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.d_conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.d_bn32 = nn.BatchNorm2d(256, momentum=bn_momentum)
        self.d_conv31 = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.d_bn31 = nn.BatchNorm2d(128, momentum=bn_momentum)

        self.d_conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.d_bn22 = nn.BatchNorm2d(128, momentum=bn_momentum)
        self.d_conv21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d_bn21 = nn.BatchNorm2d(64, momentum=bn_momentum)

        self.d_conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.d_bn12 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.d_conv11 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1)


    def forward(self, x):

        # ENCODE LAYERS
        # Stage 1
        x11 = F.relu(self.e_bn11(self.e_conv11(x)))
        x12 = F.relu(self.e_bn12(self.e_conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2, return_indices=True)
        size1 = x1p.size()

        # Stage 2
        x21 = F.relu(self.e_bn21(self.e_conv21(x1p)))
        x22 = F.relu(self.e_bn22(self.e_conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2, return_indices=True)
        size2 = x2p.size()

        # Stage 3
        x31 = F.relu(self.e_bn31(self.e_conv31(x2p)))
        x32 = F.relu(self.e_bn32(self.e_conv32(x31)))
        x33 = F.relu(self.e_bn33(self.e_conv33(x32)))
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2, return_indices=True)
        size3 = x3p.size()

        # Stage 4
        x41 = F.relu(self.e_bn41(self.e_conv41(x3p)))
        x42 = F.relu(self.e_bn42(self.e_conv42(x41)))
        x43 = F.relu(self.e_bn43(self.e_conv43(x42)))
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2, return_indices=True)
        size4 = x4p.size()

        # Stage 5
        x51 = F.relu(self.e_bn51(self.e_conv51(x4p)))
        x52 = F.relu(self.e_bn52(self.e_conv52(x51)))
        x53 = F.relu(self.e_bn53(self.e_conv53(x52)))
        x5p, id5 = F.max_pool2d(x53,kernel_size=2, stride=2, return_indices=True)

        # DECODE LAYERS
        # Stage 5
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=size4)
        x53d = F.relu(self.d_bn53(self.d_conv53(x5d)))
        x52d = F.relu(self.d_bn52(self.d_conv52(x53d)))
        x51d = F.relu(self.d_bn51(self.d_conv51(x52d)))

        # Stage 4
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=size3)
        x43d = F.relu(self.d_bn43(self.d_conv43(x4d)))
        x42d = F.relu(self.d_bn42(self.d_conv42(x43d)))
        x41d = F.relu(self.d_bn41(self.d_conv41(x42d)))

        # Stage 3
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=size2)
        x33d = F.relu(self.d_bn33(self.d_conv33(x3d)))
        x32d = F.relu(self.d_bn32(self.d_conv32(x33d)))
        x31d = F.relu(self.d_bn31(self.d_conv31(x32d)))

        # Stage 2
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=size1)
        x22d = F.relu(self.d_bn22(self.d_conv22(x2d)))
        x21d = F.relu(self.d_bn21(self.d_conv21(x22d)))

        # Stage 1
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.d_bn12(self.d_conv12(x1d)))
        x11d = self.d_conv11(x12d)

        x = F.softmax(x11d, dim=1)

        return x

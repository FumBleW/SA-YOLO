import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import EMA
from ..modules.block import *
from ..modules.conv import *
# from ultralytics.nn.modules import Conv, C2f, Bottleneck
from .dsconv import DySnakeConv, DSConv

__all__ = (
    "DySnakeConv",
    "BiFPN_Concat2",
    "BiFPN_Concat3",
    "ECAAttention",
    "C2f_EMA",
)


class Bottleneck_DySnakeConv(Bottleneck):
    """Standard bottleneck with DySnakeConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DySnakeConv(c_, c2, k[1])
        self.cv3 = Conv(c2 * 3, c2, k=1)

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class C2f_DySnakeConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DySnakeConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.gap(x)
        y = y.squeeze(-1).permute(0, 2, 1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        return x * y.expand_as(x)


class C2f_EMA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.EMA = EMA(self.c * 2, factor=8)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.EMA(self.cv1(x)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.EMA(self.cv1(x)).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class MS_SCEM(nn.Module):
    """Multi-scale strip convolution enhancement module."""
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 9], act=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = in_channels // 4

        self.branch1 = Conv(in_channels, mid_channels, k=1, act=act)

        self.branch2 = nn.Sequential(
            Conv(in_channels, mid_channels, k=1, act=act),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_sizes[0]), stride=1,
                      padding=(0, kernel_sizes[0] // 2), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_sizes[0], 1), stride=1,
                      padding=(kernel_sizes[0] // 2, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )

        self.branch3 = nn.Sequential(
            Conv(in_channels, mid_channels, k=1, act=act),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_sizes[1]), stride=1,
                      padding=(0, kernel_sizes[1] // 2), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_sizes[1], 1), stride=1,
                      padding=(kernel_sizes[1] // 2, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )

        self.branch4 = nn.Sequential(
            Conv(in_channels, mid_channels, k=1, act=act),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_sizes[1]), stride=1,
                      padding=(0, kernel_sizes[1] // 2), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_sizes[1], 1), stride=1,
                      padding=(kernel_sizes[1] // 2, 0), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(1, kernel_sizes[0]), stride=1,
                      padding=(0, kernel_sizes[0] // 2), bias=False),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=(kernel_sizes[0], 1), stride=1,
                      padding=(kernel_sizes[0] // 2, 0), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )

        self.fusion_conv = Conv(mid_channels * 4, out_channels, k=1, act=act)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        out = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.fusion_conv(out)
        return out

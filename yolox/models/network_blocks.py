#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .coord_conv import CoordConv


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="hswish", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "hswish":
        module = nn.Hardswish(inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

def channel_shuffle(x, groups=2):
    """Channel Shuffle"""

    batchsize, num_channels, height, width = x.data.size()
 
    channels_per_group = num_channels // groups
 
    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
 
    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
 
    # flatten
    x = x.view(batchsize, -1, height, width)
    
    return x

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> hswish/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="hswish", no_act=False
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)
        self.no_act = no_act

    def forward(self, x):
        if self.no_act:
            return self.bn(self.conv(x))
        else:
            return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="hswish",no_depth_act=True):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
            no_act = no_depth_act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="hswish",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    "Residual layer with `in_channels` inputs."

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act="lrelu"
        )
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act="lrelu"
        )

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="hswish"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="hswish",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)

class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, depthwise=False, act="hswish"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.conv = Conv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)

class ShuffleV2DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, c_ratio=0.5, groups=2, act="hswish"):
        super().__init__()
        self.groups = groups
        self.l_channels = int(in_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels
        
        self.dwconv_l = DWConv(in_channels, self.l_channels,ksize=3,stride=2,act=act,no_depth_act=True)
        self.conv_r1 = BaseConv(in_channels, self.r_channels,ksize=1,stride=1,act=act)
        self.dwconv_r = DWConv(self.r_channels,self.o_r_channels,ksize=3,stride=2,act=act,no_depth_act=True)

    def forward(self, x):
        out_l = self.dwconv_l(x)

        out_r = self.conv_r1(x)
        out_r = self.dwconv_r(out_r)
        x = torch.cat((out_l, out_r), dim=1)
        return channel_shuffle(x,self.groups)

#TODO:Add SE Block Support
class ShuffleV2Basic(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3,stride=1, c_ratio=0.5, groups=2, act="hswish",type=""):
        super().__init__()
        self.in_channels = in_channels
        self.l_channels = int(in_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels

        self.groups = groups
        self.conv_r1 = BaseConv(self.r_channels, self.o_r_channels, ksize=1, stride=stride, act=act)
        self.dwconv_r = DWConv(self.o_r_channels, self.o_r_channels,ksize=ksize, stride=stride, act=act, no_depth_act=True)

    def forward(self, x):
        x_l = x[:, :self.l_channels, :, :]
        x_r = x[:, self.l_channels:, :, :]
        out_r = self.conv_r1(x_r)
        out_r = self.dwconv_r(out_r)

        x = torch.cat((x_l, out_r), dim=1)

        return channel_shuffle(x,self.groups)

class ShuffleV2Reduce(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, c_ratio=0.5, groups=2, act="hswish"):
        super().__init__()
        self.in_channels = in_channels
        self.l_channels = int(out_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels

        self.groups = groups
        self.conv_r1 = BaseConv(self.r_channels, self.o_r_channels, ksize=1, stride=stride, act=act)
        self.dwconv_r = DWConv(self.o_r_channels, self.o_r_channels,ksize=ksize, stride=stride, act=act, no_depth_act=True)

    def forward(self, x):
        
        x = channel_shuffle(x,self.groups)
        
        x_l = x[:, :self.l_channels, :, :]
        x_r = x[:, self.l_channels:, :, :]
        out_r = self.conv_r1(x_r)
        out_r = self.dwconv_r(out_r)
        
        x = torch.cat((x_l, out_r), dim=1)
        
        return channel_shuffle(x,self.groups)

class ShuffleV2ReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, c_ratio=0.5, repeat=2, groups=2, act="hswish",type=""):
        super().__init__()
        # self.conv1 = DWConv(in_channels, out_channels, ksize=ksize)
        self.conv1 = ShuffleV2Reduce(in_channels, out_channels, ksize=ksize, c_ratio=c_ratio, groups=groups, act=act)
        self.shuffle_blocks_list = []

        for _ in range(repeat):
            self.shuffle_blocks_list.append(ShuffleV2Basic(out_channels, out_channels, ksize, act=act))
        self.shuffle_blocks = nn.Sequential(*self.shuffle_blocks_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle_blocks(x)

        return x
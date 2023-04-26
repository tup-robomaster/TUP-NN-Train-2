#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="hswish",no_depth_act=True,pconv_groups=1):
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
            in_channels, out_channels, ksize=1, stride=1, groups=pconv_groups, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class PConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="hswish",no_depth_act=True,pconv_groups=1):
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
            in_channels, out_channels, ksize=1, stride=1, groups=pconv_groups, act=act
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
    def __init__(self, in_channels, out_channels, c_ratio=0.5, groups=2, act="hswish", use_rep=False):
        super().__init__()
        self.groups = groups
        self.l_channels = int(in_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels
        dwconv = RepDWConv if use_rep else DWConv
        
        self.dwconv_l = dwconv(in_channels, self.l_channels,ksize=3,stride=2,act=act,no_depth_act=True)
        self.conv_r1 = BaseConv(in_channels, self.r_channels,ksize=1,stride=1,act=act)
        self.dwconv_r = dwconv(self.r_channels,self.o_r_channels,ksize=3,stride=2,act=act,no_depth_act=True)

    def forward(self, x):
        out_l = self.dwconv_l(x)

        out_r = self.conv_r1(x)
        out_r = self.dwconv_r(out_r)
        x = torch.cat((out_l, out_r), dim=1)
        return channel_shuffle(x,self.groups)

#TODO:Add SE Block Support
class ShuffleV2Basic(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3,stride=1, c_ratio=0.5, groups=2, act="hswish",use_rep=False):
        super().__init__()
        self.in_channels = in_channels
        self.l_channels = int(in_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels
        dwconv = RepDWConv if use_rep else DWConv
        
        self.groups = groups
        self.conv_r1 = BaseConv(self.r_channels, self.o_r_channels, ksize=1, stride=stride, act=act)
        self.dwconv_r = dwconv(self.o_r_channels, self.o_r_channels,ksize=ksize, stride=stride, act=act, no_depth_act=True)

    def forward(self, x):
        x_l = x[:, :self.l_channels, :, :]
        x_r = x[:, self.l_channels:, :, :]
        out_r = self.conv_r1(x_r)
        out_r = self.dwconv_r(out_r)

        x = torch.cat((x_l, out_r), dim=1)

        return channel_shuffle(x,self.groups)

class ShuffleV2Reduce(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, c_ratio=0.5, groups=2, act="hswish",use_rep=False):
        super().__init__()
        self.in_channels = in_channels
        self.l_channels = int(out_channels * c_ratio)
        self.r_channels = in_channels - self.l_channels
        self.o_r_channels = out_channels - self.l_channels
        dwconv = RepDWConv if use_rep else DWConv
        
        self.groups = groups
        self.conv_r1 = BaseConv(self.r_channels, self.o_r_channels, ksize=1, stride=stride, act=act)
        self.dwconv_r = dwconv(self.o_r_channels, self.o_r_channels,ksize=ksize, stride=stride, act=act, no_depth_act=True)

    def forward(self, x):
        
        x = channel_shuffle(x,self.groups)
        
        x_l = x[:, :self.l_channels, :, :]
        x_r = x[:, self.l_channels:, :, :]
        out_r = self.conv_r1(x_r)
        out_r = self.dwconv_r(out_r)
        
        x = torch.cat((x_l, out_r), dim=1)
        
        return channel_shuffle(x,self.groups)

class ShuffleV2ReduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, c_ratio=0.5, repeat=2, groups=2, act="hswish",use_rep=False):
        super().__init__()
        # self.conv1 = DWConv(in_channels, out_channels, ksize=ksize)
        self.conv1 = ShuffleV2Reduce(in_channels, out_channels, ksize=ksize, c_ratio=c_ratio, groups=groups, act=act, use_rep=use_rep)
        self.shuffle_blocks_list = []

        for _ in range(repeat):
            self.shuffle_blocks_list.append(ShuffleV2Basic(out_channels, out_channels, ksize, act=act, use_rep=use_rep))
        self.shuffle_blocks = nn.Sequential(*self.shuffle_blocks_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle_blocks(x)

        return x

class RepDWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="hswish", no_depth_act=True, pconv_groups=1):
        super().__init__()
        self.dconv = RepBaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
            no_act=no_depth_act
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=pconv_groups, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)
    
class RepBaseConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, ksize,
                 stride=1, groups=1, dilation=1, act="hswish", deploy=False, use_se=False,no_act=False):
        super(RepBaseConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.no_act = no_act
        padding = (ksize - 1) // 2
        assert ksize == 3
        assert padding == 1
        padding_11 = padding - ksize // 2

        self.act = get_activation(act, inplace=True)

        if use_se:
            #   Note that RepVGG-D2se uses SE before nonlinearity. But RepVGGplus models uses SE after nonlinearity.
            self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=ksize, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = self.conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)


    def forward(self, inputs):
        if self.no_act:
            if hasattr(self, 'rbr_reparam'):
                return self.se(self.rbr_reparam(inputs))
            else:
                if self.rbr_identity is None:
                    id_out = 0
                else:
                    id_out = self.rbr_identity(inputs)
                    
                return self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        else:
            if hasattr(self, 'rbr_reparam'):
                return self.act(self.se(self.rbr_reparam(inputs)))
            else:
                if self.rbr_identity is None:
                    id_out = 0
                else:
                    id_out = self.rbr_identity(inputs)
                return self.act(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def conv_bn(self,in_channels, out_channels, kernel_size, stride, padding, groups=1):
        result = nn.Sequential()
        result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
        return result

    #   Optional. This may improve the accuracy and facilitates quantization in some cases.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()
    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class SEBlock(nn.Module):
    
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = torch.nn.F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x
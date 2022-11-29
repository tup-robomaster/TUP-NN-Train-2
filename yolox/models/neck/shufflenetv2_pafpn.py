#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..network_blocks import BaseConv, CSPLayer, DWConv, ShuffleV2Basic, ShuffleV2ReduceBlock

class ShuffleNetV2PAFPN(nn.Module):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        num_extra_level (int): Number of extra conv layers for more feature levels.
            Default: 0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        activation (str): Activation layer name.
            Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        use_depthwise=False,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        num_extra_level=0,
        upsample_cfg=dict(scale_factor=2, mode="bilinear"),
        activation="hswish",
    ):
        super(ShufflePAFPN, self).__init__()
        assert num_extra_level >= 0
        assert num_blocks >= 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        # conv = DepthwiseConvModule if use_depthwise else ConvModule
        Conv = DWConv if use_depthwise else BaseConv
        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(
                BaseConv(
                    in_channels=in_channels[idx],
                    out_channels=out_channels,
                    ksize=1,
                    stride=1,
                    act=self.activation,
                )
            )
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.top_down_blocks.append(
                ShuffleV2ReduceBlock(
                    out_channels * 2,
                    out_channels,
                    ksize=kernel_size,
                    act=activation
                )
            )

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsamples.append(
                Conv(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride=2,
                    act=self.activation,
                )
            )
            self.bottom_up_blocks.append(
                ShuffleV2ReduceBlock(
                    out_channels * 2,
                    out_channels,
                    ksize=kernel_size,
                    act=activation
                )
            )

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.
        Returns:
            tuple[Tensor]: multi level features.
        """
        assert len(inputs) == len(self.in_channels)
        # Reduce layers
        inputs = [
            reduce(input_x) for input_x, reduce in zip(inputs, self.reduce_layers)
        ]
        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]

            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1)
            )
            inner_outs.insert(0, inner_out)
        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsamples[idx](feat_low)
            out = self.bottom_up_blocks[idx](
                torch.cat([downsample_feat, feat_height], 1)
            )
            outs.append(out)
        # outs = []
        # feat_low = []
        # for idx in range(len(self.in_channels) - 1):
        #     if (idx == 0):
        #         feat_low = inner_outs[0]
        #     else:
        #         feat_low = outs[-1]
        #     feat_height = inner_outs[idx + 1]
        #     downsample_feat = self.downsamples[idx](feat_low)
        #     out = self.bottom_up_blocks[idx](
        #         torch.cat([downsample_feat, feat_height], 1)
        #     )
        #     outs.append(out)

        return tuple(outs)

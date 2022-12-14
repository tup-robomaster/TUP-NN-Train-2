#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from ..network_blocks import BaseConv, CSPLayer, DWConv


class CrossPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):  
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        #P0
        self.reduce_conv0_0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.reduce_conv0_1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p0 = CSPLayer(
            int(in_channels[2] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat



        #P2
        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.reduce_conv2 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p2 = CSPLayer(
            int(in_channels[1] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )



        #P1
        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p1 = CSPLayer(
            int(in_channels[2] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = input
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        f_out0 = self.reduce_conv0_0(x0)
        f_out0_c3p1 = self.reduce_conv0_1(f_out0)
        f_out0_c3p1 = self.upsample(f_out0_c3p1)
        
        f_out2 = x2
        f_out2_c3p1 = self.bu_conv2(f_out2)
        c3p1_in = torch.cat([f_out2_c3p1, x1, f_out0_c3p1], 1)
        pan_out1 = self.C3_p1(c3p1_in)

        f_c3p1_c3p2 = self.reduce_conv2(pan_out1)
        f_c3p1_c3p2 = self.upsample(f_c3p1_c3p2)
        c3p2_in = torch.cat([f_out2, f_c3p1_c3p2],1)
        pan_out2 = self.C3_p2(c3p2_in)

        f_c3p1_c3p0 = self.bu_conv1(pan_out1)
        c3p0_in = torch.cat([f_c3p1_c3p0, f_out0],1)
        pan_out0 = self.C3_p0(c3p0_in)

        # out_features = input
        # features = [out_features[f] for f in self.in_features]
        # [x2, x1, x0] = features
        # fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # f_out0 = self.upsample(fpn_out0)  # 512/16
        # f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        # f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        # fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # f_out1 = self.upsample(fpn_out1)  # 256/8
        # f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        # pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        # p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

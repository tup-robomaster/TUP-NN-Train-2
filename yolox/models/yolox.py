#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from yolox.models.backbone.darknet import CSPDarknet

from .head.yolo_head import YOLOXHead
from .neck.yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, neck=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = CSPDarknet()
        if neck is None:
            neck = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        backbone_outs = self.backbone(x)
        fpn_outs = self.neck(backbone_outs)

        if self.training:
            assert targets is not None
            loss, reg_loss, conf_loss, cls_loss, colors_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "reg_loss": reg_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "colors_loss":colors_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from pickletools import int4
import cv2

import math
from loguru import logger

import numpy as np

import torch
from torch._C import device
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou
from yolox.utils.boxes import min_rect
from ..losses import PolyIOUloss,WingLoss,FocalLoss

from ..network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_apexes,
        num_classes,
        num_colors,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()
        self.n_anchors = 1
        self.num_apexes = num_apexes
        self.num_classes = num_classes
        self.num_colors = num_colors
        self.decode_in_inference = True  # for deploy, set to False
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.in_channels = in_channels
        self.stem_channels = [int(256 * width), int(256 * width), int(256 * width)]
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv
        #
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i]),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            #Building Classification Convolution Layer
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=5,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=5,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            #Building Region Convolution Layer
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=5,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=5,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            #Building Classification Predict Layer
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * (self.num_classes + self.num_colors),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            #Building Region Predict Layer
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.num_apexes * 2,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
        #TODO:根据样本数量调整alpha权值
        self.use_l1 = False
        self.use_distill = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.bcewithlog_loss_cls = nn.BCEWithLogitsLoss(pos_weight=self.alpha_cls, reduction="none")
        # self.bcewithlog_loss_colors = nn.BCEWithLogitsLoss(pos_weight=self.alpha_cls_colors, reduction="none")
        self.bcewithlog_loss_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.bcewithlog_loss_colors = nn.BCEWithLogitsLoss(reduction="none")
        # self.focal_loss_obj = FocalLoss(alpha=0.25, gamma=2)
        # self.focal_loss_cls = FocalLoss(alpha=self.alpha_cls, gamma=2, num_classes=self.num_classes)
        # self.focal_loss_colors = FocalLoss(alpha=self.alpha_cls_colors, gamma=2, num_classes=self.num_colors)
        self.l1 = nn.L1Loss(reduction="none")
        self.wing_loss = WingLoss()
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)


            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, self.num_apexes * 2, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, self.num_apexes * 2
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)
        if self.training:
            if not self.use_distill:
                return self.get_losses(
                    imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    dtype=xin[0].dtype,
                )
            else:
                return self.get_losses_distill(
                    imgs,
                    x_shifts,
                    y_shifts,
                    expanded_strides,
                    labels,
                    torch.cat(outputs, 1),
                    origin_preds,
                    dtype=xin[0].dtype,
                )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            # print(output.shape)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs
    #Transform
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 1 + self.num_apexes * 2 + self.num_classes + self.num_colors
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            #Generate grid
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid
        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)

        for i in range(self.num_apexes):
            output[..., 2 * i:2 * (i + 1)] = (output[..., 2 * i:2 * (i + 1)] + grid) * stride

        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        for i in range(self.num_apexes):
            outputs[..., 2 * i:2 * (i + 1)] = (outputs[..., 2 * i:2 * (i + 1)] + grids) * strides
        return outputs

    def get_losses_distill(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):

        # # Cut feature map into bbox,obj,color,cls
        bbox_preds = outputs[:, :, :self.num_apexes * 2].contiguous()  # [batch, n_anchors_all, self.num_apexes * 2]
        obj_preds = outputs[:, :, self.num_apexes * 2].contiguous()  # [batch, n_anchors_all, 1]
        color_preds = outputs[:, :, self.num_apexes * 2 + 1 :self.num_apexes * 2 + 1 + self.num_colors].contiguous()  # [batch, n_anchors_all, n_color]
        cls_preds = outputs[:, :, self.num_apexes * 2 + 1 + self.num_colors:].contiguous()  # [batch, n_anchors_all, n_cls]

        bbox_teacher = labels[:, :, :self.num_apexes * 2].contiguous()  # [batch, n_anchors_all, self.num_apexes * 2]
        obj_teacher = labels[:, :, self.num_apexes * 2].contiguous()  # [batch, n_anchors_all, 1]
        color_teacher = labels[:, :, self.num_apexes * 2 + 1:self.num_apexes * 2 + 1 + self.num_colors].contiguous()  # [batch, n_anchors_all, n_color]
        cls_teacher = labels[:, :, self.num_apexes * 2 + 1 + self.num_colors:].contiguous()  # [batch, n_anchors_all, n_cls]

        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        # expanded_strides = torch.cat(expanded_strides, 1)
        # if True:
        #     origin_preds = torch.cat(origin_preds, 1)

        gt_masks = (obj_teacher > 0.3)
        num_postive = (int)(obj_teacher.sum())
        l1_targets = []

        loss_reg = (
        self.wing_loss(bbox_preds[gt_masks], bbox_teacher[gt_masks])
        ).sum()  / num_postive

        loss_obj = (
            self.bcewithlog_loss(obj_preds, obj_teacher)
        ).sum()  / num_postive

        loss_cls = (
            self.bcewithlog_loss_cls(cls_preds[gt_masks].view(-1, self.num_classes), cls_teacher[gt_masks].view(-1, self.num_classes))
        ).sum() / num_postive

        loss_colors = (
            self.bcewithlog_loss_colors(
                color_preds[gt_masks].view(-1, self.num_colors), color_teacher[gt_masks].view(-1, self.num_colors)
            )
        ).sum() / num_postive

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(bbox_preds[gt_masks].view(-1, self.num_apexes * 2), bbox_teacher[gt_masks].view(-1, self.num_apexes * 2))
            ).sum() / num_postive
        else:
            loss_l1 = 0.0

        reg_weight = 100
        conf_weight = 1
        clr_weight = 1
        cls_weight = 1
        loss = reg_weight * loss_reg + conf_weight * loss_obj + cls_weight * loss_cls  + clr_weight * loss_colors + 0.1 * loss_l1

        return (
            loss,
            reg_weight * loss_reg,
            conf_weight * loss_obj,
            cls_weight * loss_cls,
            clr_weight * loss_colors,
            0.1 * loss_l1,
            1,
        )


    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # Cut feature map into bbox,obj,color,cls
        bbox_preds = outputs[:, :, :self.num_apexes * 2]  # [batch, n_anchors_all, self.num_apexes * 2]
        obj_preds = outputs[:, :, self.num_apexes * 2].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        color_preds = outputs[:, :, self.num_apexes * 2 + 1:self.num_apexes * 2 + 1 + self.num_colors]  # [batch, n_anchors_all, n_color]
        cls_preds = outputs[:, :, self.num_apexes * 2 + 1 + self.num_colors:]  # [batch, n_anchors_all, n_cls]

        # Calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        # print("+++")
        # print(nlabel)
        # print("---")
        # print(outputs.shape)
        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        colors_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0
        #Label format:[Class, Reg]
        #Travel all labels for all batches
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0: 
                cls_target = outputs.new_zeros((0, self.num_classes))
                colors_target = outputs.new_zeros((0, self.num_colors))
                reg_target = outputs.new_zeros((0, self.num_apexes * 2))
                l1_target = outputs.new_zeros((0, self.num_apexes * 2))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:]
                #Get ground true classes and color
                gt_classes = labels[batch_idx, :num_gt,0] % self.num_classes
                gt_colors = labels[batch_idx, :num_gt,0] // self.num_classes
                #Get all bbox preds
                bboxes_preds_per_image = bbox_preds[batch_idx]
                #Generate rect bbox for apexs
                gt_rect_bboxes_per_image = min_rect(gt_bboxes_per_image)
                rect_bboxes_preds_per_image = min_rect(bboxes_preds_per_image)
                (
                    gt_matched_classes,
                    gt_matched_colors,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.get_assignments(
                    batch_idx,                      #Batch
                    num_gt,                         #Number of ground true
                    total_num_anchors,              #Total number of anchors
                    gt_rect_bboxes_per_image,       #Ground True classes per image
                    gt_classes,                     #Ground True classes
                    gt_colors,
                    rect_bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    color_preds,
                    bbox_preds,
                    obj_preds,
                    labels,
                    imgs,
                )
                try:
                    (
                        gt_matched_classes,
                        gt_matched_colors,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,              #Batch
                        num_gt,                 #Number of ground true
                        total_num_anchors,      #Total number of anchors
                        gt_rect_bboxes_per_image,    #Ground True classes per image
                        gt_classes,             #Ground True classes
                        gt_colors,
                        rect_bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        color_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    # torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        gt_matched_colors,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(
                        batch_idx,              #Batch
                        num_gt,                 #Number of ground true
                        total_num_anchors,      #Total number of anchors
                        gt_rect_bboxes_per_image,    #Ground True classes per image
                        gt_classes,             #Ground True classes
                        gt_colors,
                        rect_bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        color_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()  
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                colors_target = F.one_hot(
                    gt_matched_colors.to(torch.int64), self.num_colors
                ) * pred_ious_this_matching.unsqueeze(-1)
                # print(cls_target)
                # print(pred_ious_this_matching.unsqueeze(-1))
                # cls_target = gt_matched_classes.to(torch.int64) * pred_ious_this_matching.unsqueeze(-1)
                # colors_target = gt_matched_colors.to(torch.int64) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, self.num_apexes * 2)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )


            cls_targets.append(cls_target)
            colors_targets.append(colors_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        colors_targets = torch.cat(colors_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        num_fg = max(num_fg, 1)

        loss_reg = (
            self.wing_loss(bbox_preds.view(-1, self.num_apexes * 2)[fg_masks], reg_targets)
        ).sum() / num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg

        loss_cls = (
            self.bcewithlog_loss_cls(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        )
        # print(loss_cls.shape)
        loss_cls = loss_cls.sum() / num_fg

        loss_colors = (
            self.bcewithlog_loss_colors(
                color_preds.view(-1, self.num_colors)[fg_masks], colors_targets
            )
        ).sum() / num_fg
        # print(cls_target)

        # loss_obj = (
        #     self.focal_loss_obj(obj_preds.view(-1, 1), obj_targets)
        # ).sum() / num_fg

        # loss_cls = (
        #     self.focal_loss_cls(
        #         cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
        #     )
        # ).sum() / num_fg

        # loss_colors = (
        #     self.focal_loss_colors(
        #         color_preds.view(-1, self.num_colors)[fg_masks], colors_targets
        #     )
        # ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, self.num_apexes * 2)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 80
        conf_weight = 1.5
        clr_weight = 1
        cls_weight = 1
        loss = reg_weight * loss_reg + conf_weight * loss_obj + cls_weight * loss_cls  + clr_weight * loss_colors + loss_l1

        return (
            loss,
            reg_weight * loss_reg,
            conf_weight * loss_obj,
            cls_weight * loss_cls,
            clr_weight * loss_colors,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    # def get_losses_transfer_learning:():
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        for i in range(self.num_apexes):
            l1_target[:, 2 * i] = gt[:, 2 * i] / stride - x_shifts
            l1_target[:, 2 * i + 1] = gt[:,2 * i + 1] / stride - y_shifts
            # l1_target[:, 0] = gt[:, 0] / stride - x_shifts
            # l1_target[:, 1] = gt[:, 1] / stride - y_shifts
            # l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
            # l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        gt_colors,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        color_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):
        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")   
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            gt_colors = gt_colors.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        #Reduce the area of mathched anchors for dynamic k matching
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # print(fg_mask.sum())
        #Reduce the anchor area
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        # print(bboxes_preds_per_image)
        # print(gt_bboxes_per_image)
        # print("="*50)
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        color_preds_ = color_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()
        
        #Caculating cost
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)


        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        gt_colors_per_image = (
            F.one_hot(gt_colors.to(torch.int64), self.num_colors)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, color_preds_, obj_preds_ = cls_preds_.cpu(), color_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            color_preds_ = (
                color_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            # pair_wise_colors_loss = F.binary_cross_entropy(
            #     color_preds_.sqrt_(), gt_colors_per_image, reduction="none"
            # ).sum(-1)
            # pair_wise_cls_loss = F.binary_cross_entropy(
            #     cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            # ).sum(-1)
            # print(colors_preds_.sqrt_())
            pair_wise_colors_loss = F.binary_cross_entropy_with_logits(
                color_preds_.sqrt_(), gt_colors_per_image, reduction="none"
            ).sum(-1)
            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

            # print(pair_wise_cls_loss.shape)
            # print(pair_wise_colors_loss.shape)
        del cls_preds_, color_preds_

        cost = (
            0.5 * pair_wise_cls_loss
            + 0.5 * pair_wise_colors_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )
        #-----------------------------------------------------------
        #Dynamic K matching
        (
            num_fg,
            gt_matched_classes,
            gt_matched_colors,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, gt_colors, num_gt, fg_mask)
        del pair_wise_cls_loss, pair_wise_colors_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            gt_matched_colors = gt_matched_colors.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            gt_matched_colors,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        """
        Reduce the area of mathched anchors for dynamic k matching
        """
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )
        #--------------------Caculating Ground True----------------------#  
        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)
        # print(bbox_deltas.shape)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all        #If one or more than one condition is satisfied

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, gt_colors, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        # print(ious_in_boxes_matrix)
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]
        gt_matched_colors = gt_colors[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, gt_matched_colors, pred_ious_this_matching, matched_gt_inds


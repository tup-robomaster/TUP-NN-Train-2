#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from shapely.geometry import Polygon, MultiPoint, polygon  # 多边形


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=2, reduction="none"):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(FocalLoss, self).__init__()
        self.reduction = reduction
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        logpt = - F.cross_entropy(preds, labels)
        pt = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        loss = self.alpha * focal_loss
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class PolyIOUloss(nn.Module):
    """
        For Caculating iou loss between two polygons
    """

    def __init__(self, reduction="none", loss_type="iou"):
        super(PolyIOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, preds, targets):
        ious = []
        if self.loss_type == "giou":
            gious = []
        for pred, target in zip(preds, targets):
            pred = pred.reshape(4, 2)
            target = target.reshape(4, 2)

            pred_poly = Polygon(pred).convex_hull
            target_poly = Polygon(target).convex_hull
            union_poly = Polygon(torch.cat((pred, target))).convex_hull

            if self.loss_type == "iou":
                if not pred_poly.intersects(target_poly):
                    iou = 0
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area
                ious.append(iou)

            elif self.loss_type == "giou":
                if not pred_poly.intersects(target_poly):
                    iou = 0
                    giou = -1
                else:
                    iou = pred_poly.intersection(target_poly).area / union_poly.area
                    giou = iou - \
                        ((union_poly.area - pred_poly.intersection(target_poly).area) / union_poly.area)

                ious.append(iou)
                gious.append(giou)

        ious = torch.tensor(ious)
        if self.loss_type == "giou":
            gious = torch.tensor(gious)

        if self.loss_type == "iou":
            loss = 1 - ious
        elif self.loss_type == "giou":
            loss = 1 - gious
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn
import torch
import random

from yolox.exp import Exp as MyExp
import torch.distributed as dist


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.25
        '''
        Size must be a multiple of 64.
        '''
        # self.input_size = (640, 640)
        # self.test_size = (640, 640)
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        self.scale = (0.5, 2)
        self.multiscale_range = 3
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.enable_mixup = False

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, ShuffleV2GhostPAN, YOLOXHead
            strides = [8, 16, 32]
            # strides = [16, 32, 64]
            # in_channels = [256, 512, 1024]
            # in_channels = [48, 96, 192]
            # in_channels = [48, 96, 192]
            # in_channels = [116, 232, 464]
            in_channels = [64, 128, 256]

            # in_channels_head = [96, 96, 96]
            # in_channels_head = [128, 128, 128]
            in_channels_head = [128, 128, 128]
            # in_channels_head = [96, 96, 96]
            # in_features=("stage2", "stage3", "stage4")
            in_features=("stage2", "stage3", "stage4")
            # NANO model use depthwise = True, which is main difference.
            backbone = ShuffleV2GhostPAN(self.depth, self.width, in_features, in_channels=in_channels, depthwise=True, act=self.act)
            head = YOLOXHead(self.num_apexes, self.num_classes, self.num_colors, self.width, strides, in_channels=in_channels_head, depthwise=True, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
    
    # def random_resize(self, data_loader, epoch, rank, is_distributed):
    #     tensor = torch.LongTensor(2).cuda()

    #     if rank == 0:
    #         size_factor = self.input_size[1] * 1.0 / self.input_size[0]
    #         if not hasattr(self, 'random_size'):
    #             min_size = int(self.input_size[0] / 64) - self.multiscale_range
    #             max_size = int(self.input_size[0] / 64) + self.multiscale_range
    #             self.random_size = (min_size, max_size)
    #         size = random.randint(*self.random_size)
    #         size = (int(64 * size), 64 * int(size * size_factor))
    #         tensor[0] = size[0]
    #         tensor[1] = size[1]

    #     if is_distributed:
    #         dist.barrier()
    #         dist.broadcast(tensor, 0)

    #     input_size = (tensor[0].item(), tensor[1].item())
    #     return input_size

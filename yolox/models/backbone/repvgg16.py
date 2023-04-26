from torch import nn
import torch

from yolox.models import coord_conv

from ..network_blocks import BaseConv, Focus, DWConv, BaseConv, ShuffleV2DownSampling, ShuffleV2Basic,RepBaseConv

class RepVGG16(nn.Module):
    def __init__(
        self,
        channels,
        out_features=("stage2", "stage3", "stage4"),
        act="silu",
    ):
        super().__init__()
        stage_unit_repeat = [2,2,3,3,3]
        self.channels = []
        self.out_features = out_features
        base_channels = channels

        self.stage0_list = [RepBaseConv(3, 16, ksize=3, stride=2, act=act)]
        self.stage0_list.append(RepBaseConv(16, 16, ksize=3, stride=1, act=act))
        self.stage0 = nn.Sequential(*self.stage0_list)

        self.stage1_list = [RepBaseConv(16, 32, ksize=3, stride=2, act=act)]
        for _ in range(stage_unit_repeat[1] - 1):
            self.stage1_list.append(RepBaseConv(32, 32, ksize=3, stride=1, act=act))
        self.stage1 = nn.Sequential(*self.stage1_list)

        self.stage2_list = [RepBaseConv(32, base_channels[0], ksize=3, stride=2, act=act)]
        for _ in range(stage_unit_repeat[2] - 1):
            self.stage2_list.append(RepBaseConv(base_channels[0], base_channels[0], ksize=3, stride=1, act=act))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [RepBaseConv(base_channels[0], base_channels[1], ksize=3, stride=2, act=act)]
        for _ in range(stage_unit_repeat[3] - 1):
            self.stage3_list.append(RepBaseConv(base_channels[1], base_channels[1], ksize=3, stride=1, act=act))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [RepBaseConv(base_channels[1], base_channels[2], ksize=3, stride=2, act=act)]
        for _ in range(stage_unit_repeat[4] - 1):
            self.stage2_list.append(RepBaseConv(base_channels[2], base_channels[2], ksize=3, stride=1, act=act))
        self.stage4 = nn.Sequential(*self.stage4_list)

    def forward(self, x):
        outputs = {}
        # print(x.shape)
        x = self.stage0(x)
        outputs["stage0"] = x
        x = self.stage1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
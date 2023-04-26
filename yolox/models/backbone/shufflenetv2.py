from torch import nn
import torch

from yolox.models import coord_conv

from ..network_blocks import BaseConv, Focus, DWConv, BaseConv, ShuffleV2DownSampling, ShuffleV2Basic,RepDWConv

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        channels,
        out_features=("stage2", "stage3", "stage4"),
        act="silu",
        use_rep=False
    ):
        super().__init__()
        stage_unit_repeat = [2,2,2]
        self.channels = []
        self.out_features = out_features
        base_channels = channels
        # print(chann)

        self.stem_list = []
        self.stem_list.append(BaseConv(3,16,ksize=5,stride=2,act=act))
        self.stem = nn.Sequential(*self.stem_list)
        
        self.conv1 = RepDWConv(16, 32, ksize=3,stride=2,act=act) if use_rep else DWConv(16, 32, ksize=3,stride=2,act=act)

        self.stage2_list = [ShuffleV2DownSampling(32, base_channels[0], act=act, use_rep=use_rep)]
        for _ in range(stage_unit_repeat[0]):
            self.stage2_list.append(ShuffleV2Basic(base_channels[0], base_channels[0], act=act, use_rep=use_rep))
        self.stage2 = nn.Sequential(*self.stage2_list)

        self.stage3_list = [ShuffleV2DownSampling(base_channels[0], base_channels[1],act=act, use_rep=use_rep)]
        for _ in range(stage_unit_repeat[1]):
            self.stage3_list.append(ShuffleV2Basic(base_channels[1], base_channels[1], act=act, use_rep=use_rep))
        self.stage3 = nn.Sequential(*self.stage3_list)

        self.stage4_list = [ShuffleV2DownSampling(base_channels[1], base_channels[2], act=act, use_rep=use_rep)]
        for _ in range(stage_unit_repeat[2]):
            self.stage4_list.append(ShuffleV2Basic(base_channels[2], base_channels[2], act=act, use_rep=use_rep))
        self.stage4 = nn.Sequential(*self.stage4_list)

    def forward(self, x):
        outputs = {}
        # print(x.shape)
        x = self.stem(x)
        # x = self.coord(x)
        outputs["stem"] = x
        x = self.conv1(x)
        outputs["stage1"] = x
        x = self.stage2(x)
        outputs["stage2"] = x
        x = self.stage3(x)
        outputs["stage3"] = x
        x = self.stage4(x)
        outputs["stage4"] = x
        # print("stem", outputs["stem"].shape)
        # print("stage2", outputs["stage2"].shape)
        # print("stage3", outputs["stage3"].shape)
        # print("stage4", outputs["stage4"].shape)
        return {k: v for k, v in outputs.items() if k in self.out_features}
    
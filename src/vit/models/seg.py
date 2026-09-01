from typing import cast, override

import timm
import torch
from lightning import LightningModule
from timm.models import Eva
from torch import Tensor

from vit.heads import DPT, DPTSegHead
from vit.utils.state_dict import StateDictExcludeModulesMixin


class Seg(
    LightningModule,
    StateDictExcludeModulesMixin,
):
    def __init__(self):
        super().__init__()

        self.backbone = cast(
            Eva, timm.create_model("vit_small_plus_patch16_dinov3", pretrained=True)
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.dpt = DPT(
            d_model=self.backbone.num_features,
            hidden_channels=(64, 128, 256, 256),
            out_channels=64,
        )
        self.seg_head = DPTSegHead(in_channels=64)

        self._exclude_modules_from_state_dict("backbone")

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def training_step(self):
        return

    def validation_step(self):
        return

    def configure_optimizers(self):
        return

    @override
    def forward(self, input: Tensor) -> Tensor:
        _, _, H, W = input.shape
        patch_size_H, patch_size_W = self.backbone.patch_embed.patch_size
        H //= patch_size_H
        W //= patch_size_W
        with torch.no_grad():
            x = self.backbone.forward_intermediates(
                input,
                indices=[2, 5, 8, 11],
                return_prefix_tokens=True,
                norm=False,
                stop_early=True,
                intermediates_only=True,
            )
        patch_tokens, intermediate_tokens = zip(*x)
        patch_tokens = [p.permute(0, 2, 3, 1) for p in patch_tokens]
        cls_tokens = [y[:, 0] for y in intermediate_tokens]
        x = self.dpt.forward(cls_tokens, patch_tokens)
        x = self.seg_head.forward(x)
        return x

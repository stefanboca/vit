from typing import cast, override

import timm
import torch
from lightning import LightningModule
from timm.models import Eva
from torch import Tensor, nn

from vit.utils.state_dict import StateDictExcludeModulesMixin
from vit.utils.tensorboard import TensorBoardImageLoggingMixin


class Depth(nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, img_feats: Tensor) -> Tensor:
        raise NotImplementedError


class Discrimintator(nn.Module):
    def __init__(self):
        super().__init__()

    @override
    def forward(self, img_feats: Tensor, depth_feats: Tensor) -> Tensor:
        raise NotImplementedError


class AdverserialDepth(
    LightningModule,
    TensorBoardImageLoggingMixin,
    StateDictExcludeModulesMixin,
):
    def __init__(self):
        super().__init__()

        self.backbone = cast(
            Eva,
            timm.create_model("vit_large_patch16_dinov3", pretrained=True),
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.depth = Depth()
        self.discriminator = Discrimintator()

        self._exclude_modules_from_state_dict("backbone")

    @override
    def forward(self):
        pass

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

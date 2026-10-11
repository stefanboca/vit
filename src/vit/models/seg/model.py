from typing import cast, override

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from timm.models import Eva
from torch import Tensor

from vit.heads import DPT, DPTSegHead
from vit.models.seg.matcher import HungarianMatcher
from vit.utils.state_dict import StateDictExcludeModulesMixin


class SegVit(
    LightningModule,
    StateDictExcludeModulesMixin,
):
    def __init__(self, max_objects: int):
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
        self.seg_head = DPTSegHead(in_channels=64, n_classes=max_objects)

        self.matcher = HungarianMatcher()

        self._exclude_modules_from_state_dict("backbone")

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.compile
    def forward_logits(self, input: Tensor) -> Tensor:
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
        cls_tokens = [y[:, 0] for y in intermediate_tokens]
        x = self.dpt.forward(cls_tokens, list(patch_tokens))
        x = self.seg_head.forward(x)
        return x

    @override
    def training_step(self, batch: tuple[Tensor, list[Tensor]], _batch_idx) -> Tensor:
        image, masks = batch
        logits = self.forward_logits(image)
        matches = self.matcher.forward(logits, masks)
        loss = mask_loss(logits, masks, matches)

        return loss

    @override
    def validation_step(self, batch: tuple[Tensor, list[Tensor]], _batch_idx):
        pass


def mask_loss(logits, targets, matches):
    # logits: [B, Q, H, W]
    # targets[b]: [N_b, H, W]
    aligned = torch.zeros_like(logits)
    matched = torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device)

    for b, (pred_idx, target_idx) in enumerate(matches):
        pred_idx = pred_idx.to(logits.device)
        target_masks = targets[b][target_idx].to(
            device=logits.device, dtype=logits.dtype
        )
        aligned[b, pred_idx] = target_masks
        matched[b, pred_idx] = True

    bce = F.binary_cross_entropy_with_logits(logits, aligned)

    if matched.any():
        prediction = logits[matched].sigmoid().flatten(1)
        truth = aligned[matched].flatten(1)
        dice = 1 - (2 * (prediction * truth).sum(1) + 1) / (
            prediction.sum(1) + truth.sum(1) + 1
        )
        return bce + dice.mean()

    return bce

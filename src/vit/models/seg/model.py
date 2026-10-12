from colorsys import hsv_to_rgb
from pathlib import Path
from typing import cast, override

import timm
import torch
import torch.nn.functional as F
from lightning import LightningModule
from timm.models import Eva
from torch import Tensor, nn
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)
from torchvision.utils import draw_segmentation_masks

from vit.heads import DPT, DPTSegHead
from vit.models.seg.matcher import HungarianMatcher
from vit.utils.state_dict import StateDictExcludeModulesMixin
from vit.utils.tensorboard import TensorBoardImageLoggingMixin
from vit.utils.visualization import imagenet_unnorm


class SegVit(
    LightningModule,
    TensorBoardImageLoggingMixin,
    StateDictExcludeModulesMixin,
):
    def __init__(
        self, max_objects: int, log_n_images: int = 4, ckpt: Path | None = None
    ):
        super().__init__()
        self._init_image_logging(log_n_images, None)

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

        def make_metrics():
            return nn.ModuleDict(
                {
                    "foreground_iou": BinaryJaccardIndex(),
                    "dice": BinaryF1Score(),
                    "precision": BinaryPrecision(),
                    "recall": BinaryRecall(),
                }
            )

        self.train_metrics = make_metrics()
        self.val_metrics = make_metrics()
        self.test_metrics = make_metrics()

        self._exclude_modules_from_state_dict("backbone")
        if ckpt is not None:
            checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
            self.load_state_dict(checkpoint["state_dict"], strict=True)

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
        return self._step("train", batch, _batch_idx, self.train_metrics)

    @torch.no_grad
    @override
    def validation_step(self, batch: tuple[Tensor, list[Tensor]], _batch_idx) -> Tensor:
        return self._step("val", batch, _batch_idx, self.val_metrics)

    @override
    @torch.no_grad
    def test_step(self, batch: tuple[Tensor, list[Tensor]], _batch_idx) -> Tensor:
        return self._step("test", batch, _batch_idx, self.test_metrics)

    def _step(
        self,
        stage: str,
        batch: tuple[Tensor, list[Tensor]],
        batch_idx: int,
        metrics: nn.ModuleDict,
    ) -> Tensor:
        images, masks = batch
        logits = self.forward_logits(images)
        matches = self.matcher(logits, masks)
        loss, bce, dice = mask_loss(logits, masks, matches)
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=images.shape[0],
        )
        self.log(
            f"{stage}/bce_loss",
            bce,
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )
        self.log(
            f"{stage}/dice_loss",
            dice,
            on_step=True,
            on_epoch=True,
            batch_size=images.shape[0],
        )

        with torch.inference_mode():
            aligned, _ = align_masks(logits, masks, matches)
            predictions = logits > 0
            for name, metric in metrics.items():
                metric(predictions, aligned.bool())
                self.log(
                    f"{stage}/{name}",
                    metric,  # ty: ignore[invalid-argument-type]
                    on_step=True,
                    on_epoch=True,
                    batch_size=images.shape[0],
                )

            if stage in ("train", "val"):
                selection = self._images_to_log(
                    stage,
                    batch_idx,
                    torch.arange(images.shape[0], device=images.device),
                )
                if selection is not None:
                    selected = selection.batch_indices
                    image_vis = (
                        imagenet_unnorm(images[selected].float())
                        .mul(255)
                        .round()
                        .to(device="cpu", dtype=torch.uint8)
                    )
                    target_vis = aligned[selected].to(device="cpu", dtype=torch.bool)
                    prediction_vis = predictions[selected].to(device="cpu")
                    colors = mask_colors(logits.shape[1])
                    target_overlays = torch.stack(
                        [
                            overlay_masks(image, masks, colors)
                            for image, masks in zip(image_vis, target_vis, strict=True)
                        ]
                    )
                    prediction_overlays = torch.stack(
                        [
                            overlay_masks(image, masks, colors)
                            for image, masks in zip(
                                image_vis, prediction_vis, strict=True
                            )
                        ]
                    )
                    self._log_image_comparison(
                        f"{stage}/image_target_prediction",
                        (image_vis, target_overlays, prediction_overlays),
                        selection,
                    )

        return loss


def mask_colors(count: int) -> list[tuple[int, int, int]]:
    colors = []
    for i in range(count):
        red, green, blue = hsv_to_rgb((i * 0.618034) % 1, 0.85, 1)
        colors.append((round(255 * red), round(255 * green), round(255 * blue)))
    return colors


def overlay_masks(
    image: Tensor, masks: Tensor, colors: list[tuple[int, int, int]]
) -> Tensor:
    active = masks.flatten(1).any(1)
    if not active.any():
        return image
    active_indices = active.nonzero().flatten().tolist()
    return draw_segmentation_masks(
        image,
        masks[active],
        alpha=0.6,
        colors=[colors[i] for i in active_indices],
    )


def align_masks(
    logits: Tensor,
    targets: list[Tensor],
    matches: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Place each target mask in its matched channel; others remain empty."""
    aligned = torch.zeros_like(logits)
    matched = torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device)

    for b, (pred_idx, target_idx) in enumerate(matches):
        if targets[b].shape[0] > logits.shape[1]:
            raise ValueError("max_objects is smaller than the target mask count")
        pred_idx = pred_idx.to(logits.device)
        target_masks = targets[b][target_idx].to(
            device=logits.device, dtype=logits.dtype
        )
        aligned[b, pred_idx] = target_masks
        matched[b, pred_idx] = True

    return aligned, matched


def mask_loss(
    logits: Tensor,
    targets: list[Tensor],
    matches: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    aligned, matched = align_masks(logits, targets, matches)

    bce = F.binary_cross_entropy_with_logits(logits, aligned)

    if matched.any():
        prediction = logits[matched].sigmoid().flatten(1)
        truth = aligned[matched].flatten(1)
        dice_per_mask = 1 - (2 * (prediction * truth).sum(1) + 1) / (
            prediction.sum(1) + truth.sum(1) + 1
        )
        dice = dice_per_mask.mean()
    else:
        dice = logits.sum() * 0

    return bce + dice, bce, dice

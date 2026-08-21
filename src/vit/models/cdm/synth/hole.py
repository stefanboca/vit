from typing import cast, override

import timm
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from segmentation_models_pytorch.losses import DiceLoss
from timm.models import Eva
from torch import Tensor, nn
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from vit.data.byte_camera_depth import BCDSample
from vit.heads.dpt import DPT, DPTSegHead


class HoleModel(LightningModule):
    def __init__(self, log_n_images: int = 4):
        super().__init__()
        self.log_n_images = log_n_images

        self.backbone = cast(
            Eva, timm.create_model("vit_large_patch16_dinov3", pretrained=True)
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.dpt = DPT(d_model=self.backbone.num_features, batch_norm=True)
        self.seg_head = DPTSegHead(n_classes=1)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary")

        self.val_metrics = nn.ModuleDict(
            {
                "foreground_iou": BinaryJaccardIndex(),
                "dice": BinaryF1Score(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "foreground_iou": BinaryJaccardIndex(),
                "dice": BinaryF1Score(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
            }
        )

        self.register_load_state_dict_pre_hook(self._capture_load_prefix)
        self.register_load_state_dict_post_hook(self._remove_backbone_keys)
        self._register_state_dict_hook(self._remove_backbone)
        self._last_visualized_step = -1

    @override
    def training_step(self, batch: BCDSample, batch_idx: int):
        return self._step("train", batch, batch_idx, {})

    @override
    def validation_step(self, batch: BCDSample, batch_idx: int):
        return self._step("val", batch, batch_idx, self.val_metrics)

    @override
    def test_step(self, batch: BCDSample, batch_idx: int):
        self._step("test", batch, batch_idx, self.test_metrics)

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.compile
    def forward(self, input: Tensor) -> Tensor:
        _, _, H, W = input.shape
        patch_size_H, patch_size_W = self.backbone.patch_embed.patch_size
        H //= patch_size_H
        W //= patch_size_W
        with torch.no_grad():
            x = self.backbone.forward_intermediates(
                input,
                indices=[5, 11, 17, 23],
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

    @torch.compile
    def _loss(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        targets = targets.float()
        bce = self.bce_loss.forward(logits, targets)
        dice = self.dice_loss.forward(logits, targets)
        return bce + dice, bce, dice

    def _step(
        self,
        step: str,
        batch: BCDSample,
        batch_idx: int,
        metrics,
    ):
        images, targets = batch["images"], batch["hole_mask"]
        batch_size = images.shape[0]
        logits = self.forward(images)
        loss, bce, dice = self._loss(logits, targets)
        self.log(
            f"{step}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            f"{step}/bce_loss",
            bce,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            f"{step}/dice_loss",
            dice,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

        with torch.inference_mode():
            probs_bin = (logits > 0.0).long()

            for name, metric in metrics.items():
                metric(probs_bin, targets)
                self.log(
                    f"{step}/{name}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                    batch_size=batch_size,
                )

            interval = self.trainer.log_every_n_steps  # ty: ignore[unresolved-attribute]
            if (
                self.trainer.is_global_zero
                and not self.trainer.sanity_checking
                and isinstance(self.logger, TensorBoardLogger)
                and (
                    (
                        step == "train"
                        and interval > 0
                        and self.global_step > 0
                        and self.global_step % interval == 0
                        and self.global_step != self._last_visualized_step
                    )
                    or (step == "val" and batch_idx == 0)
                )
            ):
                self._last_visualized_step = step

                images = images[: self.log_n_images].to(
                    device="cpu", dtype=torch.float32
                )
                targets = targets[: self.log_n_images].to(
                    device="cpu", dtype=torch.float32
                )
                probs = (
                    logits[: self.log_n_images]
                    .sigmoid()
                    .to(device="cpu", dtype=torch.float32)
                )
                probs_bin = probs_bin[: self.log_n_images].to(
                    device="cpu", dtype=torch.float32
                )
                images.clamp_(0, 1)

                writer = self.logger.experiment
                writer.add_images(
                    f"{step}/images",
                    images,
                    global_step=self.global_step,
                )
                writer.add_images(
                    f"{step}/targets",
                    targets,
                    global_step=self.global_step,
                )
                writer.add_images(
                    f"{step}/probs",
                    probs,
                    global_step=self.global_step,
                )
                writer.add_images(
                    f"{step}/probs_bin",
                    probs_bin,
                    global_step=self.global_step,
                )

        return loss

    def _capture_load_prefix(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._load_state_dict_prefix = prefix

    def _remove_backbone_keys(self, module, incompatible_keys):
        backbone_prefix = f"{self._load_state_dict_prefix}backbone."
        incompatible_keys.missing_keys[:] = [
            key
            for key in incompatible_keys.missing_keys
            if not key.startswith(backbone_prefix)
        ]

    def _remove_backbone(self, module, state_dict, prefix, local_metadata):
        backbone_prefix = f"{prefix}backbone."
        for key in list(state_dict):
            if key.startswith(backbone_prefix):
                del state_dict[key]

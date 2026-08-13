from typing import override

import torch
from dinov3.hub.backbones import dinov3_vitl16
from dinov3.models.vision_transformer import DinoVisionTransformer
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from segmentation_models_pytorch.losses import DiceLoss
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
    def __init__(self, dinov3_vitl16_checkpoint_path: str, log_n_images: int = 4):
        super().__init__()
        self.log_n_images = 4

        self.backbone: DinoVisionTransformer = dinov3_vitl16(pretrained=False)
        self.dpt = DPT(d_model=self.backbone.embed_dim, batch_norm=True)
        self.seg_head = DPTSegHead(n_classes=1)

        dino_weights = torch.load(
            dinov3_vitl16_checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        self.backbone.load_state_dict(dino_weights)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

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

        self._register_load_state_dict_pre_hook(self._remove_backbone_keys)
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
        B, _, H, W = input.shape
        H //= self.backbone.patch_size
        W //= self.backbone.patch_size
        with torch.no_grad():
            x = self.backbone.get_intermediate_layers(
                input, n=[5, 11, 17, 23], return_class_token=True
            )
        patch_tokens, cls_tokens = zip(*x)
        patch_tokens = [p.reshape(B, H, W, -1) for p in patch_tokens]
        x = self.dpt.forward(cls_tokens, patch_tokens)  # ty: ignore[invalid-argument-type]
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

    def _remove_backbone_keys(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        backbone_prefix = f"{prefix}backbone."
        missing_keys[:] = [k for k in missing_keys if not k.startswith(backbone_prefix)]

    def _remove_backbone(self, module, state_dict, prefix, local_metadata):
        backbone_prefix = f"{prefix}backbone."
        for key in list(state_dict):
            if key.startswith(backbone_prefix):
                del state_dict[key]

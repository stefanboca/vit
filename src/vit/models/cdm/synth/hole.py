from typing import cast, override

import timm
import torch
from lightning import LightningModule
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
from vit.utils.state_dict import StateDictExcludeModulesMixin
from vit.utils.tensorboard import TensorBoardImageLoggingMixin
from vit.utils.visualization import (
    imagenet_unnorm,
)


class HoleModel(
    LightningModule,
    TensorBoardImageLoggingMixin,
    StateDictExcludeModulesMixin,
):
    def __init__(
        self,
        log_n_images: int = 4,
        val_log_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self._init_image_logging(log_n_images, val_log_indices)

        self.backbone = cast(
            Eva, timm.create_model("vit_large_patch16_dinov3", pretrained=True)
        )
        self.backbone.requires_grad_(False)
        self.backbone.eval()

        self.dpt = DPT(d_model=self.backbone.num_features, batch_norm=True)
        self.seg_head = DPTSegHead(n_classes=1)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(mode="binary")

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

    @override
    def training_step(self, batch: BCDSample, batch_idx: int):
        return self.step("train", batch, batch_idx, self.train_metrics)

    @override
    def validation_step(self, batch: BCDSample, batch_idx: int):
        return self.step("val", batch, batch_idx, self.val_metrics)

    @override
    def test_step(self, batch: BCDSample, batch_idx: int):
        self.step("test", batch, batch_idx, self.test_metrics)

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @override
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
        cls_tokens = [y[:, 0] for y in intermediate_tokens]
        x = self.dpt.forward(cls_tokens, list(patch_tokens))
        x = self.seg_head.forward(x)
        return x

    def loss(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        targets = targets.float()
        bce = self.bce_loss.forward(logits, targets)
        dice = self.dice_loss.forward(logits, targets)
        return bce + dice, bce, dice

    @torch.compile
    def forward_with_loss(
        self, input: Tensor, targets: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        logits = self.forward(input)
        loss, bce, dice = self.loss(logits, targets)
        return logits, loss, bce, dice

    def step(
        self,
        step: str,
        batch: BCDSample,
        batch_idx: int,
        metrics,
    ):
        images, targets = batch["images"], batch["holes"]
        batch_size = images.shape[0]
        logits, loss, bce, dice = self.forward_with_loss(images, targets)
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

            selection = self._images_to_log(step, batch_idx, batch["index"])
            if selection is not None:
                vis_indices = selection.batch_indices
                images_vis = images[vis_indices].to(device="cpu", dtype=torch.float32)
                targets_vis = targets[vis_indices].to(device="cpu", dtype=torch.float32)
                probs_bin_vis = probs_bin[vis_indices].to(
                    device="cpu", dtype=torch.float32
                )
                probs_vis = (
                    logits[vis_indices].sigmoid().to(device="cpu", dtype=torch.float32)
                )
                images_vis = imagenet_unnorm(images_vis)

                self._log_image_comparison(
                    f"{step}/input_target_probability_prediction",
                    (images_vis, targets_vis, probs_bin_vis, probs_vis),
                    selection,
                )

        return loss

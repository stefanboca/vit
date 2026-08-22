from typing import override

import matplotlib
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.model import DepthAnything3Net
from lightning import LightningModule
from torch import Tensor, nn
from torchmetrics.aggregation import MeanMetric

from vit.data.byte_camera_depth import BCDSample
from vit.utils.alignment import align_depth_affine_irls
from vit.utils.tensorboard import TensorBoardImageLoggingMixin
from vit.utils.visualization import (
    colorize_depth,
    imagenet_unnorm,
)


class ValueModel(LightningModule, TensorBoardImageLoggingMixin):
    depth_color_lut: Tensor

    def __init__(
        self,
        log_n_images: int = 4,
        val_log_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self._init_image_logging(log_n_images, val_log_indices)

        self.model: DepthAnything3Net = DepthAnything3.from_pretrained(
            "depth-anything/DA3MONO-LARGE", pretrained=True
        ).model
        self.model.backbone.requires_grad_(False)
        self.model.head.use_sky_head = False
        self.model.head.scratch.sky_output_conv2 = nn.Identity()
        self.model.backbone.eval()

        self.l1_loss = nn.SmoothL1Loss(reduction="none", beta=0.02)

        self.train_metrics = nn.ModuleDict(
            {"affine_abs_rel": MeanMetric(nan_strategy="ignore")}
        )
        eval_metric_names = (
            "affine_mae",
            "affine_rmse",
            "affine_abs_rel",
            "affine_delta_1_05",
            "affine_delta_1_10",
            "affine_delta_1_25",
        )
        self.val_metrics = nn.ModuleDict(
            {name: MeanMetric(nan_strategy="ignore") for name in eval_metric_names}
        )
        self.test_metrics = nn.ModuleDict(
            {name: MeanMetric(nan_strategy="ignore") for name in eval_metric_names}
        )

        cmap = matplotlib.colormaps["viridis"]
        lut = cmap(np.linspace(0, 1, 256))[:, :3]
        self.register_buffer(
            "depth_color_lut",
            torch.from_numpy(lut).float(),
            persistent=False,
        )

        self.register_load_state_dict_pre_hook(self._capture_load_prefix)
        self.register_load_state_dict_post_hook(self._remove_backbone_keys)
        self._register_state_dict_hook(self._remove_backbone)

    @override
    def training_step(self, batch: BCDSample, batch_idx: int):
        return self._step("train", batch, batch_idx, self.train_metrics)

    @override
    def validation_step(self, batch: BCDSample, batch_idx: int):
        return self._step("val", batch, batch_idx, self.val_metrics)

    @override
    def test_step(self, batch: BCDSample, batch_idx: int):
        self._step("test", batch, batch_idx, self.test_metrics)

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self.model.backbone.eval()
        return self

    @override
    def forward(self, input: Tensor) -> Tensor:
        return self.model.forward(input.unsqueeze(1))["depth"]

    def loss(
        self, preds: Tensor, targets: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            scale, shift = align_depth_affine_irls(
                preds.flatten(-2),
                targets.flatten(-2),
                valid.flatten(-2),
            )
        aligned_preds = preds * scale[..., None, None] + shift[..., None, None]
        pixel_loss = self.l1_loss.forward(aligned_preds, targets)
        loss = (pixel_loss * valid).sum() / valid.sum().clamp_min(1)
        return aligned_preds, loss

    def forward_with_loss(
        self, input: Tensor, targets: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        preds = self.forward(input)
        aligned_preds, loss = self.loss(preds, targets, valid)
        return preds, aligned_preds, loss

    @torch.inference_mode()
    def metrics(
        self,
        aligned_preds: Tensor,
        targets: Tensor,
        valid: Tensor,
        eval_metrics: bool,
    ) -> dict[str, Tensor]:
        targets = targets.float()
        valid = valid.bool()
        weight = valid.float()
        reduce_dims = tuple(range(1, aligned_preds.ndim))
        count = weight.sum(dim=reduce_dims).clamp_min(1)

        absolute_error = (aligned_preds.float() - targets).abs()
        abs_rel = (absolute_error / targets.clamp_min(1) * weight).sum(
            dim=reduce_dims
        ) / count
        values = {"affine_abs_rel": abs_rel}
        if not eval_metrics:
            return values

        values["affine_mae"] = (absolute_error * weight).sum(dim=reduce_dims) / count
        values["affine_rmse"] = (
            (absolute_error.square() * weight).sum(dim=reduce_dims) / count
        ).sqrt()

        positive_preds = aligned_preds.float().clamp_min(1e-6)
        positive_targets = targets.clamp_min(1e-6)
        ratio = torch.maximum(
            positive_preds / positive_targets,
            positive_targets / positive_preds,
        )
        for name, threshold in (
            ("affine_delta_1_05", 1.05),
            ("affine_delta_1_10", 1.10),
            ("affine_delta_1_25", 1.25),
        ):
            values[name] = ((ratio < threshold) & valid).float().sum(
                dim=reduce_dims
            ) / count

        return values

    def _step(
        self,
        step: str,
        batch: BCDSample,
        batch_idx: int,
        metrics,
    ):
        images, targets, holes = (
            batch["images"],
            batch["depth"] / 1000.0,
            batch["holes"],
        )
        batch_size = images.shape[0]
        valid = ~holes
        _preds, aligned_preds, loss = self.forward_with_loss(images, targets, valid)
        self.log(
            f"{step}/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        with torch.inference_mode():
            values = self.metrics(
                aligned_preds,
                targets,
                valid,
                eval_metrics=step != "train",
            )
            for name, metric in metrics.items():
                metric(values[name])
                self.log(
                    f"{step}/{name}",
                    metric,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=name == "affine_abs_rel",
                    batch_size=batch_size,
                )

            selection = self._images_to_log(step, batch_idx, batch["index"])
            if selection is not None:
                vis_indices = selection.batch_indices
                images_vis = images[vis_indices]
                targets_vis = targets[vis_indices]
                preds_vis = aligned_preds[vis_indices]
                valid_vis = valid[vis_indices]

                vmin, vmax = targets_vis[valid_vis].quantile(
                    targets_vis.new_tensor([0.01, 0.99])
                )

                targets_rgb = colorize_depth(
                    targets_vis,
                    valid_vis,
                    self.depth_color_lut,
                    vmin,
                    vmax,
                )
                preds_rgb = colorize_depth(
                    preds_vis,
                    None,
                    self.depth_color_lut,
                    vmin,
                    vmax,
                )

                images_vis = images_vis.to(device="cpu", dtype=torch.float32)
                targets_rgb = targets_rgb.to(device="cpu", dtype=torch.float32)
                preds_rgb = preds_rgb.to(device="cpu", dtype=torch.float32)
                images_vis = imagenet_unnorm(images_vis)

                self._log_image_comparison(
                    f"{step}/input_target_prediction",
                    (images_vis, targets_rgb, preds_rgb),
                    selection,
                )

        return loss

    def _capture_load_prefix(
        self,
        _module,
        _state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ):
        self._load_state_dict_prefix = prefix

    def _remove_backbone_keys(self, _module, incompatible_keys):
        backbone_prefix = f"{self._load_state_dict_prefix}model.backbone."
        incompatible_keys.missing_keys[:] = [
            key
            for key in incompatible_keys.missing_keys
            if not key.startswith(backbone_prefix)
        ]

    def _remove_backbone(self, _module, state_dict, prefix, _local_metadata):
        backbone_prefix = f"{prefix}model.backbone."
        for key in list(state_dict):
            if key.startswith(backbone_prefix):
                del state_dict[key]

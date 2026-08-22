from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor


@dataclass(frozen=True)
class ImageLogSelection:
    batch_indices: Tensor
    sample_indices: Tensor | None


class TensorBoardImageLoggingMixin:
    def _init_image_logging(
        self,
        log_n_images: int,
        val_log_indices: Sequence[int] | None,
    ) -> None:
        self.log_n_images = log_n_images
        self.val_log_indices = (
            tuple(val_log_indices) if val_log_indices is not None else None
        )
        self._last_visualized_step = -1

    @torch.inference_mode()
    def _images_to_log(
        self,
        step: str,
        batch_idx: int,
        batch_indices: Tensor,
    ) -> ImageLogSelection | None:
        module = cast(LightningModule, self)
        trainer = module.trainer

        selected_indices = self.val_log_indices if step == "val" else None
        vis_indices = self._select_image_indices(batch_indices, selected_indices)

        if (
            not trainer.is_global_zero
            or trainer.sanity_checking
            or not isinstance(module.logger, TensorBoardLogger)
            or vis_indices.numel() == 0
        ):
            return None

        interval = trainer.log_every_n_steps  # ty: ignore[unresolved-attribute]
        log_train = (
            step == "train"
            and interval > 0
            and module.global_step > 0
            and module.global_step % interval == 0
            and module.global_step != self._last_visualized_step
        )
        log_val = step == "val" and (selected_indices is not None or batch_idx == 0)
        if not (log_train or log_val):
            return None

        self._last_visualized_step = module.global_step
        sample_indices = (
            batch_indices[vis_indices] if selected_indices is not None else None
        )
        return ImageLogSelection(vis_indices, sample_indices)

    @torch.inference_mode()
    def _select_image_indices(
        self,
        batch_indices: Tensor,
        selected_indices: Sequence[int] | None,
    ) -> Tensor:
        if selected_indices is None:
            count = max(0, min(self.log_n_images, batch_indices.numel()))
            return torch.arange(count, device=batch_indices.device)

        wanted = batch_indices.new_tensor(selected_indices)
        return torch.where(torch.isin(batch_indices, wanted))[0]

    @torch.inference_mode()
    def _log_image_comparison(
        self,
        tag: str,
        panels: Sequence[Tensor],
        selection: ImageLogSelection,
    ) -> None:
        module = cast(LightningModule, self)
        logger = cast(TensorBoardLogger, module.logger)

        channels = max(panel.shape[1] for panel in panels)
        panels = tuple(
            panel.expand(-1, channels, -1, -1) if panel.shape[1] == 1 else panel
            for panel in panels
        )
        comparison = torch.cat(panels, dim=-2)

        if selection.sample_indices is None:
            logger.experiment.add_images(
                tag,
                comparison,
                global_step=module.global_step,
            )
            return

        for index, image in zip(selection.sample_indices.cpu().tolist(), comparison):
            logger.experiment.add_image(
                f"{tag}/{index}",
                image,
                global_step=module.global_step,
            )

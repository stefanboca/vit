import os
import secrets
from pathlib import Path
from typing import Literal

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.utilities.model_summary.model_summary import summarize


class VitTrainer(Trainer):
    def export_onnx(
        self,
        model: LightningModule,
        ckpt_path: Path | None = None,
        output_path: Path = Path("model.onnx"),
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        opset_version: int = 21,
        precision: Literal["full", "half"] = "full",
        verify: bool = True,
    ):
        """Export a checkpoint to ONNX."""
        if ckpt_path is not None:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(
                checkpoint["state_dict"],
                strict=True,
            )
        device = self.strategy.root_device
        model.to(device)
        model.eval()

        if precision == "half":
            model.half()

        print(summarize(model))

        input_sample = torch.randn(input_shape, dtype=model.dtype, device=device)  # ty: ignore[no-matching-overload]
        batch = torch.export.Dim("batch", min=1)
        model.to_onnx(
            output_path,
            input_sample,
            input_names=["input"],
            output_names=["output"],
            dynamic_shapes={"input": {0: batch}},
            opset_version=opset_version,
            external_data=False,
            dynamo=True,
            verify=verify,
        )


class VitCLI(LightningCLI):
    @staticmethod
    def subcommands() -> dict[str, set[str]]:
        return {
            **LightningCLI.subcommands(),
            "export_onnx": {"model"},
        }


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    seed = int(os.environ.get("PL_GLOBAL_SEED", secrets.randbelow(2**32)))
    VitCLI(
        trainer_class=VitTrainer,
        parser_kwargs={"parser_mode": "omegaconf"},
        seed_everything_default=seed,
    )

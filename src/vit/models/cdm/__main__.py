import secrets

import torch
from lightning.pytorch.cli import LightningCLI

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        # Lightning 2.6.5 passes None when this default is True, and
        # seed_everything(None) currently falls back to zero instead of choosing
        # a random seed. Keep each run random while recording the chosen seed in
        # the resolved CLI config.
        seed_everything_default=secrets.randbelow(2**32),
    )

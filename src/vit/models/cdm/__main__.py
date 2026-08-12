import torch
from lightning.pytorch.cli import LightningCLI

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    LightningCLI(parser_kwargs={"parser_mode": "omegaconf"})

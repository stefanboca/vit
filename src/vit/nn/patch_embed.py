import torch
from torch import Tensor, nn

from ..types import Size


class PatchEmbed(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        in_chans: int = 3,
        patch_size: Size = 16,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, d_model, kernel_size=patch_size, stride=patch_size, device=device
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Shapes:
            x: [B, C, H, W]
            returns: [B, H, W, D]
        """
        x = self.proj.forward(input)  # [B, D, H, W]
        x = x.permute([0, 2, 3, 1])  # [B, H, W, D]
        return x

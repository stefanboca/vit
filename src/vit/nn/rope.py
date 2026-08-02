from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

type NormalizeCoords = Literal["min", "max", "separate"]


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


@dataclass
class RoPECoeffs:
    sin: Tensor
    cos: Tensor

    def apply(self, input: Tensor) -> Tensor:
        pass
        input_dtype = input.dtype
        rope_dtype = self.sin.dtype
        x = input.to(dtype=rope_dtype)
        N = x.shape[-2]
        prefix = N - self.sin.shape[-2]
        assert prefix >= 0
        x_prefix = x[:, :, :prefix, :]
        x = rope_apply(
            x[:, :, prefix:, :], self.sin, self.cos
        )  # [B, head, hw, D//head]
        x = torch.cat((x_prefix, x), dim=-2)  # [B, head, N, D//head]
        x = x.to(dtype=input_dtype)
        return x


class RoPE(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        base: float = 100.0,
        normalize_coords: NormalizeCoords = "separate",
        # TODO: training randomization params
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_model % (4 * n_heads) == 0
        self.normalize_coords = normalize_coords
        self.dtype = dtype
        self.device = device

        d_head = d_model // n_heads
        self.register_buffer(
            "periods",
            base
            ** (
                2
                * torch.arange(d_head // 4, dtype=self.dtype, device=self.device)
                / (d_head // 2)
            ),
            persistent=True,
        )

    def forward(self, height: int, width: int) -> RoPECoeffs:
        coords_h = torch.arange(0.5, height, dtype=self.dtype, device=self.device)
        coords_w = torch.arange(0.5, width, dtype=self.dtype, device=self.device)
        match self.normalize_coords:
            case "min":
                min_hw = min(height, width)
                coords_h /= min_hw
                coords_w /= min_hw
            case "max":
                max_hw = max(height, width)
                coords_h /= max_hw
                coords_w /= max_hw
            case "separate":
                coords_h /= height
                coords_w /= width
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = coords.flatten(0, 1)
        coords = 2.0 * coords - 1.0

        angles = (2 * torch.pi) * coords[:, :, None] / self.periods[None, None, :]  # ty:ignore[not-subscriptable]
        angles = angles.flatten(1, 2)
        angles = angles.tile(2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        return RoPECoeffs(sin, cos)

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class RotaryEmbeddingValues:
    cos: Tensor
    sin: Tensor

    def apply(self, x: Tensor) -> Tensor:
        # x: [batch, heads, sequence, head_dim]
        cos = self.cos.to(dtype=x.dtype)
        sin = self.sin.to(dtype=x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * cos + rotated * sin


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}")
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> RotaryEmbeddingValues:
        inv_freq = cast(Tensor, self.inv_freq)
        positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
        angles = torch.outer(positions, inv_freq).repeat_interleave(2, dim=-1)
        cos = angles.cos().to(dtype=dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=dtype)[None, None, :, :]
        return RotaryEmbeddingValues(cos, sin)

    def at(
        self, position: int, device: torch.device, dtype: torch.dtype
    ) -> RotaryEmbeddingValues:
        inv_freq = cast(Tensor, self.inv_freq)
        angles = inv_freq.to(device=device) * position
        angles = angles.repeat_interleave(2, dim=-1)
        return RotaryEmbeddingValues(
            angles.cos().to(dtype=dtype)[None, None, None, :],
            angles.sin().to(dtype=dtype)[None, None, None, :],
        )

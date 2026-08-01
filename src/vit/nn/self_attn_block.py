from typing import Callable

import torch
from torch import Tensor, nn

from vit.nn.rope import RoPECoeffs

from .ffn import Mlp
from .layer_scale import LayerScale
from .self_attn import SelfAttn


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        ffn_ratio: int = 4,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        layerscale_init: float | None = None,
        sample_drop_ratio: float = 0.0,
        norm: Callable[..., nn.Module] = nn.LayerNorm,
        attn: Callable[..., nn.Module] = SelfAttn,
        ffn: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.norm1 = norm(d_model, device=device)
        self.attn = attn(
            d_model=d_model,
            n_heads=n_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim=d_model, init_values=layerscale_init, device=device)
            if layerscale_init
            else nn.Identity()
        )

        self.norm2 = norm(d_model, device=device)
        self.mlp = ffn(
            in_features=d_model,
            hidden_features=d_model * ffn_ratio,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim=d_model, init_values=layerscale_init, device=device)
            if layerscale_init
            else nn.Identity()
        )

        self.sample_drop_ratio = sample_drop_ratio

    def forward(self, input: Tensor, rope: RoPECoeffs | None = None) -> Tensor:
        if self.training and self.sample_drop_ratio > 0.0:
            raise NotImplementedError()
        else:
            x = input + self.ls1.forward(
                self.attn.forward(self.norm1.forward(input), rope)
            )
            x = x + self.ls2.forward(self.mlp.forward(self.norm2.forward(x)))
            return x

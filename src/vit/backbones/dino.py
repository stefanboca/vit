from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
from torch import Tensor, nn

from ..nn.ffn import Mlp, SwiGluFfn
from ..nn.patch_embed import PatchEmbed
from ..nn.rope import NormalizeCoords, RoPE
from ..nn.self_attn_block import SelfAttnBlock
from ..types import Size


@dataclass
class DinoOutput:
    cls_token: Tensor
    storage_tokens: Tensor
    patches: Tensor


class Dino(nn.Module):
    def __init__(
        self,
        *,
        in_chans: int = 3,
        patch_size: Size = 16,
        rope_base: float = 100.0,
        rope_normalize_coords: NormalizeCoords = "separate",
        rope_shift_coords: float | None = None,
        rope_jitter_coords: float | None = None,
        rope_rescale_coords: float | None = 2,
        rope_dtype: torch.dtype = torch.float32,
        d_model: int = 768,
        n_blocks: int = 12,
        n_heads: int = 12,
        ffn_ratio: int = 4,
        qkv_bias: bool = True,
        sample_drop_ratio: float = 0.0,
        layerscale_init: float | None = 1e-5,
        norm: Callable[..., nn.Module] = nn.LayerNorm,
        ffn: Callable[..., nn.Module] = Mlp,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 4,
        mask_k_bias: bool = True,
        untie_global_and_local_cls_norm: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, patch_size=patch_size, d_model=d_model, device=device
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model, device=device))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, n_storage_tokens, d_model, device=device)
        )

        self.rope_embed = RoPE(
            d_model=d_model,
            n_heads=n_heads,
            base=rope_base,
            normalize_coords=rope_normalize_coords,
            shift_coords=rope_shift_coords,
            jitter_coords=rope_jitter_coords,
            rescale_coords=rope_rescale_coords,
            dtype=rope_dtype,
            device=device,
        )

        self.blocks = nn.ModuleList(
            [
                SelfAttnBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_ratio=ffn_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    sample_drop_ratio=sample_drop_ratio,
                    norm=norm,
                    ffn=ffn,
                    layerscale_init=layerscale_init,
                    mask_k_bias=mask_k_bias,
                    device=device,
                )
                for _ in range(n_blocks)
            ]
        )

        self.norm = norm(d_model, device=device)

        if untie_global_and_local_cls_norm:
            # NOTE: we only have local_cls_norm so that load_state_dict with vit7b16 doesn't complain
            self.local_cls_norm = norm(d_model)

        self.mask_token = nn.Parameter(torch.empty(1, d_model, device=device))

    def forward(self, input: Tensor, masks: Tensor | None = None) -> DinoOutput:
        x = self.patch_embed.forward(input)
        B, H, W, D = x.shape
        x = x.flatten(1, 2)  # [B, H * W, D]

        cls_token = self.cls_token
        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
            )
        else:
            cls_token += 0 * self.mask_token

        x = torch.cat(
            [
                cls_token.expand([B, -1, -1]),
                self.storage_tokens.expand([B, -1, -1]),
                x,
            ],
            dim=1,
        )

        rope = self.rope_embed.forward(H, W)
        for block in self.blocks:
            x = block.forward(x, rope)
        x: Tensor = self.norm.forward(x)

        n_storage_tokens = self.storage_tokens.shape[1]
        cls_token = x[:, 0]
        storage_tokens = x[:, 1 : n_storage_tokens + 1]
        patches = x[:, n_storage_tokens + 1 :].reshape([B, H, W, D])

        return DinoOutput(cls_token, storage_tokens, patches)


def dinov3_vits16(**kwargs):
    return Dino(
        d_model=384,
        n_blocks=12,
        n_heads=6,
        **kwargs,
    )


def dinov3_vits16plus(**kwargs):
    return Dino(
        d_model=384,
        n_blocks=12,
        n_heads=6,
        ffn_ratio=6,
        ffn=SwiGluFfn,
        **kwargs,
    )


def dinov3_vitb16(**kwargs):
    return Dino(
        d_model=768,
        n_blocks=12,
        n_heads=12,
        **kwargs,
    )


def dinov3_vitl16(**kwargs):
    return Dino(
        d_model=1024,
        n_blocks=24,
        n_heads=16,
        **kwargs,
    )


def dinov3_vith16plus(**kwargs):
    return Dino(
        d_model=1280,
        n_blocks=32,
        n_heads=20,
        ffn_ratio=6,
        ffn=SwiGluFfn,
        **kwargs,
    )


def dinov3_vit7b16(**kwargs):
    return Dino(
        d_model=4096,
        n_blocks=40,
        n_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        ffn=partial(SwiGluFfn, align_to=64),
        untie_global_and_local_cls_norm=True,
        **kwargs,
    )

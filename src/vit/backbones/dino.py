from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch
from torch import Tensor, nn

from ..nn.ffn import Mlp, SwiGluFfn
from ..nn.patch_embed import PatchEmbed
from ..nn.rope import RoPE, RoPEParams
from ..nn.self_attn_block import SelfAttnBlock
from ..types import Size


@dataclass
class PosInterplateParams:
    size: int = 37
    antialias: bool = False
    offset: float | None = 0.1


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
        pos_embedding: RoPEParams | PosInterplateParams = RoPEParams(),
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
        # TODO: DINOv2 bag of channels
        super().__init__()
        self.d_model = d_model

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, patch_size=patch_size, d_model=d_model, device=device
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model, device=device))
        self.storage_tokens = nn.Parameter(
            torch.empty(1, n_storage_tokens, d_model, device=device)
        )

        if isinstance(pos_embedding, RoPEParams):
            self.rope_embed = RoPE(
                d_model=d_model,
                n_heads=n_heads,
                params=pos_embedding,
                device=device,
            )
            self.pos_interpolate_params = None
            self.pos_embed = None
        elif isinstance(pos_embedding, PosInterplateParams):
            self.rope_embed = None
            self.pos_interpolate_params = pos_embedding
            self.pos_embed = nn.Parameter(
                torch.empty(1, pos_embedding.size**2 + 1, d_model, device=device)
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

        x = torch.cat([cls_token.expand([B, -1, -1]), x], dim=1)

        rope = None
        if self.rope_embed is not None:
            rope = self.rope_embed.forward(H, W)
        elif self.pos_embed is not None:
            x = x + self._interpolate_pos_encoding(H, W).to(x.dtype)

        x = torch.cat(
            [
                x[:, :1],
                self.storage_tokens.expand([B, -1, -1]),
                x[:, 1:],
            ],
            dim=1,
        )

        for block in self.blocks:
            x = block.forward(x, rope)
        x: Tensor = self.norm.forward(x)

        n_storage_tokens = self.storage_tokens.shape[1]
        cls_token = x[:, 0]
        storage_tokens = x[:, 1 : n_storage_tokens + 1]
        patches = x[:, n_storage_tokens + 1 :].reshape([B, H, W, D])

        return DinoOutput(cls_token, storage_tokens, patches)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        storage_tokens = f"{prefix}storage_tokens"
        register_tokens = f"{prefix}register_tokens"

        if storage_tokens not in state_dict and register_tokens in state_dict:
            # rename register_tokens to storage_tokens for dinov2
            state_dict[storage_tokens] = state_dict.pop(register_tokens)

        for key in list(state_dict.keys()):
            # rewrite swiglu weights
            if ".mlp.w12." in key:
                w12 = state_dict.pop(key)
                n = w12.shape[0] // 2
                state_dict[key.replace("w12", "w1")] = w12[:n]
                state_dict[key.replace("w12", "w2")] = w12[n:]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _interpolate_pos_encoding(self, H: int, W: int) -> Tensor:
        assert self.pos_embed is not None and self.pos_interpolate_params is not None
        M = self.pos_interpolate_params.size
        N = M**2
        if N == H * W and H == W:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]

        kwargs: dict[str, Any] = {}
        if self.pos_interpolate_params.offset:
            # Historical kludge: add a small number to avoid floating point error in the interpolation, see https://github.com/facebookresearch/dino/issues/8
            # Note: still needed for backward-compatibility, the underlying operators are using both output size and scale factors
            kwargs["scale_factor"] = (
                float(H + self.pos_interpolate_params.offset) / M,
                float(W + self.pos_interpolate_params.offset) / M,
            )
        else:
            # Simply specify an output size instead of a scale factor
            kwargs["size"] = (H, W)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, M, M, self.d_model).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.pos_interpolate_params.antialias,
            **kwargs,
        )
        assert (H, W) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(
            1, H * W, self.d_model
        )
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def dinov2_vits14_reg(**kwargs):
    return Dino(
        patch_size=14,
        pos_embedding=PosInterplateParams(antialias=True, offset=0.0),
        d_model=384,
        n_blocks=12,
        n_heads=6,
        norm=partial(nn.LayerNorm, eps=1e-6),
        mask_k_bias=False,
        **kwargs,
    )


def dinov2_vitb14_reg(**kwargs):
    return Dino(
        patch_size=14,
        pos_embedding=PosInterplateParams(antialias=True, offset=0.0),
        d_model=768,
        n_blocks=12,
        n_heads=12,
        norm=partial(nn.LayerNorm, eps=1e-6),
        mask_k_bias=False,
        **kwargs,
    )


def dinov2_vitl14_reg(**kwargs):
    return Dino(
        patch_size=14,
        pos_embedding=PosInterplateParams(antialias=True, offset=0.0),
        d_model=1024,
        n_blocks=24,
        n_heads=16,
        norm=partial(nn.LayerNorm, eps=1e-6),
        mask_k_bias=False,
        **kwargs,
    )


def dinov2_vitg14_reg(**kwargs):
    return Dino(
        patch_size=14,
        pos_embedding=PosInterplateParams(antialias=True, offset=0.0),
        d_model=1536,
        n_blocks=40,
        n_heads=24,
        ffn=SwiGluFfn,
        norm=partial(nn.LayerNorm, eps=1e-6),
        mask_k_bias=False,
        **kwargs,
    )


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

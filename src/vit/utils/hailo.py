from collections.abc import Iterator
from contextlib import contextmanager
from math import log

import torch
import torch.nn.functional as F
from timm.layers import (
    apply_keep_indices_nlc,
    apply_rot_embed_cat,
    resample_abs_pos_embed,
)
from timm.models._features import feature_take_indices
from timm.models._manipulate import checkpoint
from timm.models.eva import Eva, EvaAttention
from torch import Tensor, nn


class HailoEvaAttention(EvaAttention):
    """EVA attention with an exactly weighted Hailo prefix-token layout."""

    def forward(
        self,
        x: Tensor,
        rope: Tensor | None = None,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        B, N, C = x.shape
        gate = self.gate(x).sigmoid() if self.gate is not None else None

        if self.qkv is not None:
            # timm evaluates one fused QKV projection and then unbinds it.
            # Export three equivalent projections so DFC receives independent
            # Q/K/V branches and does not have to lower the unbind/slices.
            if self.q_bias is None:
                qkv_bias = self.qkv.bias
            else:
                qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))  # ty: ignore[invalid-argument-type]
                if self.qkv_bias_separate and self.qkv.bias is not None:
                    # timm applies qkv.bias in self.qkv(x), then adds the
                    # separate biases; combine them before splitting instead.
                    qkv_bias = qkv_bias + self.qkv.bias

            q_weight, k_weight, v_weight = self.qkv.weight.chunk(3, dim=0)
            if qkv_bias is None:
                q_bias = k_bias = v_bias = None
            else:
                q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
            q = F.linear(x, q_weight, q_bias)
            k = F.linear(x, k_weight, k_bias)
            v = F.linear(x, v_weight, v_bias)
            q = q.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            k = k.reshape(B, N, self.num_heads, -1).transpose(1, 2)
            v = v.reshape(B, N, self.num_heads, -1).transpose(1, 2)
        else:
            q = self.q_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # ty: ignore[call-non-callable]
            k = self.k_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # ty: ignore[call-non-callable]
            v = self.v_proj(x).reshape(B, N, self.num_heads, -1).transpose(1, 2)  # ty: ignore[call-non-callable]

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            # timm slices the prefix from Q/K, rotates only patch tokens, then
            # concatenates the prefix back. DFC cannot allocate those token
            # slices at large resolutions. An identity rotary embedding for the
            # prefix makes rotating the complete sequence exactly equivalent.
            npt = self.num_prefix_tokens
            prefix_shape = (*rope.shape[:-2], npt, rope.shape[-1] // 2)
            identity_rope = torch.cat(
                (rope.new_zeros(prefix_shape), rope.new_ones(prefix_shape)), dim=-1
            )
            rope_with_prefix = torch.cat((identity_rope, rope), dim=-2)
            half = getattr(self, "rotate_half", False)
            q = apply_rot_embed_cat(q, rope_with_prefix, half=half).type_as(v)
            k = apply_rot_embed_cat(k, rope_with_prefix, half=half).type_as(v)

        if attn_mask is not None or is_causal:
            raise ValueError(
                "The Hailo EVA export layout does not support an additional "
                "attention mask or causal attention"
            )

        # timm uses SDPA (or its ordinary softmax fallback). The Hailo layout
        # repeats each real prefix token m times. Give every copy a -log(m)
        # key bias, so their combined softmax contribution is exactly the
        # original token's: m * exp(score - log(m)) == exp(score). This keeps
        # native softmax, and repeated tokens remain identical inductively
        # because attention and the MLP apply the same operations to each copy.
        multiplicities = self._hailo_prefix_multiplicities
        prefix_bias = q.new_tensor(
            tuple(
                -log(multiplicity)
                for multiplicity in multiplicities  # ty: ignore[not-iterable]
                for _ in range(multiplicity)
            )
        )
        key_bias = torch.cat(
            (prefix_bias, q.new_zeros(N - self.num_prefix_tokens))
        ).reshape(1, 1, 1, N)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        attention = (scores + key_bias).softmax(dim=-1)
        attention = self.attn_drop(attention)
        x = attention @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        if gate is not None:
            x = x * gate
        x = self.proj(x)
        return self.proj_drop(x)


class HailoEva(Eva):
    """EVA export variant with a Hailo-compatible prefix-token concat."""

    # DFC can buffer at most 255 rows on the problematic attention paths. Its
    # decomposed Q/K/V dataflow can require three copies, so constrain the
    # source layout to floor(255 / 3) rows rather than relying on the allocator
    # to discover a reshape after export.
    _hailo_max_layout_rows = 255 // 3
    _hailo_max_layout_width = 255

    @classmethod
    def _hailo_layout_width(cls, patch_tokens: int, prefix_tokens: int) -> int:
        """Choose an exact 2-D layout from the current static image shape."""
        for width in range(
            prefix_tokens,
            min(patch_tokens, cls._hailo_max_layout_width) + 1,
        ):
            if (
                patch_tokens % width == 0
                and patch_tokens // width + 1 <= cls._hailo_max_layout_rows
            ):
                return width
        raise ValueError(
            "No exact Hailo EVA layout exists for "
            f"{patch_tokens} patch tokens and {prefix_tokens} prefix tokens: "
            "a patch-token divisor <= 255 producing <= 85 rows is required"
        )

    def _prepend_prefix_tokens(
        self,
        x: Tensor,
        cls_token: Tensor | None,
        reg_token: Tensor | None,
    ) -> Tensor:
        patch_tokens = x.shape[1]
        cls_tokens = 0 if cls_token is None else cls_token.shape[1]
        reg_tokens = 0 if reg_token is None else reg_token.shape[1]
        prefix_tokens = cls_tokens + reg_tokens

        # timm concatenates exactly [CLS, registers, patches]. Repeat each
        # prefix token enough times to fill one shape-derived row. Attention
        # compensates for each multiplicity exactly, and forward_intermediates
        # removes the complete repeated-prefix row before reshaping patches.
        row_width = self._hailo_layout_width(patch_tokens, prefix_tokens)
        copies_per_token, extra_copies = divmod(row_width, prefix_tokens)
        multiplicities = tuple(
            copies_per_token + (index < extra_copies) for index in range(prefix_tokens)
        )
        self.num_prefix_tokens = row_width
        self._hailo_prefix_multiplicities = multiplicities
        for module in self.modules():
            if isinstance(module, HailoEvaAttention):
                module.num_prefix_tokens = row_width
                module._hailo_prefix_multiplicities = multiplicities  # ty: ignore[invalid-assignment]

        x = x.transpose(1, 2).reshape(
            x.shape[0], x.shape[2], patch_tokens // row_width, row_width
        )
        prefix_parts = [t for t in (cls_token, reg_token) if t is not None]
        prefix = (
            prefix_parts[0]
            if len(prefix_parts) == 1
            else torch.cat(prefix_parts, dim=1)
        )
        prefix = torch.cat(
            tuple(
                prefix[:, index : index + 1].expand(-1, multiplicity, -1)
                for index, multiplicity in enumerate(multiplicities)
            ),
            dim=1,
        )
        prefix = prefix.transpose(1, 2).unsqueeze(2)
        x = torch.cat((prefix, x), dim=2)

        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])

    def _pos_embed(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed,
                    new_size=(H, W),
                    old_size=self.patch_embed.grid_size,
                    num_prefix_tokens=(
                        0 if self.no_embed_class else self.num_prefix_tokens
                    ),
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = (
                self.rope.get_embed(shape=(H, W)) if self.rope is not None else None  # ty: ignore[call-non-callable]
            )
        else:
            pos_embed = self.pos_embed
            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None  # ty: ignore[call-non-callable]

        # timm builds one `to_cat` list and performs a direct token-axis concat.
        # Keep the same order and expansion, but pass the prefix and patches to
        # the DFC-specific, layout-preserving concat below.
        cls_token = (
            self.cls_token.expand(x.shape[0], -1, -1)
            if self.cls_token is not None
            else None
        )
        reg_token = (
            self.reg_token.expand(x.shape[0], -1, -1)
            if self.reg_token is not None
            else None
        )

        if self.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if cls_token is not None or reg_token is not None:
                x = self._prepend_prefix_tokens(x, cls_token, reg_token)
        else:
            if cls_token is not None or reg_token is not None:
                x = self._prepend_prefix_tokens(x, cls_token, reg_token)
            if pos_embed is not None:
                # timm has one absolute position per real prefix token. Repeat
                # those positions with the same multiplicities as the tokens.
                real_npt = (0 if cls_token is None else cls_token.shape[1]) + (
                    0 if reg_token is None else reg_token.shape[1]
                )
                if self.num_prefix_tokens != real_npt:
                    prefix_pos_embed = torch.cat(
                        tuple(
                            pos_embed[:, index : index + 1].expand(-1, multiplicity, -1)
                            for index, multiplicity in enumerate(
                                self._hailo_prefix_multiplicities
                            )
                        ),
                        dim=1,
                    )
                    pos_embed = torch.cat(
                        (
                            prefix_pos_embed,
                            pos_embed[:, real_npt:],
                        ),
                        dim=1,
                    )
                x = x + pos_embed

        x = self.pos_drop(x)

        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
                if getattr(self, "rope_mixed", False):
                    rot_pos_embed = rot_pos_embed.transpose(0, 1)
                else:
                    rot_pos_embed = rot_pos_embed.unsqueeze(1)

        return x, rot_pos_embed

    def forward_intermediates(
        self,
        x: Tensor,
        indices: int | list[int] | None = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        stop_early: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ):
        """Run timm's intermediate path with a spatial prefix-row crop."""
        assert output_fmt in ("NCHW", "NLC")
        reshape = output_fmt == "NCHW"
        intermediates = []
        take_indices, max_index = feature_take_indices(len(self.blocks), indices)

        # This is timm's Eva.forward_intermediates execution path. Keep it here
        # because its normal post-processing slices the flattened token axis;
        # DFC maps that 1x1280 slice to an unsupported two-APU 16x4 kernel.
        B, _, height, width = x.shape
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        if torch.jit.is_scripting() or not stop_early:
            blocks = self.blocks
        else:
            blocks = self.blocks[: max_index + 1]

        if getattr(self, "rope_mixed", False) and rot_pos_embed is not None:
            for index, block in enumerate(blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(
                        block,
                        x,
                        rope=rot_pos_embed[index],
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                    )
                else:
                    x = block(
                        x,
                        rope=rot_pos_embed[index],
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                    )
                if index in take_indices:
                    intermediates.append(self.norm(x) if norm else x)
        else:
            for index, block in enumerate(blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(
                        block,
                        x,
                        rope=rot_pos_embed,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                    )
                else:
                    x = block(
                        x,
                        rope=rot_pos_embed,
                        attn_mask=attn_mask,
                        is_causal=is_causal,
                    )
                if index in take_indices:
                    intermediates.append(self.norm(x) if norm else x)

        H, W = self.patch_embed.dynamic_feat_size((height, width))
        row_width = self.num_prefix_tokens
        patch_rows = H * W // row_width
        prefix_tokens = [y[:, :row_width] for y in intermediates]

        # Hailo-only divergence from timm: recover the 2-D layout before
        # removing the repeated prefix row. Values and order are identical to
        # y[:, row_width:], but DFC receives an 80x16 spatial crop instead of
        # an unsupported 1x1280 token slice at the current 512x640 resolution.
        intermediates = [
            y.reshape(B, patch_rows + 1, row_width, -1)[:, 1:] for y in intermediates
        ]
        if reshape:
            intermediates = [
                y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                for y in intermediates
            ]
        else:
            intermediates = [y.reshape(B, H * W, -1) for y in intermediates]

        if not torch.jit.is_scripting() and return_prefix_tokens:
            intermediates = list(zip(intermediates, prefix_tokens))
        if intermediates_only:
            return intermediates

        x = self.norm(x)
        return x, intermediates


@contextmanager
def hailo_eva_attention_export(model: nn.Module) -> Iterator[None]:
    """Use Hailo-compatible EVA operations temporarily during export."""
    patched_models: list[tuple[Eva, int]] = []
    patched_attention: list[tuple[EvaAttention, int]] = []
    for module in model.modules():
        # timm has no export hook for these substitutions. Patch exact classes
        # only (not subclasses), and restore them in finally, so eager training
        # and inference continue to execute timm's original implementation.
        if type(module) is Eva:
            original_num_prefix_tokens = module.num_prefix_tokens
            module.__class__ = HailoEva
            patched_models.append((module, original_num_prefix_tokens))
        if type(module) is EvaAttention:
            original_num_prefix_tokens = module.num_prefix_tokens
            module.__class__ = HailoEvaAttention
            patched_attention.append((module, original_num_prefix_tokens))

    try:
        yield
    finally:
        for module, original_num_prefix_tokens in patched_attention:
            module.num_prefix_tokens = original_num_prefix_tokens
            if hasattr(module, "_hailo_prefix_multiplicities"):
                del module._hailo_prefix_multiplicities
            module.__class__ = EvaAttention
        for module, original_num_prefix_tokens in patched_models:
            module.num_prefix_tokens = original_num_prefix_tokens
            if hasattr(module, "_hailo_prefix_multiplicities"):
                del module._hailo_prefix_multiplicities
            module.__class__ = Eva

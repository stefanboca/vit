from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DPT(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_channels: tuple[int, int, int, int] = (256, 512, 1024, 1024),
        out_channels: int = 256,
        batch_norm: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.reassemble_blocks = nn.ModuleList(
            ReassembleBlock(
                d_model=d_model,
                hidden_channels=hidden_channels[i],
                out_channels=out_channels,
                index=i,  # ty: ignore[invalid-argument-type]
                device=device,
            )
            for i in range(4)
        )

        self.fusion_blocks = nn.ModuleList(
            FeatureFusionBlock(
                features=out_channels,
                act=nn.ReLU,
                batch_norm=batch_norm,
                has_residual_input=i != 3,
                device=device,
            )
            for i in range(4)
        )

    def forward(self, cls_tokens: list[Tensor], patch_tokens: list[Tensor]) -> Tensor:
        assert len(cls_tokens) == len(patch_tokens) == 4

        xs: list[Tensor] = [
            blk.forward(cls, patches)
            for blk, cls, patches in zip(
                self.reassemble_blocks, cls_tokens, patch_tokens
            )
        ]

        x3 = self.fusion_blocks[3].forward(xs[3])
        x2 = self.fusion_blocks[2].forward(x3, xs[2])
        x1 = self.fusion_blocks[1].forward(x2, xs[1])
        x0 = self.fusion_blocks[0].forward(x1, xs[0])

        return x0


class DPTSegHead(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 256,
        n_classes: int = 2,
        drop: float = 0.1,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(in_channels, device=device),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop),
            nn.Conv2d(in_channels, n_classes, kernel_size=1, device=device),
        )

    def forward(self, input: Tensor) -> Tensor:
        x: Tensor = self.head.forward(input)
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        return x


class ReassembleBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        hidden_channels: int,
        out_channels: int,
        index: Literal[0, 1, 2, 3],
        device: torch.device | None = None,
    ):
        super().__init__()

        self.patch_proj = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=1,
            device=device,
        )
        self.cls_proj = nn.Linear(
            d_model,
            d_model,
            bias=False,
            device=device,
        )

        self.conv1 = nn.Conv2d(d_model, hidden_channels, kernel_size=1, device=device)
        if index in (0, 1):
            size = [4, 2][index]
            self.conv2 = nn.ConvTranspose2d(
                hidden_channels,
                hidden_channels,
                kernel_size=size,
                stride=size,
                device=device,
            )
        elif index == 2:
            self.conv2 = None
        elif index == 3:
            self.conv2 = nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1
            )

        self.conv3 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            device=device,
        )

    def forward(self, cls_token: Tensor, patch_tokens: Tensor) -> Tensor:
        patch_x = self.patch_proj(patch_tokens)
        cls_x = self.cls_proj(cls_token)[:, :, None, None]

        x = F.gelu(patch_x + cls_x)
        x = self.conv1(x)
        if self.conv2 is not None:
            x = self.conv2(x)

        return self.conv3(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_weight_key = f"{prefix}proj.0.weight"
        old_bias_key = f"{prefix}proj.0.bias"

        patch_weight_key = f"{prefix}patch_proj.weight"
        patch_bias_key = f"{prefix}patch_proj.bias"
        cls_weight_key = f"{prefix}cls_proj.weight"

        if old_weight_key in state_dict:
            old_weight = state_dict.pop(old_weight_key)
            split = self.patch_proj.in_channels
            expected_shape = (
                self.patch_proj.out_channels,
                self.patch_proj.in_channels + self.cls_proj.in_features,
            )

            if tuple(old_weight.shape) != expected_shape:
                error_msgs.append(
                    f"{old_weight_key} has shape {tuple(old_weight.shape)}, "
                    f"expected {expected_shape}"
                )
            else:
                patch_weight, cls_weight = old_weight.split(split, dim=1)
                state_dict[patch_weight_key] = patch_weight[:, :, None, None].clone()
                state_dict[cls_weight_key] = cls_weight.clone()

        if old_bias_key in state_dict:
            state_dict[patch_bias_key] = state_dict.pop(old_bias_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FeatureFusionBlock(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        act: Callable[..., nn.Module] = nn.ReLU,
        batch_norm: bool = False,
        align_corners: bool = False,
        has_residual_input: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.align_corners = align_corners

        self.out_conv = nn.Conv2d(
            features,
            features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
            device=device,
        )

        self.res_conv_1 = ResidualConvUnit(
            features=features, act=act, batch_norm=batch_norm, device=device
        )
        if has_residual_input:
            self.res_conv_2 = ResidualConvUnit(
                features=features, act=act, batch_norm=batch_norm, device=device
            )

    def forward(self, x: Tensor, x2: Tensor | None = None) -> Tensor:
        x = self.res_conv_1(x)
        if x2 is not None:
            if x.shape[-2:] != x2.shape[-2:]:
                x = F.interpolate(
                    x,
                    size=x2.shape[-2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
            x += self.res_conv_2(x2)

        x = F.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        x = self.out_conv(x)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(
        self,
        *,
        features: int,
        act: Callable[..., nn.Module],
        batch_norm: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.block = nn.Sequential(
            act(inplace=True),
            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
                bias=not batch_norm,
                device=device,
            ),
            nn.BatchNorm2d(features, device=device) if batch_norm else nn.Identity(),
            act(inplace=True),
            nn.Conv2d(
                features,
                features,
                kernel_size=3,
                padding=1,
                bias=not batch_norm,
                device=device,
            ),
            nn.BatchNorm2d(features, device=device) if batch_norm else nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block.forward(x)

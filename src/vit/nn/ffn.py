from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Mlp(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, input: Tensor) -> Tensor:
        x = self.fc1.forward(input)
        x = self.act.forward(x)
        x = self.drop.forward(x)
        x = self.fc2.forward(x)
        x = self.drop.forward(x)
        return x


class SwiGluFfn(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        align_to: int = 8,
        device: torch.device | None = None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        d = hidden_features * 2 // 3
        hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias, device=device)

    def forward(self, input: Tensor) -> Tensor:
        x1 = self.w1.forward(input)
        x2 = self.w2.forward(input)
        x = F.silu(x1) * x2
        x = self.w3.forward(x)
        return x

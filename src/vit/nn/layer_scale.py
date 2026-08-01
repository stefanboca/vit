import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        init_values: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device))

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma

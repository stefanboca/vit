import torch
from torch import Tensor


def align_depth_affine_irls(
    depth_src: Tensor,
    depth_tgt: Tensor,
    weight: Tensor,
    max_iter: int = 100,
    eps: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    with torch.autocast("cuda", enabled=False):
        src = depth_src.float()
        tgt = depth_tgt.float()
        base_weight = weight.float()

        x = torch.stack((src, torch.ones_like(src)), dim=-1)
        current_weight = base_weight

        for _ in range(max_iter):
            lhs = x.transpose(-1, -2) @ (current_weight[..., None] * x)
            rhs = x.transpose(-1, -2) @ (current_weight * tgt)[..., None]
            beta = torch.linalg.solve(lhs, rhs).squeeze(-1)
            residual = tgt - (x @ beta[..., None]).squeeze(-1)
            current_weight = base_weight / residual.abs().clamp_min(eps)

    return beta[..., 0], beta[..., 1]

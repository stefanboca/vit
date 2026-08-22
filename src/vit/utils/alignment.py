import torch
from torch import Tensor


def align_depth_affine_irls(
    depth_src: Tensor,
    depth_tgt: Tensor,
    weight: Tensor,
    max_iter: int = 30,
    eps: float = 1e-4,
    solve_eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    with torch.autocast("cuda", enabled=False):
        src = depth_src.float()
        tgt = depth_tgt.float()
        base_weight = weight.float()

        weight_sum = base_weight.sum(dim=-1, keepdim=True).clamp_min(1)
        src_mean = (base_weight * src).sum(dim=-1, keepdim=True) / weight_sum
        src_variance = (
            base_weight * (src - src_mean).square()
        ).sum(dim=-1, keepdim=True) / weight_sum
        src_std = src_variance.sqrt().clamp_min(eps)
        src_normalized = (src - src_mean) / src_std

        x = torch.stack((src_normalized, torch.ones_like(src)), dim=-1)
        current_weight = base_weight
        eye = torch.eye(2, dtype=x.dtype, device=x.device)

        for _ in range(max_iter):
            lhs = x.transpose(-1, -2) @ (current_weight[..., None] * x)
            lhs = lhs + solve_eps * weight_sum[..., None] * eye
            rhs = x.transpose(-1, -2) @ (current_weight * tgt)[..., None]
            beta = torch.linalg.solve(lhs, rhs).squeeze(-1)
            residual = tgt - (x @ beta[..., None]).squeeze(-1)
            current_weight = base_weight / residual.abs().clamp_min(eps)

    scale = beta[..., :1] / src_std
    shift = beta[..., 1:] - scale * src_mean
    return scale.squeeze(-1), shift.squeeze(-1)

import torch
from torch import Tensor


@torch.inference_mode()
def imagenet_unnorm(images: Tensor) -> Tensor:
    mean = images.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = images.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return (images * std + mean).clamp(0, 1)


@torch.inference_mode()
def colorize_depth(
    depth: Tensor,
    valid: Tensor | None,
    lut: Tensor,
    vmin: float | Tensor,
    vmax: float | Tensor,
) -> Tensor:
    depth = depth.float().squeeze(1)

    lower = torch.as_tensor(vmin, dtype=depth.dtype, device=depth.device)
    upper = torch.as_tensor(vmax, dtype=depth.dtype, device=depth.device)
    denominator = (upper - lower).clamp_min(torch.finfo(depth.dtype).eps)
    normalized = ((depth - lower) / denominator).clamp(0, 1)
    indices = (normalized * (lut.shape[0] - 1)).round().long()
    rgb = lut[indices].permute(0, 3, 1, 2)  # [B, H, W, 3] -> [B, 3, H, W]
    if valid is not None:
        valid = valid.squeeze(1).bool()
        rgb = torch.where(valid[:, None], rgb, torch.zeros_like(rgb))
    return rgb

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn


class HungarianMatcher(nn.Module):
    """Match predicted mask channels to target instance masks."""

    def __init__(self, cost_mask: float = 1, cost_dice: float = 1):
        super().__init__()
        if cost_mask < 0 or cost_dice < 0 or cost_mask + cost_dice == 0:
            raise ValueError(
                "mask costs must be nonnegative, with at least one positive"
            )
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self, outputs: Tensor, targets: list[Tensor]
    ) -> list[tuple[Tensor, Tensor]]:
        """Match logits [B, Q, H, W] to binary masks [N, H, W] per image.

        Returns CPU index pairs (prediction channels, target masks). Each pair
        has min(Q, N) entries, ordered by prediction channel.
        """
        matches = []
        for logits, masks in zip(outputs, targets, strict=True):
            if masks.shape[0] == 0 or logits.shape[0] == 0:
                empty = torch.empty(0, dtype=torch.int64)
                matches.append((empty, empty))
                continue

            pred = logits.flatten(1).float()  # [Q, H*W]
            target = masks.flatten(1).to(device=pred.device, dtype=pred.dtype)
            pixels = pred.shape[1]

            # BCEWithLogits(x, y) = softplus(x) - x * y
            cost_mask = (
                F.softplus(pred).mean(1, keepdim=True) - pred @ target.T / pixels
            )

            probability = pred.sigmoid()
            intersection = probability @ target.T
            cost_dice = 1 - (2 * intersection + 1) / (
                probability.sum(1, keepdim=True) + target.sum(1)[None, :] + 1
            )

            cost = self.cost_mask * cost_mask + self.cost_dice * cost_dice
            pred_indices, target_indices = linear_sum_assignment(cost.cpu())
            matches.append(
                (
                    torch.as_tensor(pred_indices, dtype=torch.int64),
                    torch.as_tensor(target_indices, dtype=torch.int64),
                )
            )
        return matches

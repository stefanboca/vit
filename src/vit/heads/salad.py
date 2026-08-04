import math

import torch
from torch import Tensor, nn


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(
    log_a: Tensor, log_b: Tensor, M: Tensor, num_iters: int = 20, reg: float = 1.0
) -> Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : Tensor
            Source weights
        log_b : Tensor
            Target weights
        M : Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(
    S: Tensor, dustbin_score: float | Tensor = 1.0, num_iters: int = 3, reg: float = 1.0
) -> Tensor:
    """sinkhorn"""
    B, M, N = S.shape
    # augment scores matrix
    S_aug = torch.empty(B, M + 1, N, dtype=S.dtype, device=S.device)
    S_aug[:, :M, :N] = S
    S_aug[:, M, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(N + M), device=S.device)
    log_a, log_b = norm.expand(M + 1).contiguous(), norm.expand(N).contiguous()
    log_a[-1] = log_a[-1] + math.log(N - M)
    log_a, log_b = log_a.expand(B, -1), log_b.expand(B, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class SALAD(nn.Module):
    """
    This class represents the Sinkhorn Algorithm for Locally Aggregated Descriptors (SALAD) model.
    https://arxiv.org/abs/2311.15937

    Attributes:
        n_channels (int): The number of channels of the inputs (d).
        n_clusters (int): The number of clusters in the model (m).
        d_cluster (int): The number of channels of the clusters (l).
        d_token (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
    """

    def __init__(
        self,
        *,
        n_channels: int = 1536,
        n_clusters: int = 64,
        d_cluster: int = 128,
        d_token: int = 256,
        hidden_features_token: int = 512,
        hidden_features_cluster: int = 512,
        hidden_features_score: int = 512,
        drop: float = 0.3,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.n_clusters = n_clusters
        self.d_cluster = d_cluster

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(n_channels, hidden_features_token),
            nn.ReLU(),
            nn.Linear(hidden_features_token, d_token, device=device),
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(n_channels, hidden_features_cluster, 1, device=device),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Conv2d(hidden_features_cluster, self.d_cluster, 1, device=device),
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(n_channels, hidden_features_score, 1, device=device),
            nn.Dropout(drop),
            nn.ReLU(),
            nn.Conv2d(hidden_features_score, self.n_clusters, 1, device=device),
        )
        # Dustbin parameter z
        self.dustbin = nn.Parameter(torch.tensor(1.0, device=device))

    def forward(self, cls_token: Tensor, patch_tokens: Tensor) -> Tensor:
        """
        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        t, x = cls_token, patch_tokens
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        f = self.cluster_features.forward(x).flatten(2)
        p = self.score.forward(x).flatten(2)
        t = self.token_features.forward(t)

        # Sinkhorn algorithm
        p = get_matching_probs(p, self.dustbin.data, 3)
        p = torch.exp(p)
        # Normalize to maintain mass
        p = p[:, :-1, :]

        p = p.unsqueeze(1).repeat(1, self.d_cluster, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.n_clusters, 1)

        f = torch.cat(
            [
                nn.functional.normalize(t, p=2, dim=-1),
                nn.functional.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return nn.functional.normalize(f, p=2, dim=-1)

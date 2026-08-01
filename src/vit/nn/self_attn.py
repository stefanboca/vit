import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..nn.rope import RoPECoeffs


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            bias_mask = torch.ones_like(self.bias, device=self.bias.device)
            bias_mask[o // 3 : 2 * o // 3].fill_(0)
            self.register_buffer("bias_mask", bias_mask)

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask if self.bias is not None else None  # ty:ignore[unsupported-operator]
        return F.linear(input, self.weight, masked_bias)


class SelfAttn(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.n_heads = n_heads

        linear = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear(d_model, d_model * 3, bias=qkv_bias, device=device)
        self.proj = nn.Linear(d_model, d_model, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, input: Tensor, rope: RoPECoeffs | None = None) -> Tensor:
        # TODO: double check all shapes
        qkv = self.qkv.forward(input)

        b, n, _ = qkv.shape
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.d_head)
        q, k, v = (t.transpose(1, 2) for t in torch.unbind(qkv, 2))
        if rope:
            q = rope.apply(q)
            k = rope.apply(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(b, n, self.d_model)

        x = self.proj.forward(x)
        x = self.proj_drop.forward(x)
        return x

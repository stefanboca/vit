import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import RotaryEmbedding, RotaryEmbeddingValues


class RotaryDecoderLayer(nn.Module):
    """Pre-norm causal decoder layer with cross-attention and self-attention RoPE."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if n_head < 1 or d_model % n_head:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )

        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.dropout_p = dropout
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model

        self.self_norm = nn.LayerNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model)
        self.self_out = nn.Linear(d_model, d_model)

        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_q = nn.Linear(d_model, d_model)
        self.cross_kv = nn.Linear(d_model, 2 * d_model)
        self.cross_out = nn.Linear(d_model, d_model)

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        self_attn_mask: Tensor,
        rope: RotaryEmbeddingValues,
    ) -> Tensor:
        """Process batch-first tensors: tgt [B,T,D], memory [B,S,D]."""
        q, k, v = self.self_qkv(self.self_norm(tgt)).chunk(3, dim=-1)
        q, k, v = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        q = rope.apply(q)
        k = rope.apply(k)
        self_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=self_attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        tgt = tgt + self.dropout(self.self_out(self._merge_heads(self_attn)))

        q = self._split_heads(self.cross_q(self.cross_norm(tgt)))
        k, v = self.cross_kv(memory).chunk(2, dim=-1)
        k, v = self._split_heads(k), self._split_heads(v)
        cross_attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        tgt = tgt + self.dropout(self.cross_out(self._merge_heads(cross_attn)))

        return tgt + self.dropout(self.ffn(self.ffn_norm(tgt)))


class RotaryTransformerDecoder(nn.Module):
    """Batch-first causal decoder with padding masks and shared per-pass RoPE."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_layers: int,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rope = RotaryEmbedding(d_model // n_head)
        self.layers = nn.ModuleList(
            RotaryDecoderLayer(d_model, n_head, dim_feedforward, dropout)
            for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_valid: Tensor) -> Tensor:
        """Return [B,T,D] from tgt [B,T,D], memory [B,S,D], valid mask [B,T]."""
        tgt_len = tgt.shape[1]
        causal = torch.ones(
            tgt_len, tgt_len, dtype=torch.bool, device=tgt.device
        ).tril()
        # SDPA boolean masks use True for allowed query/key pairs.
        self_attn_mask = causal[None, None, :, :] & tgt_valid[:, None, None, :]
        rope = self.rope(tgt_len, tgt.device, tgt.dtype)
        for layer in self.layers:
            tgt = layer(tgt, memory, self_attn_mask, rope)
        return self.norm(tgt)

from __future__ import annotations

import math

import torch
import torch.nn as nn

from eecs148b_hw1.linear import Linear


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Numerically stable softmax along `dim`."""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


class Attention(nn.Module):
    """Utility attention class providing softmax for attention computations."""

    def __init__(self, Q, K, V, mask=None):
        super().__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.mask = mask

    def apply_softmax(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        return softmax(x, dim)
    
    def scaled_dot_product_attention(self):
        return scaled_dot_product_attention(self.Q, self.K, self.V, self.mask)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention over the last two dimensions.

    Shapes:
      Q: (..., queries, d_k)
      K: (..., keys, d_k)
      V: (..., keys, d_v)
      mask: broadcastable to (..., queries, keys), where True means allowed.
    Returns:
      (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = softmax(scores, dim=-1)
    return torch.matmul(probs, V)


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with d_k = d_v = d_model / num_heads."""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    # TODO check the dims!
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
        seq_len = x.shape[-2]
        leading_shape = x.shape[:-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (..., seq_len, num_heads, head_dim) -> (..., num_heads, seq_len, head_dim)
        q = q.reshape(*leading_shape, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.reshape(*leading_shape, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.reshape(*leading_shape, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)

        # True entries are allowed.
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        attn_out = attn_out.transpose(-3, -2).reshape(*leading_shape, seq_len, self.d_model)
        return self.output_proj(attn_out)

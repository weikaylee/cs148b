"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # theta_i = base ** (-2i / head_dim)  for i in [0, head_dim/2).
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        # (head_dim // 2,)
        t = torch.arange(max_seq_len).float()
        # (max_seq_len,)
        freqs = torch.outer(t, inv_freq)
        # freqs: (max_seq_len, head_dim // 2) — entry [m, i] = m * theta_i.

        # Buffers move with .to(device) but are NOT trained and NOT saved in
        # state_dict (persistent=False), so reloads don't pin a stale cache.
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x:         (B, num_heads, T, head_dim)
        # positions: (T,) — one absolute position per token.
        cos = self.cos_cached[positions]  # (T, head_dim // 2)
        sin = self.sin_cached[positions]  # (T, head_dim // 2)

        # Group head_dim into d/2 pairs: (x[..., 0::2], x[..., 1::2]).
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # Broadcast (T, head_dim // 2) against (B, H, T, head_dim // 2).
        new_even = x_even * cos - x_odd * sin
        new_odd = x_even * sin + x_odd * cos

        # Re-interleave so the output retains the (..., x_0, x_1, x_2, ...) layout.
        out = torch.empty_like(x)
        out[..., 0::2] = new_even
        out[..., 1::2] = new_odd
        return out


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        # Each spatial axis gets head_dim / 2 dims, which means head_dim / 4
        # rotation pairs (each pair carries one (cos, sin) frequency). The
        # frequency schedule is the standard one for "1D RoPE on a vector of
        # length head_dim / 2".
        half = head_dim // 2
        inv_freq = base ** (-torch.arange(0, half, 2).float() / half)
        # inv_freq: (head_dim // 4,)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)
        # freqs: (grid_size, head_dim // 4) — entry [m, i] = m * theta_i.

        # One cache shared by both axes — x_coords and y_coords index the same
        # table but at different positions.
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        # x:        (B, num_heads, T, head_dim)
        # x_coords, y_coords: (T,)
        half = self.head_dim // 2
        # First half rotates by the x-coordinate; second half by the y-coordinate.
        part_x = x[..., :half]
        part_y = x[..., half:]

        cos_x = self.cos_cached[x_coords]  # (T, head_dim // 4)
        sin_x = self.sin_cached[x_coords]
        cos_y = self.cos_cached[y_coords]
        sin_y = self.sin_cached[y_coords]

        def _rotate(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            # Same pair-wise rotation as RoPE1D, just factored out so we can
            # apply it to each half with its own (cos, sin).
            t_even = t[..., 0::2]
            t_odd = t[..., 1::2]
            new_even = t_even * cos - t_odd * sin
            new_odd = t_even * sin + t_odd * cos
            out = torch.empty_like(t)
            out[..., 0::2] = new_even
            out[..., 1::2] = new_odd
            return out

        return torch.cat(
            [_rotate(part_x, cos_x, sin_x), _rotate(part_y, cos_y, sin_y)], dim=-1
        )


if __name__ == "__main__":
    # Manual norm-preservation check (writeup §6 rope_1d).
    torch.manual_seed(0)
    head_dim, max_seq = 64, 128
    B, H, T = 4, 8, 64
    rope = RoPE1D(head_dim=head_dim, max_seq_len=max_seq, base=10_000.0)

    x = torch.randn(B, H, T, head_dim)
    positions = torch.arange(T)
    y = rope(x, positions)

    in_norms = x.norm(dim=-1)   # (B, H, T)
    out_norms = y.norm(dim=-1)
    abs_diff = (in_norms - out_norms).abs()

    print(f"RoPE1D norm-preservation check (head_dim={head_dim}, T={T}):")
    print(f"  max |‖x‖ - ‖RoPE(x)‖|     = {abs_diff.max().item():.3e}")
    print(f"  mean |‖x‖ - ‖RoPE(x)‖|    = {abs_diff.mean().item():.3e}")
    print(f"  relative max diff         = "
          f"{(abs_diff / in_norms.clamp_min(1e-12)).max().item():.3e}")

    # Identity check at position 0.
    x0 = torch.randn(1, 1, 4, head_dim)
    y0 = rope(x0, torch.zeros(4, dtype=torch.long))
    print(f"  ‖RoPE_pos=0(x) - x‖_max   = {(y0 - x0).abs().max().item():.3e}")

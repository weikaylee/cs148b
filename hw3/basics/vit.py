"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import MLP, Block
from basics.rope import RoPE1D, RoPE2D

class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        # A strided Conv2d with kernel == stride == patch_size is equivalent to
        # slicing the image into non-overlapping patches and applying a shared
        # linear projection (the conv kernel) to each one.
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (B, 3, img_size, img_size)
        x = self.proj(x)
        # x:   (B, d_model, grid_size, grid_size)
        x = x.flatten(2)
        # x:   (B, d_model, num_patches)            where num_patches = grid_size ** 2
        x = x.transpose(1, 2)
        # x:   (B, num_patches, d_model)            one token per patch
        return x


class _RoPEAttention(nn.Module):
    """Bidirectional multi-head self-attention with RoPE applied to (q, k).

    Lives in vit.py (not model.py) because the staff `Head` / `MultiHeadAttention`
    don't expose position information. RoPE is shared across all heads via a
    single RoPE1D table per layer; sharing across *layers* isn't done, but the
    buffers are small.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE1D(self.head_dim, max_seq_len, base=rope_base)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model), positions: (T,)
        B, T, _ = x.shape

        def reshape_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(self.q_proj(x))  # (B, H, T, head_dim)
        k = reshape_heads(self.k_proj(x))
        v = reshape_heads(self.v_proj(x))

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v  # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_dropout(self.out_proj(out))


class _RoPEBlock(nn.Module):
    """Pre-LayerNorm block matching `basics.model.Block`'s shape but using
    RoPE-aware attention. ViT uses bidirectional attention (no causal mask)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPEAttention(
            d_model, num_heads, max_seq_len, rope_base=rope_base, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), positions)
        x = x + self.mlp(self.ln2(x))
        return x


class _RoPE2DAttention(nn.Module):
    """Multi-head self-attention with 2D RoPE applied to (q, k) before the dot
    product. Each token is described by an (x_coord, y_coord) pair on the patch
    grid; the CLS token uses coordinate (0, 0) and patches are shifted by +1
    on both axes so they don't collide with CLS."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        grid_size: int,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for RoPE2D"

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RoPE2D(self.head_dim, grid_size=grid_size, base=rope_base)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor
    ) -> torch.Tensor:
        B, T, _ = x.shape

        def reshape_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(self.q_proj(x))
        k = reshape_heads(self.k_proj(x))
        v = reshape_heads(self.v_proj(x))

        q = self.rope(q, x_coords, y_coords)
        k = self.rope(k, x_coords, y_coords)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_dropout(self.out_proj(out))


class _RoPE2DBlock(nn.Module):
    """Pre-LayerNorm block with 2D-RoPE-aware attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        grid_size: int,
        rope_base: float = 10_000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = _RoPE2DAttention(
            d_model, num_heads, grid_size, rope_base=rope_base, dropout=dropout
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model=d_model, dropout=dropout)

    def forward(
        self, x: torch.Tensor, x_coords: torch.Tensor, y_coords: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), x_coords, y_coords)
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3. Inject position information either via:
           - "learned"  — additive learnable pos_embed (the classic ViT recipe).
           - "rope"     — 1D RoPE applied to (q, k) inside attention; no additive
                          pos_embed. Token positions are their 1D index, so CLS
                          is at position 0 and patches at 1..N.
      4. `num_blocks` Transformer Blocks (bidirectional; `is_decoder=False`).
      5. Final LayerNorm.
      6. Return CLS slice (B, d_model), or the full sequence if
         `return_all_tokens=True`.

    For learned PE, `forward` transparently bilinearly-interpolates `pos_embed`
    if the input produces a different number of patches than training (the CLS
    slot is kept separate from the interpolation). For RoPE, `rope_max_seq_len`
    controls the cos/sin cache size; set it generously when you want to
    evaluate at a larger image size than training.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pos_encoding:      "learned" (default) or "rope".
        rope_max_seq_len:  Cache size for RoPE; defaults to 4*(num_patches+1)
                           so a 2× linear-resolution extrapolation just works.
        rope_base:         RoPE base; ignored for learned PE.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
        rope_max_seq_len: int | None = None,
        rope_grid_size: int | None = None,
        rope_base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if pos_encoding not in ("learned", "rope", "rope2d"):
            raise ValueError(f"unknown pos_encoding: {pos_encoding!r}")

        self.d_model = d_model  # exposed for vlm/eval.py's projection-head probe
        self.pos_encoding = pos_encoding
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches
        block_size = num_patches + 1  # +1 for the CLS token
        train_grid_side = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        if pos_encoding == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, block_size, d_model))
            self.blocks = nn.ModuleList(
                [
                    Block(
                        d_model=d_model,
                        num_heads=num_heads,
                        block_size=block_size,
                        is_decoder=False,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        elif pos_encoding == "rope":
            max_seq_len = rope_max_seq_len if rope_max_seq_len is not None else 4 * block_size
            self.rope_max_seq_len = max_seq_len
            self.blocks = nn.ModuleList(
                [
                    _RoPEBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        max_seq_len=max_seq_len,
                        rope_base=rope_base,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )
        else:  # rope2d
            # CLS gets coord (0, 0); patches occupy (1..grid, 1..grid), so the
            # cos/sin cache needs one slot beyond the patch grid side. Default
            # to 4× the (training grid + 1) for generous extrapolation room.
            default_grid = 4 * (train_grid_side + 1)
            grid_size = rope_grid_size if rope_grid_size is not None else default_grid
            self.rope_grid_size = grid_size
            self.blocks = nn.ModuleList(
                [
                    _RoPE2DBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        grid_size=grid_size,
                        rope_base=rope_base,
                        dropout=dropout,
                    )
                    for _ in range(num_blocks)
                ]
            )

        self.norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # PE handling
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_learned_pe(pos_embed: torch.Tensor, target_n: int) -> torch.Tensor:
        """Bilinearly interpolate the learned patch grid from its native square
        size to a new square size. CLS slot (position 0) is kept separate.

        Args:
            pos_embed: (1, n_old, d_model) with the CLS PE at index 0.
            target_n:  Target sequence length, including CLS.
        """
        cls_pe = pos_embed[:, :1]                       # (1, 1, d_model)
        patch_pe = pos_embed[:, 1:]                     # (1, n_old_patches, d_model)
        n_old_patches = patch_pe.shape[1]
        n_new_patches = target_n - 1
        old_side = int(round(n_old_patches ** 0.5))
        new_side = int(round(n_new_patches ** 0.5))
        assert old_side * old_side == n_old_patches, "old PE grid must be square"
        assert new_side * new_side == n_new_patches, "new PE grid must be square"
        d_model = patch_pe.shape[2]
        # (1, d_model, old_side, old_side) -> bilinear -> (1, d_model, new_side, new_side)
        patch_pe = patch_pe.reshape(1, old_side, old_side, d_model).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe, size=(new_side, new_side), mode="bilinear", align_corners=False
        )
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_side * new_side, d_model)
        return torch.cat([cls_pe, patch_pe], dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                                  # (B, N, d_model)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                           # (B, N+1, d_model)
        T = x.shape[1]

        if self.pos_encoding == "learned":
            pos_embed = self.pos_embed
            if pos_embed.shape[1] != T:
                pos_embed = self._interpolate_learned_pe(pos_embed, T)
            x = x + pos_embed
            x = self.dropout(x)
            for block in self.blocks:
                x = block(x)
        elif self.pos_encoding == "rope":
            x = self.dropout(x)
            positions = torch.arange(T, device=x.device)
            if T > self.rope_max_seq_len:
                raise RuntimeError(
                    f"RoPE cache is sized {self.rope_max_seq_len} but the input has "
                    f"{T} tokens. Rebuild ViT with a larger rope_max_seq_len, or call "
                    f"set_rope_max_seq_len(T) before evaluating at a bigger image size."
                )
            for block in self.blocks:
                x = block(x, positions)
        else:  # rope2d
            x = self.dropout(x)
            n_patches = T - 1
            grid_side = int(round(n_patches ** 0.5))
            if grid_side * grid_side != n_patches:
                raise RuntimeError(
                    f"RoPE2D needs a square patch grid; got {n_patches} patches."
                )
            # patch_embed flattens (grid_y, grid_x) in row-major order, so we
            # generate coords with the same convention. Shift by +1 so CLS at
            # (0, 0) is distinguishable from the top-left patch.
            ys, xs = torch.meshgrid(
                torch.arange(grid_side, device=x.device),
                torch.arange(grid_side, device=x.device),
                indexing="ij",
            )
            patch_x = (xs.reshape(-1) + 1)
            patch_y = (ys.reshape(-1) + 1)
            cls_coord = torch.zeros(1, device=x.device, dtype=patch_x.dtype)
            x_coords = torch.cat([cls_coord, patch_x])
            y_coords = torch.cat([cls_coord, patch_y])
            if x_coords.max().item() >= self.rope_grid_size:
                raise RuntimeError(
                    f"RoPE2D cache sized {self.rope_grid_size} but input grid needs "
                    f"{int(x_coords.max().item()) + 1}. Call set_rope_grid_size(...) "
                    f"before evaluating at a larger image."
                )
            for block in self.blocks:
                x = block(x, x_coords, y_coords)

        x = self.norm(x)
        if return_all_tokens:
            return x
        return x[:, 0]

    def set_rope_max_seq_len(self, new_max_seq_len: int) -> None:
        """Rebuild the cos/sin cache of every RoPE block. Only valid in
        pos_encoding='rope' mode."""
        if self.pos_encoding != "rope":
            raise RuntimeError("set_rope_max_seq_len only applies to RoPE ViTs")
        self.rope_max_seq_len = new_max_seq_len
        for block in self.blocks:
            rope = block.attn.rope
            inv_freq = rope.base ** (
                -torch.arange(0, rope.head_dim, 2, device=rope.cos_cached.device).float()
                / rope.head_dim
            )
            t = torch.arange(new_max_seq_len, device=rope.cos_cached.device).float()
            freqs = torch.outer(t, inv_freq)
            rope.max_seq_len = new_max_seq_len
            rope.register_buffer("cos_cached", freqs.cos(), persistent=False)
            rope.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def set_rope_grid_size(self, new_grid_size: int) -> None:
        """Rebuild the per-axis cos/sin cache of every RoPE2D block. Only valid
        in pos_encoding='rope2d' mode."""
        if self.pos_encoding != "rope2d":
            raise RuntimeError("set_rope_grid_size only applies to RoPE2D ViTs")
        self.rope_grid_size = new_grid_size
        for block in self.blocks:
            rope = block.attn.rope
            half = rope.head_dim // 2
            inv_freq = rope.base ** (
                -torch.arange(0, half, 2, device=rope.cos_cached.device).float() / half
            )
            t = torch.arange(new_grid_size, device=rope.cos_cached.device).float()
            freqs = torch.outer(t, inv_freq)
            rope.grid_size = new_grid_size
            rope.register_buffer("cos_cached", freqs.cos(), persistent=False)
            rope.register_buffer("sin_cached", freqs.sin(), persistent=False)



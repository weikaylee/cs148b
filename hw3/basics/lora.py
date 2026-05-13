"""LoRA adapters — §4.

You implement: LoRALinear, apply_lora_to_attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer.

    Computes:  W' x = base_layer(x) + (alpha / rank) * (B A x)

    where:
      - base_layer is the frozen pretrained linear (its weights are not trained).
      - A in R^{rank x d_in}  is initialized with kaiming_uniform_.
      - B in R^{d_out x rank} is initialized to zero (so the adapted layer
        starts equal to the base layer).

    Only A and B receive gradients; base_layer's parameters are frozen.

    Args:
        base_layer: Existing nn.Linear to wrap.
        rank:       Adapter rank `r` (typically 4..32).
        alpha:      Scaling factor; effective scale is `alpha / rank`.
    """

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.base_layer = base_layer

        for p in self.base_layer.parameters():
            p.requires_grad = False

        d_in = base_layer.in_features
        d_out = base_layer.out_features

        # A: kaiming-uniform; B: zero — so at init, the LoRA term is zero and
        # the wrapper returns exactly the base layer's output.
        self.A = nn.Parameter(torch.empty(rank, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_layer(x) + self.scaling * (x @ self.A.T @ self.B.T)


def apply_lora_to_attention(model: nn.Module, rank: int, alpha: float) -> nn.Module:
    """Replace `q_proj` and `v_proj` linear layers in every attention head
    with LoRA-wrapped versions.

    The HW writeup recommends adapting only Q and V projections (per the
    original LoRA paper). Walk the module tree and wherever you find an
    nn.Linear named `q_proj` or `v_proj` inside a Head, swap it for a
    LoRALinear.

    The function modifies `model` in place AND returns it for convenience.

    Args:
        model: A module containing one or more `basics.model.Head` instances
               (e.g., a ViT).
        rank, alpha: Forwarded to LoRALinear.
    """
    from basics.model import Head

    # Freeze every existing parameter; LoRALinear will then introduce the only
    # trainable tensors (A, B) when it wraps q_proj / v_proj.
    for p in model.parameters():
        p.requires_grad = False

    for module in model.modules():
        if isinstance(module, Head):
            module.q_proj = LoRALinear(module.q_proj, rank, alpha)
            module.v_proj = LoRALinear(module.v_proj, rank, alpha)
    return model


if __name__ == "__main__":
    # Parameter-count report for the §4 ViT (matches configs/clip_eurosat.yaml)
    # with LoRA rank 8.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from basics.vit import ViT

    rank, alpha = 8, 16.0
    model = ViT(
        img_size=64,
        patch_size=8,
        d_model=384,
        num_heads=6,
        num_blocks=6,
        dropout=0.1,
    )
    model = apply_lora_to_attention(model, rank=rank, alpha=alpha)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = trainable / total

    print(f"ViT:               img_size=64, patch_size=8, d_model=384, "
          f"num_heads=6, num_blocks=6")
    print(f"LoRA rank / alpha: {rank} / {alpha}")
    print(f"(i)   total parameters:     {total:,}")
    print(f"(ii)  trainable parameters: {trainable:,}")
    print(f"(iii) trainable / total:    {ratio:.4%}")

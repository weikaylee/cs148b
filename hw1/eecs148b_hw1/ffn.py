import torch
import torch.nn as nn


from eecs148b_hw1.linear import Linear


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        """Construct a standard 2-layer Transformer feed-forward network."""
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model if d_ff is None else d_ff

        self.fc1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `fc2(relu(fc1(x)))` with manual ReLU implementation."""
        hidden = self.fc1(x) # invokes forward()
        zeros = torch.zeros_like(hidden)
        hidden = torch.where(hidden > 0, hidden, zeros)
        return self.fc2(hidden)

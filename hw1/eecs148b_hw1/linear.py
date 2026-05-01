import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # Variance = 2 / (d_in + d_out), truncated at [-3σ, 3σ].
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transformation to the input."""
        return x @ self.weights.T
    

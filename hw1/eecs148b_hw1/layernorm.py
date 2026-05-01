import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """Construct the LayerNorm module."""
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weights = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.biases = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of
        shape (batch size, sequence length, d model) and return a tensor of the same
        shape."""
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        mean = x_fp32.mean(dim=-1, keepdim=True) # dim of (batch size, sequeeunce length, 1)
        var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x_fp32 - mean) / torch.sqrt(var + self.eps) # think about the dims!! 

        # Apply affine transform in fp32, then cast back to original dtype.
        w_fp32 = self.weights.to(torch.float32)
        b_fp32 = self.biases.to(torch.float32)
        result = x_hat * w_fp32 + b_fp32
        return result.to(in_dtype)

        
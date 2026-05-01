import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Shape: (max_seq_len, d_model)
        encodings = torch.zeros((max_seq_len, d_model), device=device, dtype=torch.float32)

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(1)
        two_i = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        div_term = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * (two_i / d_model))

        encodings[:, 0::2] = torch.sin(positions * div_term)
        encodings[:, 1::2] = torch.cos(positions * div_term)

        if dtype is not None:
            encodings = encodings.to(dtype)

        # Fixed (non-learned) positional embeddings.
        self.register_buffer("encodings", encodings)

    def forward(self, token_positions: torch.Tensor) -> torch.Tensor:
        """Given token positions of shape (..., seq_len), return (..., seq_len, d_model)."""
        encodings = self.get_buffer("encodings")
        return encodings[token_positions.long()]

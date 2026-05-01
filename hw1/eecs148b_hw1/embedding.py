import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """Construct an embedding transformation module."""
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embeddings = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.embeddings, mean=0.0, std=1.0, a=-3.0, b=3.0)
     
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply embedding lookup to token IDs."""
        return self.embeddings[x]




    

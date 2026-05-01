from __future__ import annotations

import torch
import torch.nn as nn

from eecs148b_hw1.attention import CausalMultiHeadSelfAttention
from eecs148b_hw1.embedding import Embedding
from eecs148b_hw1.layernorm import LayerNorm
from eecs148b_hw1.ffn import FFN
from eecs148b_hw1.sinusoidal_positional_encoding import SinusoidalPositionalEncoding


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.ln1 = LayerNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
        )
        self.ln2 = LayerNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# TODO check dimensions
# this is a decoder onyl transofmer
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.positional_embeddings = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_seq_len=context_length,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = LayerNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        seq_len = in_indices.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("Input sequence length exceeds context_length")

        x = self.token_embeddings(in_indices)

        # Build position indices with same leading batch dimensions as input.
        shape_prefix = [1] * (in_indices.ndim - 1)
        token_positions = torch.arange(seq_len, device=in_indices.device).view(*shape_prefix, seq_len)
        token_positions = token_positions.expand(*in_indices.shape)

        pos = self.positional_embeddings(token_positions).to(x.dtype)
        x = x + pos

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        return self.lm_head(x)

    @staticmethod
    def _sample_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Sample token IDs from a probability distribution with nucleus (top-p) filtering."""
        """It sorts tokens by probability, keeps only the smallest set whose cumulative probability exceeds top_p, renormalizes, and samples from that restricted set."""
        if top_p >= 1.0:
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Keep tokens while cumulative mass <= top_p, but always keep at least one token.
        keep = cumulative <= top_p
        keep[..., 0] = True

        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)

        sampled_in_sorted = torch.multinomial(filtered, num_samples=1)
        sampled = sorted_idx.gather(dim=-1, index=sampled_in_sorted).squeeze(-1)
        return sampled # shape (batch size,) (one sampled token per batch)
        # (rmember: batch means you are processing multiple input sequences in parallel.)

    @torch.no_grad()
    def decode(
        self,
        prompt_ids: torch.Tensor | list[int],
        *,
        eos_token_id: int | None = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively decode tokens from a prompt.

        Args:
            prompt_ids: Prompt token IDs of shape `(seq_len,)` or `(1, seq_len)`.
            eos_token_id: Optional end token ID to stop generation.
            max_new_tokens: Maximum number of new tokens to sample.
            temperature: Softmax temperature (>0). Lower is sharper, higher is flatter.
            top_p: Nucleus sampling threshold in (0, 1].

        Returns:
            Tensor of shape `(1, prompt_len + generated_len)` with generated token IDs.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if not (0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        if isinstance(prompt_ids, list):
            generated = torch.tensor(prompt_ids, dtype=torch.long)
        else:
            generated = prompt_ids.to(dtype=torch.long)

        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("prompt_ids must be shape (seq_len,) or (1, seq_len)") # here, batch siz emust be 1

        device = next(self.parameters()).device
        generated = generated.to(device) # (batch, seq_len)

        was_training = self.training
        self.eval()

        # this si quenctial (autoregressive) sampling!
        for _ in range(max_new_tokens):
            # Use only the latest context window.
            model_input = generated[:, -self.context_length :]
            logits = self(model_input) # runs through forward pass of transofmrer lm, to get shape (batch, seq_len, vocab_size)
            next_logits = logits[:, -1, :] # shape (batch, vocab_size)

            scaled = next_logits / temperature
            probs = torch.softmax(scaled, dim=-1)
            next_token = self._sample_top_p(probs, top_p=top_p) #(batch, ) (one token id per batch)

            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1) # append the token to each sequeunce in gnenerated, producing (batch, seq_len + 1)

            if eos_token_id is not None and next_token.item() == eos_token_id:  #.item() converts 1-element tensor to python scalar (since batch size = 1 here, so (1, 1) is shape of next_token after getting unsequeeze in line above)
                break

        if was_training:
            self.train()

        return generated # (batch size = 1, prompt_len + gneerated_len)


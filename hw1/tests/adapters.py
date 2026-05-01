from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from eecs148b_hw1 import bpe_trainer, tokenizer, linear, embedding, layernorm, ffn, attention, transformer
from eecs148b_hw1.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    Given the weights of a Linear layer, compute the transformation of a batched input.

    Args:
        in_dim (int): The size of the input dimension
        out_dim (int): The size of the output dimension
        weights (Float[Tensor, "d_out d_in"]): The linear weights to use
        in_features (Float[Tensor, "... d_in"]): The output tensor to apply the function to

    Returns:
        Float[Tensor, "... d_out"]: The transformed output of your linear module.
    """
    lin = linear.Linear(
        d_in,
        d_out,
        device=weights.device,
        dtype=weights.dtype,
    )
    lin.load_state_dict({"weights": weights}) # loads weights in lin's weight param
    return lin(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer

    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """
    emb =  embedding.Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    emb.load_state_dict({"embeddings": weights}) # load weights in embeddings
    return emb.forward(token_ids)


def run_ffn(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the weights of a 2-layer ReLU FFN, return
    the output of your implementation with these weights.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        d_ff (int): Inner dimensionality of the feed-forward network.
        w1_weight (Float[Tensor, "d_ff d_model"]): Stored weights for the first linear layer.
        w2_weight (Float[Tensor, "d_model d_ff"]): Stored weights for the second linear layer.
        in_features (Float[Tensor, "... d_model"]): Input embeddings to the feed-forward layer.

    Returns:
        Float[Tensor, "... d_model"]: Output embeddings of the same shape as the input embeddings.
    """
    # Example:
    # If your state dict keys match, you can use `load_state_dict()`
    # ffn.load_state_dict({"fc1.weight": w1_weight, "fc2.weight": w2_weight})
    # You can also manually assign the weights
    # ffn.fc1.weight.data = w1_weight
    # ffn.fc2.weight.data = w2_weight
    ffn_module = ffn.FFN(
        d_model=d_model,
        d_ff=d_ff,
        device=w1_weight.device,
        dtype=w1_weight.dtype,
    )
    ffn_module.load_state_dict({
        "fc1.weights": w1_weight,
        "fc2.weights": w2_weight,
    })
    return ffn_module(in_features)


def run_layernorm(
    d_model: int,
    eps: float,
    weight: Float[Tensor, " d_model"],
    bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """Given the affine parameters of LayerNorm, return the output of running
    LayerNorm on the input features.

    Args:
        d_model (int): The dimensionality of the LayerNorm input.
        eps (float): A value added to the denominator for numerical stability.
        weight (Float[Tensor, "d_model"]): LayerNorm scale parameters.
        bias (Float[Tensor, "d_model"]): LayerNorm bias parameters.
        in_features (Float[Tensor, "... d_model"]): Input features to run LayerNorm on.

    Returns:
        Float[Tensor, "... d_model"]: Tensor with the output of running LayerNorm on `in_features`.
    """
    ln = layernorm.LayerNorm(
        d_model=d_model,
        eps=eps,
        device=weight.device,
        dtype=weight.dtype,
    )
    ln.load_state_dict({"weights": weight, "biases": bias})
    return ln(in_features)

def run_sinusoidal_pe(
    d_model: int,
    max_seq_len: int,
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """Return sinusoidal positional embeddings for the given token positions."""
    pe = SinusoidalPositionalEncoding(
        d_model=d_model,
        max_seq_len=max_seq_len,
        device=token_positions.device,
        dtype=torch.float32,
    )
    return pe(token_positions)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    return attention.scaled_dot_product_attention(Q=Q, K=K, V=V, mask=mask)


# TODO code this yourselef!
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    """
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
    """
    attn = attention.CausalMultiHeadSelfAttention(
        d_model=d_model,
        num_heads=num_heads,
        device=q_proj_weight.device,
        dtype=q_proj_weight.dtype,
    )
    attn.load_state_dict(
        {
            "q_proj.weights": q_proj_weight,
            "k_proj.weights": k_proj_weight,
            "v_proj.weights": v_proj_weight,
            "output_proj.weights": o_proj_weight,
        }
    )
    return attn(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    Given the weights of a pre-norm Transformer block and input features,
    return the output of running the Transformer block on the input features.

    This function should use LayerNorm and the 2-layer ReLU FFN.

    Args:
        d_model (int): The dimensionality of the Transformer block input.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer.
        weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln1.bias`
                Bias of affine transform for the first LayerNorm.
                Shape is (d_model,).
            - `ffn.fc1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.fc2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln2.bias`
                Bias of affine transform for the second LayerNorm.
                Shape is (d_model,).
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            Tensor to run your implementation on.

    Returns:
        Float[Tensor, "batch sequence_length d_model"] Tensor with the output of
        running the Transformer block on the input features.
    """
    block = transformer.TransformerBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        device=in_features.device,
        dtype=in_features.dtype,
    )

    block.load_state_dict(
        {
            "attn.q_proj.weights": weights["attn.q_proj.weight"],
            "attn.k_proj.weights": weights["attn.k_proj.weight"],
            "attn.v_proj.weights": weights["attn.v_proj.weight"],
            "attn.output_proj.weights": weights["attn.output_proj.weight"],
            "ln1.weights": weights["ln1.weight"],
            "ln1.biases": weights["ln1.bias"],
            "ffn.fc1.weights": weights["ffn.fc1.weight"],
            "ffn.fc2.weights": weights["ffn.fc2.weight"],
            "ln2.weights": weights["ln2.weight"],
            "ln2.biases": weights["ln2.bias"],
        }
    )

    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """Given the weights of a Transformer language model and input indices,
    return the output of running a forward pass on the input indices.

    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ln1.bias`
                Bias of affine transform for the first LayerNorm.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.fc1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.fc2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second LayerNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ln2.bias`
                Bias of affine transform for the second LayerNorm.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for LayerNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `ln_final.bias`
                Bias of affine transform for the final LayerNorm.
                Shape is (d_model,).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.

    Returns:
        Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
        next-word distribution for each token.
    """
    model = transformer.TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        device=in_indices.device,
        dtype=weights["token_embeddings.weight"].dtype,
    )

    mapped_state = {
        "token_embeddings.embeddings": weights["token_embeddings.weight"],
        "ln_final.weights": weights["ln_final.weight"],
        "ln_final.biases": weights["ln_final.bias"],
        "lm_head.weight": weights["lm_head.weight"],
    }

    for i in range(num_layers):
        mapped_state.update(
            {
                f"layers.{i}.attn.q_proj.weights": weights[f"layers.{i}.attn.q_proj.weight"],
                f"layers.{i}.attn.k_proj.weights": weights[f"layers.{i}.attn.k_proj.weight"],
                f"layers.{i}.attn.v_proj.weights": weights[f"layers.{i}.attn.v_proj.weight"],
                f"layers.{i}.attn.output_proj.weights": weights[f"layers.{i}.attn.output_proj.weight"],
                f"layers.{i}.ln1.weights": weights[f"layers.{i}.ln1.weight"],
                f"layers.{i}.ln1.biases": weights[f"layers.{i}.ln1.bias"],
                f"layers.{i}.ffn.fc1.weights": weights[f"layers.{i}.ffn.fc1.weight"],
                f"layers.{i}.ffn.fc2.weights": weights[f"layers.{i}.ffn.fc2.weight"],
                f"layers.{i}.ln2.weights": weights[f"layers.{i}.ln2.weight"],
                f"layers.{i}.ln2.biases": weights[f"layers.{i}.ln2.bias"],
            }
        )

    model.load_state_dict(mapped_state, strict=False)
    return model(in_indices)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    max_start = len(dataset) - context_length
    assert max_start > 0, "dataset must be longer than context_length"

    start_indices = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([dataset[s : s + context_length] for s in start_indices], axis=0)
    y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in start_indices], axis=0)

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
    

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    return attention.softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Support arbitrary batch-like prefix dimensions before the vocab dimension.
    vocab_size = inputs.shape[-1]
    flat_inputs = inputs.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    # Numerical stability: subtract max logit per example before exp.
    row_max = torch.max(flat_inputs, dim=-1, keepdim=True).values
    shifted = flat_inputs - row_max

    # logsumexp(shifted) = log(sum(exp(shifted))).
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) # we don't keep the last dim, so dims of logsumexp are (n, )

    # Gather target logit from shifted logits.
    target_logits = shifted.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1) # unsqueezing adds a dim (changing flat_targets from (n, ) to (n, 1)) and squeeze returns to (n, )

    # -log softmax(target) = -(target - logsumexp)
    losses = -(target_logits - logsumexp) # so, we can subtract correclty here!
    return losses.mean()


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    return tokenizer.Tokenizer(vocab, merges, special_tokens) 


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    with open(input_path, encoding="utf-8") as f:
        corpus = f.read().splitlines()

    bpe = bpe_trainer.BPETrainer(corpus, vocab_size, special_tokens)
    bpe.train_bpe()
    return (bpe.vocab, bpe.merges)

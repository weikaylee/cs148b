from __future__ import annotations

"""Generate a TinyStories text dump from the tuned best checkpoint.

How to run from the project root:

1) With uv (recommended in this repo):
    uv run python -m eecs148b_hw1.generate_text_dump

2) With plain Python (if the package is already installed):
    python -m eecs148b_hw1.generate_text_dump

Output:
- Writes a text dump file to:
  checkpoints/final_single_tuned_online/generated_text_dump_256.txt
- Decoding stops at the first <|endoftext|> token or after `max_new_tokens`.
"""

import json
from pathlib import Path

import torch

from eecs148b_hw1.tokenizer import Tokenizer
from eecs148b_hw1.transformer import TransformerLM


def load_model(checkpoint_path: Path, device: str) -> tuple[TransformerLM, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = checkpoint["args"]

    model = TransformerLM(
        vocab_size=int(args["vocab_size"]),
        context_length=int(args["context_length"]),
        d_model=int(args["d_model"]),
        num_layers=int(args["num_layers"]),
        num_heads=int(args["num_heads"]),
        d_ff=int(args["d_ff"]),
        device="cpu",
        dtype=torch.float32,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, args


def main() -> None:
    checkpoint_path = Path("checkpoints/final_single_tuned_online/checkpoint_best.pt")
    output_path = Path("checkpoints/final_single_tuned_online/generated_text_dump_256.txt")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = Tokenizer.from_files(
        vocab_filepath="data/tinystories_bpe_10k/vocab.json",
        merges_filepath="data/tinystories_bpe_10k/merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    eos_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")

    model, _ = load_model(checkpoint_path, device)

    prompt = "Once upon a time, there was a dog named Clifford. It was big and red, and his best friend was a human Elizabeth."
    prompt_ids = tokenizer.encode(prompt)

    # Tuned for fluency/consistency tradeoff.
    temperature = 0.2
    top_p = 0.1
    max_new_tokens = 256

    sampled_ids = model.decode(
        prompt_ids,
        eos_token_id=eos_token_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )[0].detach().cpu().tolist()

    generated_tokens = len(sampled_ids) - len(prompt_ids)
    ended_with_eos = bool(eos_token_id is not None and sampled_ids[-1] == eos_token_id)
    decoded_text = tokenizer.decode(sampled_ids)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# TinyStories generation dump\n")
        f.write(f"checkpoint: {checkpoint_path}\n")
        f.write(f"device: {device}\n")
        f.write(f"prompt: {prompt!r}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"top_p: {top_p}\n")
        f.write(f"max_new_tokens: {max_new_tokens}\n")
        f.write(f"generated_tokens: {generated_tokens}\n")
        f.write(f"ended_with_eos: {ended_with_eos}\n")
        f.write("\n")
        f.write(decoded_text)
        f.write("\n")

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "generated_tokens": generated_tokens,
                "ended_with_eos": ended_with_eos,
                "temperature": temperature,
                "top_p": top_p,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

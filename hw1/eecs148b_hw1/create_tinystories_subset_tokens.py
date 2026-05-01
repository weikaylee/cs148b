from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eecs148b_hw1.tokenizer import Tokenizer


def tokenize_prefix(
    text_path: Path,
    tokenizer: Tokenizer,
    max_chars: int,
    output_path: Path,
) -> int:
    with text_path.open("r", encoding="utf-8") as f:
        text = f.read(max_chars)

    ids = tokenizer.encode(text)
    arr = np.asarray(ids, dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create tiny tokenized subsets for smoke-test LM training.")
    parser.add_argument("--vocab", type=Path, default=Path("data/tinystories_bpe_10k/vocab.json"))
    parser.add_argument("--merges", type=Path, default=Path("data/tinystories_bpe_10k/merges.txt"))
    parser.add_argument("--train-text", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--val-text", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--train-chars", type=int, default=1_000_000)
    parser.add_argument("--val-chars", type=int, default=200_000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/tinystories_subset"))
    args = parser.parse_args()

    tok = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )

    train_out = args.out_dir / "train_ids_subset.npy"
    val_out = args.out_dir / "val_ids_subset.npy"
    train_n = tokenize_prefix(args.train_text, tok, args.train_chars, train_out)
    val_n = tokenize_prefix(args.val_text, tok, args.val_chars, val_out)

    print(f"Wrote {train_out} with {train_n} tokens")
    print(f"Wrote {val_out} with {val_n} tokens")


if __name__ == "__main__":
    main()

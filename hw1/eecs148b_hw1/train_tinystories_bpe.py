from __future__ import annotations

import argparse
import json
from pathlib import Path

from eecs148b_hw1.bpe_trainer import BPETrainer

"""
Longest token id: 7160
Longest token bytes: 15
Longest token text: ' accomplishment'
Longest token hex: 206163636f6d706c6973686d656e74
"""

def _to_display_text(token: bytes) -> str:
    try:
        return token.decode("utf-8")
    except UnicodeDecodeError:
        return token.decode("latin-1")


def train_and_serialize(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = 10_000,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    with input_path.open("r", encoding="utf-8") as f:
        corpus = f.read().splitlines()

    trainer = BPETrainer(corpus=corpus, vocab_size=vocab_size, special_tokens=special_tokens)
    trainer.train_bpe()

    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = {
        str(token_id): {
            "utf8": _to_display_text(token_bytes),
            "hex": token_bytes.hex(),
            "length_bytes": len(token_bytes),
        }
        for token_id, token_bytes in trainer.vocab.items()
    }
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    with (output_dir / "merges.txt").open("w", encoding="utf-8") as f:
        for left, right in trainer.merges:
            f.write(f"{left.hex()} {right.hex()}\n")

    longest_id, longest_token = max(trainer.vocab.items(), key=lambda kv: len(kv[1]))
    summary = {
        "input_path": str(input_path),
        "vocab_size": len(trainer.vocab),
        "requested_vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "num_merges": len(trainer.merges),
        "longest_token_id": longest_id,
        "longest_token_num_bytes": len(longest_token),
        "longest_token_utf8": _to_display_text(longest_token),
        "longest_token_hex": longest_token.hex(),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return trainer.vocab, trainer.merges


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer on TinyStories and serialize outputs.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="Path to TinyStories training text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tinystories_bpe_10k"),
        help="Directory for serialized vocab/merges.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    args = parser.parse_args()

    vocab, _ = train_and_serialize(
        input_path=args.input_path,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
    )

    longest_id, longest_token = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(f"Serialized tokenizer to {args.output_dir}")
    print(f"Longest token id: {longest_id}")
    print(f"Longest token bytes: {len(longest_token)}")
    print(f"Longest token text: {_to_display_text(longest_token)!r}")
    print(f"Longest token hex: {longest_token.hex()}")


if __name__ == "__main__":
    main()

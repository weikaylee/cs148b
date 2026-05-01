from __future__ import annotations

import argparse
import json
import random
from array import array
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer

def iter_documents(input_path: Path, special_token: str):
    """Yield documents split on `special_token` without loading entire file in memory."""
    carry = ""
    with input_path.open("r", encoding="utf-8") as f:
        for chunk in f:
            carry += chunk
            parts = carry.split(special_token)
            carry = parts.pop()
            for part in parts:
                if part:
                    yield part
        if carry:
            yield carry


def reservoir_sample_documents(input_path: Path, special_token: str, k: int, seed: int):
    """Reservoir sample `k` documents from TinyStories."""
    rng = random.Random(seed)
    sample: list[str] = []
    seen = 0

    for doc in iter_documents(input_path, special_token):
        seen += 1
        if len(sample) < k:
            sample.append(doc)
        else:
            j = rng.randint(1, seen)
            if j <= k:
                sample[j - 1] = doc

    return sample


def compression_ratio_bytes_per_token(tokenizer: Tokenizer, documents: list[str]) -> tuple[float, int, int]:
    total_bytes = 0
    total_tokens = 0

    for doc in documents:
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(tokenizer.encode(doc))

    ratio = (total_bytes / total_tokens) if total_tokens > 0 else float("inf")
    return ratio, total_bytes, total_tokens


def encode_file_to_uint16(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> int:
    """Encode a text file to token IDs and save as .npy uint16 array (batched for speed)."""
    token_batches = []
    batch_size = 100_000
    current_batch = []

    with input_path.open("r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            if token_id < 0 or token_id > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Token ID {token_id} out of uint16 range. "
                    "Use a smaller vocab or a wider dtype."
                )
            current_batch.append(token_id)
            if len(current_batch) >= batch_size:
                token_batches.append(np.asarray(current_batch, dtype=np.uint16))
                current_batch = []
                print(f"  Tokenized {sum(b.size for b in token_batches)} tokens...")

    if current_batch:
        token_batches.append(np.asarray(current_batch, dtype=np.uint16))

    arr = np.concatenate(token_batches) if token_batches else np.array([], dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample docs, compute compression, and encode TinyStories splits.")
    parser.add_argument("--tokenizer-vocab", type=Path, default=Path("data/tinystories_bpe_10k/vocab.json"))
    parser.add_argument("--tokenizer-merges", type=Path, default=Path("data/tinystories_bpe_10k/merges.txt"))
    parser.add_argument("--train-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid-path", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--special-token", type=str, default="<|endoftext|>")
    parser.add_argument("--num-sample-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=148)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tinystories_tokenized"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.tokenizer_vocab,
        merges_filepath=args.tokenizer_merges,
        special_tokens=[args.special_token],
    )

    sample_docs = reservoir_sample_documents(
        input_path=args.train_path,
        special_token=args.special_token,
        k=args.num_sample_docs,
        seed=args.seed,
    )
    ratio, sample_total_bytes, sample_total_tokens = compression_ratio_bytes_per_token(tokenizer, sample_docs)

    train_out = args.output_dir / "train_ids_uint16.npy"
    valid_out = args.output_dir / "valid_ids_uint16.npy"
    print(f"Tokenizing {args.train_path}...")
    train_tokens = encode_file_to_uint16(tokenizer, args.train_path, train_out)
    print(f"Tokenized {train_tokens} train tokens -> {train_out}")
    print(f"Tokenizing {args.valid_path}...")
    valid_tokens = encode_file_to_uint16(tokenizer, args.valid_path, valid_out)
    print(f"Tokenized {valid_tokens} valid tokens -> {valid_out}")

    report = {
        "sample": {
            "num_docs": len(sample_docs),
            "total_bytes": sample_total_bytes,
            "total_tokens": sample_total_tokens,
            "compression_ratio_bytes_per_token": ratio,
            "seed": args.seed,
        },
        "tokenized_outputs": {
            "train_path": str(train_out),
            "train_num_tokens": train_tokens,
            "valid_path": str(valid_out),
            "valid_num_tokens": valid_tokens,
            "dtype": "uint16",
        },
        "uint16_reason": "With vocab size <= 10,000, token IDs are in [0, 9999], which fits in uint16 [0, 65535].",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "tokenization_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

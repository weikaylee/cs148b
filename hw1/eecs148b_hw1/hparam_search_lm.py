"""Grid-search entrypoint for LM optimizer hyperparameters.

How to run (from workspace root):

1) Quick search on pretokenized subset data:

     uv run python eecs148b_hw1/hparam_search_lm.py \
         --train-data data/tinystories_subset/train_ids_subset.npy \
         --val-data data/tinystories_subset/val_ids_subset.npy \
         --output-dir checkpoints/hparam_search_subset \
         --max-trials 4 --max-steps 5000

2) Search on raw text files (auto-tokenizes once, then reuses cached .npy files):

     uv run python eecs148b_hw1/hparam_search_lm.py \
         --train-data data/TinyStoriesV2-GPT4-train.txt \
         --val-data data/TinyStoriesV2-GPT4-valid.txt \
         --output-dir checkpoints/hparam_search_real \
         --tokenizer-vocab data/tinystories_bpe_10k/vocab.json \
         --tokenizer-merges data/tinystories_bpe_10k/merges.txt

3) Enable wandb logging:

     uv run python eecs148b_hw1/hparam_search_lm.py [args above] \
         --use-wandb --wandb-project eecs148b-hw1-tuned --wandb-run-name hparam-search

Outputs:
- Per-trial artifacts are written to --output-dir/trial_XXX.
- Aggregated ranking is written to --output-dir/search_results.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
from array import array
from pathlib import Path

import numpy as np

from eecs148b_hw1.train_lm import train_once
from eecs148b_hw1.tokenizer import Tokenizer


def encode_text_file_to_npy(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> Path:
    """Tokenize a raw text file and cache the result as a uint16 .npy file (batched for speed)."""
    token_batches = []
    batch_size = 100_000
    current_batch = []

    with input_path.open("r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            if token_id < 0 or token_id > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"Token ID {token_id} out of uint16 range; use a smaller vocab or a wider dtype."
                )
            current_batch.append(token_id)
            if len(current_batch) >= batch_size:
                token_batches.append(np.asarray(current_batch, dtype=np.uint16))
                current_batch = []

    if current_batch:
        token_batches.append(np.asarray(current_batch, dtype=np.uint16))

    arr = np.concatenate(token_batches) if token_batches else np.array([], dtype=np.uint16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, arr)
    return output_path


def resolve_dataset_path(path: Path, tokenizer: Tokenizer, cache_dir: Path) -> Path:
    """Return a token-id file path, converting raw text to cached tokens when needed."""
    if path.suffix == ".npy":
        return path

    if path.suffix == ".txt":
        cached_path = cache_dir / f"{path.stem}_ids_uint16.npy"
        if not cached_path.exists():
            print(f"Tokenizing {path} -> {cached_path}")
            encode_text_file_to_npy(tokenizer, path, cached_path)
        else:
            print(f"Reusing cached tokens: {cached_path}")
        return cached_path

    raise ValueError(f"Unsupported dataset format for {path}; use .txt or .npy")

def main() -> None:
    p = argparse.ArgumentParser(description="Simple grid search over LM optimizer hyperparameters.")
    p.add_argument("--train-data", type=Path, default=Path("data/tinystories_subset/train_ids_subset.npy"))
    p.add_argument("--val-data", type=Path, default=Path("data/tinystories_subset/val_ids_subset.npy"))
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints/hparam_search"))
    p.add_argument("--tokenizer-vocab", type=Path, default=Path("data/tinystories_bpe_10k/vocab.json"))
    p.add_argument("--tokenizer-merges", type=Path, default=Path("data/tinystories_bpe_10k/merges.txt"))
    p.add_argument("--special-token", type=str, default="<|endoftext|>")

    # Fixed architecture requested by user.
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=2048)

    # tokens_processed = batch_size * max_steps * context_length ~= 40,960,000
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=5000)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs148_hyperparam_search")
    p.add_argument("--wandb-run-name", type=str, default=None)

    # Search space (small grid by default; expand as needed)
    p.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4])
    p.add_argument("--warmup-steps-list", nargs="+", type=int, default=[100, 500])
    p.add_argument("--beta1-list", nargs="+", type=float, default=[0.9])
    p.add_argument("--beta2-list", nargs="+", type=float, default=[0.95, 0.99])
    p.add_argument("--adam-eps-list", nargs="+", type=float, default=[1e-8, 1e-6])
    p.add_argument("--weight-decay-list", nargs="+", type=float, default=[0.01, 0.1])
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--max-trials", type=int, default=4)

    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.tokenizer_vocab,
        merges_filepath=args.tokenizer_merges,
        special_tokens=[args.special_token],
    )
    preprocess_dir = args.output_dir / "preprocessed"
    args.train_data = resolve_dataset_path(args.train_data, tokenizer, preprocess_dir)
    args.val_data = resolve_dataset_path(args.val_data, tokenizer, preprocess_dir)

    grid = list(
        itertools.product(
            args.learning_rates,
            args.warmup_steps_list,
            args.beta1_list,
            args.beta2_list,
            args.adam_eps_list,
            args.weight_decay_list,
        )
    )

    all_results = []
    for trial_idx, (lr, warmup, b1, b2, eps, wd) in enumerate(grid[: args.max_trials], start=1):
        trial_dir = args.output_dir / f"trial_{trial_idx:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        run_args = argparse.Namespace(
            train_data=args.train_data,
            val_data=args.val_data,
            output_dir=trial_dir,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=lr,
            min_lr=lr * args.min_lr_ratio,
            warmup_steps=warmup,
            weight_decay=wd,
            beta1=b1,
            beta2=b2,
            adam_eps=eps,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            eval_steps=args.eval_steps,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            grad_clip=args.grad_clip,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=(f"trial_{trial_idx:03d}" if args.wandb_run_name is None else f"{args.wandb_run_name}_{trial_idx:03d}"),
        )

        print(
            f"\n[trial {trial_idx}] lr={lr} warmup={warmup} beta1={b1} "
            f"beta2={b2} eps={eps} wd={wd}"
        )
        result = train_once(run_args)
        val_loss = result.get("last_eval", {}).get("val_loss", float("inf"))

        trial_result = {
            "trial": trial_idx,
            "learning_rate": lr,
            "warmup_steps": warmup,
            "beta1": b1,
            "beta2": b2,
            "adam_eps": eps,
            "weight_decay": wd,
            "val_loss": val_loss,
            "tokens_processed": result.get("tokens_processed"),
            "checkpoint": result.get("final_checkpoint"),
        }
        all_results.append(trial_result)

        with (trial_dir / "result.json").open("w", encoding="utf-8") as f:
            json.dump(trial_result, f, indent=2)

    all_results.sort(key=lambda r: r["val_loss"])
    summary_path = args.output_dir / "search_results.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Hyperparameter search complete ===")
    print(f"Results written to {summary_path}")
    if all_results:
        print("Best trial:")
        print(json.dumps(all_results[0], indent=2))


if __name__ == "__main__":
    main()

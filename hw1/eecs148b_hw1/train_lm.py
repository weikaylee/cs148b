"""Training entrypoint for language-model experiments.

How to run (from workspace root):

1) Standard Transformer (sinusoidal positional encoding):

    uv run python eecs148b_hw1/train_lm.py \
      --train-data checkpoints/hparam_search_real/preprocessed/TinyStoriesV2-GPT4-train_ids_uint16.npy \
      --val-data checkpoints/hparam_search_real/preprocessed/TinyStoriesV2-GPT4-valid_ids_uint16.npy \
      --output-dir checkpoints/final_single_tuned_online \
      --vocab-size 10000 --context-length 256 --d-model 768 --num-layers 6 --num-heads 12 --d-ff 3072 \
      --device mps --batch-size 24 --learning-rate 8e-5 --min-lr 8e-6 --warmup-steps 200 \
      --weight-decay 0.05 --beta1 0.9 --beta2 0.98 --adam-eps 1e-8 \
      --max-steps 8000 --eval-interval 100 --eval-steps 20 --log-interval 100 --save-interval 500 \
      --grad-clip 1.0 --model-variant standard --use-wandb --wandb-project eecs148b-hw1-tuned

2) No LayerNorm ablation:

    uv run python eecs148b_hw1/train_lm.py [same args as above] --model-variant no_layernorm

3) No positional encoding (NoPE) ablation:

    uv run python eecs148b_hw1/train_lm.py [same args as above] --model-variant no_positional

Notes:
- Use --device auto to let the script choose cuda/mps/cpu.
- Checkpoints, run config, and training history are written to --output-dir.
- If wandb is unavailable, training continues without external logging.
"""

from __future__ import annotations

import argparse
import json
import time
import math
from pathlib import Path
from typing import Any, cast
from dataclasses import dataclass, field
from contextlib import suppress

import numpy as np
import torch
import torch.nn.functional as F

from eecs148b_hw1.no_layernorm_transformer import NoLayerNormTransformerLM
from eecs148b_hw1.no_positional_transformer import NoPositionalEncodingTransformerLM
from eecs148b_hw1.transformer import TransformerLM


@dataclass
class ExperimentLogger:
    output_dir: Path
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "training_log.jsonl"
        self.curve_path = self.output_dir / "loss_curve.png"

    def log(self, record: dict[str, Any]) -> None:
        self.history.append(record)
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self) -> None:
        if not self.history:
            return

        with (self.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        with suppress(Exception):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            eval_rows = [row for row in self.history if row.get("phase") == "eval"]
            if not eval_rows:
                return

            steps = [row["step"] for row in eval_rows]
            train_loss = [row["train_loss"] for row in eval_rows]
            val_loss = [row["val_loss"] for row in eval_rows]

            plt.figure(figsize=(8, 5))
            plt.plot(steps, train_loss, label="train loss")
            plt.plot(steps, val_loss, label="val loss")
            plt.xlabel("step")
            plt.ylabel("loss")
            plt.title("Training and validation loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.curve_path, dpi=150)
            plt.close()


def _serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert argparse Namespace values to JSON-serializable primitives."""
    out: dict[str, Any] = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def load_memmap(path: Path) -> np.ndarray:
    """Load token IDs with memory mapping."""
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.memmap(path, dtype=np.uint16, mode="r")
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("Dataset must be longer than context_length")

    starts = np.random.randint(0, max_start, size=batch_size)
    x_np = np.stack([dataset[s : s + context_length] for s in starts], axis=0)
    y_np = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts], axis=0)

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    eval_steps: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split_name, split_data in (("train", train_data), ("val", val_data)):
        losses = []
        accuracies = []
        for _ in range(eval_steps):
            x, y = get_batch(split_data, batch_size, context_length, device)
            logits = model(x)
            # Reshape for cross entropy: (batch * seq_len, vocab_size)
            logits_flat = logits.reshape(-1, vocab_size)
            y_flat = y.reshape(-1)
            loss = F.cross_entropy(logits_flat, y_flat)
            losses.append(loss.item())

            # Accuracy
            preds = torch.argmax(logits_flat, dim=-1)
            acc = (preds == y_flat).float().mean()
            accuracies.append(acc.item())

        out[f"{split_name}_loss"] = float(np.mean(losses))
        out[f"{split_name}_accuracy"] = float(np.mean(accuracies))
    model.train()
    return out


def maybe_init_wandb(args: argparse.Namespace):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except Exception:
        print("wandb not available; continuing without wandb logging")
        return None

    wandb_module = cast(Any, wandb)
    init_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "config": _serialize_args(args),
    }

    try:
        wandb_module.init(**init_kwargs)
    except Exception as exc:
        print(f"wandb init failed ({exc}); retrying in offline mode")
        try:
            wandb_module.init(mode="offline", **init_kwargs)
        except Exception as offline_exc:
            print(f"wandb offline init failed ({offline_exc}); continuing without wandb logging")
            return None
    return wandb_module


def compute_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int, min_lr: float) -> float:
    """Linear warmup followed by cosine decay to min_lr."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)

    if max_steps <= warmup_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (base_lr - min_lr)


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": _serialize_args(args),
        },
        path,
    )


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Transformer LM with memmapped token datasets.")

    # Data / IO
    p.add_argument("--train-data", type=Path, required=True, help="Path to train token IDs (.npy preferred)")
    p.add_argument("--val-data", type=Path, required=True, help="Path to val token IDs (.npy preferred)")
    p.add_argument("--output-dir", type=Path, default=Path("checkpoints"))

    # Model hyperparameters
    p.add_argument("--vocab-size", type=int, required=True)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=1024)
    p.add_argument(
        "--model-variant",
        type=str,
        choices=["standard", "no_layernorm", "no_positional"],
        default="standard",
        help="Choose the Transformer variant to train.",
    )

    # Optimizer / training hyperparameters
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Logging
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="eecs148b-hw1")
    p.add_argument("--wandb-run-name", type=str, default=None)

    return p.parse_args()


def train_once(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "auto":
        args.device = get_default_device()
    print(f"Using device: {args.device}")

    train_data = load_memmap(args.train_data)
    val_data = load_memmap(args.val_data)

    model_variants = {
        "standard": TransformerLM,
        "no_layernorm": NoLayerNormTransformerLM,
        "no_positional": NoPositionalEncodingTransformerLM,
    }
    model_cls = model_variants[args.model_variant]
    model = model_cls(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=args.device,
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    wandb = maybe_init_wandb(args)
    logger = ExperimentLogger(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(_serialize_args(args), f, indent=2)

    model.train()
    t0 = time.time()
    last_eval: dict[str, float] = {"train_loss": float("nan"), "val_loss": float("nan")}
    best_val_loss = float("inf")
    best_checkpoint_path = args.output_dir / "checkpoint_best.pt"
    tokens_per_step = args.batch_size * args.context_length

    for step in range(1, args.max_steps + 1):
        lr = compute_lr(
            step=step,
            max_steps=args.max_steps,
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_lr=args.min_lr,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        logits_flat = logits.reshape(-1, args.vocab_size)
        y_flat = y.reshape(-1)
        loss = F.cross_entropy(logits_flat, y_flat)
        loss.backward()

        if args.grad_clip is not None and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        with torch.no_grad():
            preds = torch.argmax(logits_flat, dim=-1)
            train_acc = (preds == y_flat).float().mean().item()

        elapsed = time.time() - t0
        processed_tokens = step * tokens_per_step

        log_record = {
            "phase": "train",
            "step": step,
            "lr": lr,
            "tokens_processed": processed_tokens,
            "elapsed_s": elapsed,
            "batch_loss": loss.item(),
            "batch_accuracy": train_acc,
        }
        logger.log(log_record)

        metrics: dict[str, float] | None = None
        should_eval = step % args.eval_interval == 0 or step == args.max_steps
        if should_eval:
            metrics = evaluate(
                model=model,
                train_data=train_data,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                vocab_size=args.vocab_size,
                eval_steps=args.eval_steps,
                device=args.device,
            )
            last_eval = metrics

            eval_record = {
                "phase": "eval",
                "step": step,
                "lr": lr,
                "tokens_processed": processed_tokens,
                "elapsed_s": elapsed,
                "train_loss": metrics["train_loss"],
                "train_accuracy": metrics["train_accuracy"],
                "val_loss": metrics["val_loss"],
                "val_accuracy": metrics["val_accuracy"],
            }
            logger.log(eval_record)

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(best_checkpoint_path, model, optimizer, step, args)
                with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "step": step,
                            "val_loss": metrics["val_loss"],
                            "train_loss": metrics["train_loss"],
                            "train_accuracy": metrics["train_accuracy"],
                            "val_accuracy": metrics["val_accuracy"],
                            "checkpoint": str(best_checkpoint_path),
                        },
                        f,
                        indent=2,
                    )

            if step % args.log_interval == 0 or step == args.max_steps:
                print(
                    f"step={step} train_loss_batch={loss.item():.4f} train_acc_batch={train_acc:.4f} "
                    f"train_loss={metrics['train_loss']:.4f} train_acc={metrics['train_accuracy']:.4f} "
                    f"val_loss={metrics['val_loss']:.4f} val_acc={metrics['val_accuracy']:.4f} "
                    f"lr={lr:.3e} tokens={processed_tokens} elapsed_s={elapsed:.1f}"
                )
            if wandb is not None:
                cast(Any, wandb).log(
                    {
                        "step": step,
                        "lr": lr,
                        "tokens_processed": processed_tokens,
                        "elapsed_s": elapsed,
                        "training/batch_loss": loss.item(),
                        "training/batch_accuracy": train_acc,
                        "training/loss": metrics["train_loss"],
                        "training/accuracy": metrics["train_accuracy"],
                        "validation/loss": metrics["val_loss"],
                        "validation/accuracy": metrics["val_accuracy"],
                    }
                )
        elif step % args.log_interval == 0 or step == args.max_steps:
            print(
                f"step={step} train_loss_batch={loss.item():.4f} train_acc_batch={train_acc:.4f} "
                f"lr={lr:.3e} tokens={processed_tokens} elapsed_s={elapsed:.1f} (eval skipped until step {args.eval_interval})"
            )

        if step % args.save_interval == 0 or step == args.max_steps:
            ckpt_path = args.output_dir / f"checkpoint_step_{step}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step, args)

    final_path = args.output_dir / "checkpoint_final.pt"
    save_checkpoint(final_path, model, optimizer, args.max_steps, args)
    print(f"Training complete. Final checkpoint: {final_path}")

    logger.finalize()

    if wandb is not None:
        cast(Any, wandb).finish()

    return {
        "final_checkpoint": str(final_path),
        "last_eval": last_eval,
        "tokens_processed": args.max_steps * tokens_per_step,
    }


def main() -> None:
    args = parse_args()
    result = train_once(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

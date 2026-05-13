"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    # Train one configuration:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt

    # After training all three methods, print the comparison table:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --pretrained runs/clip_eurosat/best.pt --summarize

Each run writes `metrics.json` into its output directory. `--summarize` reads
all three and prints the deliverable table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from basics.lora import apply_lora_to_attention  # noqa: E402
from basics.vit import ViT  # noqa: E402
from vlm.data import build_resisc45_loaders  # noqa: E402


def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


class ViTClassifier(nn.Module):
    """ViT backbone + linear classification head on the CLS embedding."""

    def __init__(self, vit: ViT, num_classes: int) -> None:
        super().__init__()
        self.vit = vit
        self.head = nn.Linear(vit.d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.vit(x))


def configure_backbone(vit: ViT, method: str, rank: int, alpha: float) -> None:
    """In-place: freeze/unfreeze ViT params + (optionally) attach LoRA."""
    if method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad = False
    elif method == "lora":
        # apply_lora_to_attention first freezes everything, then introduces
        # trainable A/B in q_proj and v_proj.
        apply_lora_to_attention(vit, rank=rank, alpha=alpha)
    elif method == "full_ft":
        for p in vit.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown method: {method!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.numel()
    model.train(was_training)
    return correct / max(total, 1)


def train_one(args: argparse.Namespace, cfg: dict) -> dict:
    device = torch.device(args.device)
    method = args.method

    # The pretrained checkpoint owns the ViT architecture spec — pull from it
    # so the loaded weights always match the rebuilt module.
    ckpt = torch.load(args.pretrained, map_location="cpu")
    vit_cfg = ckpt["config"]["vit"]
    num_classes = cfg["num_classes"]

    train_dl, test_dl = build_resisc45_loaders(
        img_size=vit_cfg["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    vit = ViT(**vit_cfg)
    vit.load_state_dict(ckpt["vit"])
    configure_backbone(vit, method, rank=args.rank, alpha=args.alpha)

    model = ViTClassifier(vit, num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Default to the shared `optim.lr` for all three methods (the spec's
    # "same learning rate" requirement). --lr overrides per run; per-method
    # `methods.<x>.lr` is *not* auto-applied — it's just documented in the
    # YAML as suggested tuning if a method underperforms.
    if args.lr is not None:
        method_lr = args.lr
    else:
        method_lr = cfg["optim"]["lr"]
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=method_lr,
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    total_steps = cfg["train"]["num_epochs"] * len(train_dl)
    scheduler = LambdaLR(
        optimizer, cosine_warmup_lambda(cfg["optim"]["warmup_steps"], total_steps)
    )

    # Peak-memory window starts now (after model + optimizer construction so
    # we capture the activation/grad/optimizer-state peak during training).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = {"epoch": [], "train_loss": [], "test_acc": []}
    t_start = time.time()
    for epoch in range(cfg["train"]["num_epochs"]):
        model.train()
        running_loss, n_batches = 0.0, 0
        pbar = tqdm(
            train_dl,
            desc=f"[{method}] epoch {epoch + 1}/{cfg['train']['num_epochs']}",
            leave=False,
        )
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(n_batches, 1)
        test_acc = evaluate(model, test_dl, device)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["test_acc"].append(test_acc)
        print(
            f"[{method}] epoch {epoch + 1}/{cfg['train']['num_epochs']}  "
            f"train_loss={epoch_loss:.4f}  test_acc={test_acc:.4f}"
        )

    train_time_s = time.time() - t_start
    peak_mem_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    metrics = {
        "method": method,
        "rank": args.rank if method == "lora" else None,
        "alpha": args.alpha if method == "lora" else None,
        "lr": method_lr,
        "num_epochs": cfg["train"]["num_epochs"],
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": trainable_params / total_params,
        "final_test_acc": history["test_acc"][-1],
        "best_test_acc": max(history["test_acc"]),
        "peak_mem_bytes": int(peak_mem_bytes),
        "peak_mem_mb": peak_mem_bytes / (1024 ** 2),
        "train_time_s": train_time_s,
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {out_path}")
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _default_run_dir(runs_root: Path, method: str, rank: int) -> Path:
    if method == "lora":
        return runs_root / f"resisc_lora_rank{rank}"
    return runs_root / f"resisc_{method}"


def print_table(metrics_by_method: dict[str, dict]) -> None:
    cols = [
        ("method",       "method",           14),
        ("lr",           "lr",                9),
        ("test_acc",     "final_test_acc",   10),
        ("trainable",    "trainable_params", 14),
        ("peak_mem_MB",  "peak_mem_mb",      12),
        ("time_s",       "train_time_s",     10),
    ]
    header = " | ".join(f"{name:>{w}}" for name, _, w in cols)
    print()
    print(header)
    print("-" * len(header))
    for method in ("linear_probe", "lora", "full_ft"):
        m = metrics_by_method.get(method)
        if m is None:
            continue
        row = []
        for name, key, w in cols:
            v = m[key]
            if name == "test_acc":
                s = f"{v:.4f}"
            elif name == "trainable":
                s = f"{v:,}"
            elif name == "peak_mem_MB":
                s = f"{v:.1f}"
            elif name == "time_s":
                s = f"{v:.1f}"
            elif name == "lr":
                s = f"{v:.0e}"
            else:
                s = str(v)
            row.append(f"{s:>{w}}")
        print(" | ".join(row))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], default=None)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override the learning rate. Defaults to `optim.lr` in the YAML "
                        "(shared across all three methods). Pass an explicit value to "
                        "tune per-method and report the tuning.")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--summarize", action="store_true",
                   help="Skip training; read metrics.json for all three methods and print the table.")
    p.add_argument("--runs-root", type=Path, default=Path("runs"),
                   help="Where per-method run directories live (used with --summarize).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.summarize:
        metrics_by_method: dict[str, dict] = {}
        for method in ("linear_probe", "lora", "full_ft"):
            run_dir = _default_run_dir(args.runs_root, method, args.rank)
            mp = run_dir / "metrics.json"
            if mp.exists():
                metrics_by_method[method] = json.loads(mp.read_text())
            else:
                print(f"[warn] no metrics at {mp}")
        print_table(metrics_by_method)
        return

    if args.method is None:
        raise SystemExit("--method is required (or pass --summarize).")

    if args.output_dir is None:
        args.output_dir = _default_run_dir(Path("runs"), args.method, args.rank)

    metrics = train_one(args, cfg)
    print_table({args.method: metrics})


if __name__ == "__main__":
    main()

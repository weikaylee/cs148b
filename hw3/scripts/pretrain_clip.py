"""§3 — CLIP-style pretraining on EuroSAT.

Trains your ViT (basics.vit.ViT) against a frozen text encoder
(basics.text_encoder.FrozenTextEncoder) with a symmetric InfoNCE loss
(vlm.clip.clip_loss). Per-epoch zero-shot validation accuracy is computed via
vlm.eval.zeroshot_classification_accuracy on the 10 EuroSAT class prompts.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml --wandb

The same `train(cfg, args)` entry point can be imported and called from a
notebook (see notebooks/clip_pretrain.ipynb).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import yaml

# Make basics/ and vlm/ importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from basics.text_encoder import FrozenTextEncoder  # noqa: E402
from basics.vit import ViT  # noqa: E402
from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale  # noqa: E402
from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders  # noqa: E402
from vlm.eval import zeroshot_classification_accuracy  # noqa: E402


def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup for `warmup_steps`, then cosine decay to 0 over the rest."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _split_decay_params(modules):
    """Decay 2D+ tensors (weights), exclude biases / 1D params (norms etc.)."""
    decay, nodecay = [], []
    for m in modules:
        for p in m.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else nodecay).append(p)
    return decay, nodecay


def train(cfg: dict, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # CLI override for PE choice (§6 rope-vs-learned ablation).
    if getattr(args, "pos_encoding", None) is not None:
        cfg = {**cfg, "vit": {**cfg["vit"], "pos_encoding": args.pos_encoding}}

    # -------- W&B --------
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**cfg, "output_dir": str(args.output_dir)},
        )

    # -------- Data --------
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    # -------- Models --------
    vit = ViT(**cfg["vit"]).to(device)
    text_encoder = FrozenTextEncoder(**cfg["text_encoder"]).to(device)
    proj = ProjectionHeads(
        d_image=cfg["vit"]["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=cfg["projection"]["d_proj"],
    ).to(device)
    # `.to(device)` on a Parameter produces a non-leaf tensor and loses the
    # Parameter wrapper, which breaks AdamW. Rebuild as a leaf Parameter on the
    # target device.
    logit_scale = torch.nn.Parameter(init_logit_scale().detach().to(device))

    # -------- Optimizer --------
    decay, nodecay = _split_decay_params([vit, proj])
    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": cfg["optim"]["weight_decay"]},
            {"params": nodecay, "weight_decay": 0.0},
            {"params": [logit_scale], "weight_decay": 0.0},
        ],
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    total_steps = cfg["train"]["num_epochs"] * len(train_dl)
    scheduler = LambdaLR(
        optimizer, cosine_warmup_lambda(cfg["optim"]["warmup_steps"], total_steps)
    )

    # -------- Zero-shot eval setup --------
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    # -------- Loop --------
    history = {
        "epoch": [],
        "train_loss": [],
        "val_acc": [],
        "lr": [],
        "logit_scale": [],
    }
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(cfg["train"]["num_epochs"]):
        vit.train()
        proj.train()
        running_loss = 0.0
        n_batches = 0
        t_start = time.time()

        pbar = tqdm(
            train_dl,
            desc=f"epoch {epoch + 1}/{cfg['train']['num_epochs']}",
            leave=False,
        )
        for images, captions in pbar:
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                # `.clone()` lifts the tensor out of inference mode so it can
                # be saved for backward through the trainable text projection.
                text_embeds = text_encoder(captions).to(device).clone()

            image_embeds = vit(images)
            img_p, txt_p = proj(image_embeds, text_embeds)
            loss = clip_loss(img_p, txt_p, logit_scale)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Clamp logit_scale so the temperature can't run away.
            with torch.no_grad():
                logit_scale.data.clamp_(max=math.log(100.0))

            running_loss += loss.item()
            n_batches += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if wandb_run is not None and global_step % cfg["train"]["log_every"] == 0:
                wandb_run.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/logit_scale_exp": logit_scale.exp().item(),
                        "step": global_step,
                    }
                )

        epoch_loss = running_loss / max(n_batches, 1)
        val_acc = zeroshot_classification_accuracy(
            vit,
            proj,
            text_encoder,
            val_dl,
            class_prompts=class_prompts,
            class_indices=class_indices,
            device=device,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["logit_scale"].append(logit_scale.exp().item())

        elapsed = time.time() - t_start
        print(
            f"[epoch {epoch + 1}/{cfg['train']['num_epochs']}] "
            f"train_loss={epoch_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  ({elapsed:.1f}s)"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/epoch_loss": epoch_loss,
                    "val/zeroshot_acc": val_acc,
                    "epoch_time_s": elapsed,
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "vit": vit.state_dict(),
                    "projection_heads": proj.state_dict(),
                    "logit_scale": logit_scale.detach().cpu(),
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "config": cfg,
                },
                args.output_dir / "best.pt",
            )

    # -------- Final test eval --------
    test_acc = zeroshot_classification_accuracy(
        vit,
        proj,
        text_encoder,
        test_dl,
        class_prompts=class_prompts,
        class_indices=class_indices,
        device=device,
    )
    history["best_val_acc"] = best_val_acc
    history["test_acc"] = test_acc
    print(f"[final] best_val_acc={best_val_acc:.4f}  test_acc={test_acc:.4f}")

    # -------- §6 rope-vs-learned: length-extrapolation eval --------
    eval_img_size = getattr(args, "eval_img_size", None)
    if eval_img_size is not None and eval_img_size != cfg["vit"]["img_size"]:
        print(f"[extrapolation] running zero-shot eval at img_size={eval_img_size}")
        # New val / test loaders at the larger resolution (same stratified split).
        _, val_dl_large, test_dl_large = build_eurosat_loaders(
            img_size=eval_img_size,
            batch_size=cfg["train"]["batch_size"],
            num_workers=cfg["train"]["num_workers"],
        )
        # For RoPE, ensure the cos/sin cache covers the new sequence length.
        # Learned PE is handled transparently in ViT.forward via interpolation.
        pe = cfg["vit"].get("pos_encoding", "learned")
        if pe == "rope":
            new_grid = eval_img_size // cfg["vit"]["patch_size"]
            new_tokens = new_grid * new_grid + 1
            if new_tokens > vit.rope_max_seq_len:
                vit.set_rope_max_seq_len(new_tokens)
                print(f"  extended RoPE cache to {new_tokens} tokens")
        elif pe == "rope2d":
            new_grid = eval_img_size // cfg["vit"]["patch_size"]
            # CLS uses (0, 0) and patches occupy 1..new_grid, so we need at
            # least new_grid + 1 slots per axis.
            needed = new_grid + 1
            if needed > vit.rope_grid_size:
                vit.set_rope_grid_size(needed)
                print(f"  extended RoPE2D grid cache to {needed} per axis")

        extra_val_acc = zeroshot_classification_accuracy(
            vit, proj, text_encoder, val_dl_large,
            class_prompts=class_prompts, class_indices=class_indices, device=device,
        )
        extra_test_acc = zeroshot_classification_accuracy(
            vit, proj, text_encoder, test_dl_large,
            class_prompts=class_prompts, class_indices=class_indices, device=device,
        )
        history["extrapolation"] = {
            "img_size": eval_img_size,
            "val_acc": extra_val_acc,
            "test_acc": extra_test_acc,
        }
        print(
            f"[extrapolation] img_size={eval_img_size}  "
            f"val_acc={extra_val_acc:.4f}  test_acc={extra_test_acc:.4f}"
        )

    if wandb_run is not None:
        wandb_run.summary["best_val_acc"] = best_val_acc
        wandb_run.summary["test_acc"] = test_acc
        wandb_run.finish()

    torch.save(history, args.output_dir / "history.pt")
    return history


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clip_eurosat"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument("--wandb-project", default="cs148b-hw3-clip")
    p.add_argument("--wandb-run-name", default=None)
    # §6 rope-vs-learned: lets the notebook switch PE without editing the YAML.
    p.add_argument(
        "--pos-encoding",
        choices=["learned", "rope", "rope2d"],
        default=None,
        help="Override cfg.vit.pos_encoding. 'learned', 'rope', or 'rope2d'.",
    )
    p.add_argument(
        "--eval-img-size",
        type=int,
        default=None,
        help="If set, run an extra zero-shot eval at this image size after training "
             "(length-extrapolation test). Learned-PE pos_embed is bilinearly "
             "interpolated and RoPE's cache is extended as needed.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg, args)


if __name__ == "__main__":
    main()

"""§5 — VLM training on CLEVR.

For the injection-comparison problem: freeze ViT + decoder, train only the
projector for 500 steps with batch_size 32, lr 1e-4, AdamW. Reports for each
injection strategy: val exact-match accuracy on up to 500 CLEVR-val examples,
number of visual tokens injected per example, peak GPU memory, and wall-clock
time per step.

Usage (one method):
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt \\
        --injection all_patches --mask-mode causal --freeze-config A \\
        --num-steps 500 --batch-size 32 --grad-accum 1 --lr 1e-4

Usage (summary table across all three injection runs):
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --pretrained-vit runs/clip_eurosat/best.pt --summarize
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.optim import AdamW  # noqa: E402
from torch.optim.lr_scheduler import LambdaLR  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from basics.vit import ViT  # noqa: E402
from vlm.data import CLEVRMiniDataset  # noqa: E402
from vlm.eval import batch_clevr_accuracy  # noqa: E402
from vlm.model import IGNORE_INDEX, InjectionMode, VisionLanguageModel  # noqa: E402
from vlm.projector import VisionLanguageProjector  # noqa: E402

IMAGE_TOKEN = "<image>"


def cosine_warmup_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _format_prompt(question: str, is_interleaved: bool) -> str:
    if is_interleaved:
        return f"{IMAGE_TOKEN} Question: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_collate(tokenizer, injection: str, max_len: int):
    """Returns a CLEVR collate_fn that produces a training batch with
    answer-only labels (-100 on prompt + padding positions)."""
    is_interleaved = injection == "interleaved"
    pad_id = tokenizer.pad_token_id
    eos = tokenizer.eos_token

    def collate(batch):
        images = torch.stack([b["image"] for b in batch])
        ids_list, lbl_list = [], []
        for b in batch:
            prompt = _format_prompt(b["question"], is_interleaved)
            answer = str(b["answer"])
            # `add_special_tokens=True` injects BOS once on the prompt; the
            # answer continuation does NOT get its own BOS.
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            answer_ids = tokenizer.encode(" " + answer + eos, add_special_tokens=False)
            full = (prompt_ids + answer_ids)[:max_len]
            # labels: -100 on the prompt so only answer tokens score loss.
            labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_len]
            ids_list.append(torch.tensor(full, dtype=torch.long))
            lbl_list.append(torch.tensor(labels, dtype=torch.long))
        L = max(t.shape[0] for t in ids_list)
        input_ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        labels = torch.full((len(batch), L), IGNORE_INDEX, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(ids_list, lbl_list)):
            input_ids[i, : ids.shape[0]] = ids
            attn[i, : ids.shape[0]] = 1
            labels[i, : lab.shape[0]] = lab
        return {
            "image": images,
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "question": [b["question"] for b in batch],
            "answer": [str(b["answer"]) for b in batch],
            "q_type": [b.get("q_type", "other") for b in batch],
        }

    return collate


@torch.no_grad()
def evaluate_clevr(
    model: VisionLanguageModel,
    val_dl: DataLoader,
    injection: InjectionMode,
    max_examples: int,
    max_new_tokens: int,
    device: torch.device,
) -> dict:
    """Run generation and grade with batch_clevr_accuracy on up to max_examples."""
    model.eval()
    is_interleaved = injection == "interleaved"
    tok = model.tokenizer
    preds: list[str] = []
    golds: list[str] = []
    q_types: list[str] = []
    seen = 0
    for batch in val_dl:
        if seen >= max_examples:
            break
        images = batch["image"].to(device)
        prompts = [_format_prompt(q, is_interleaved) for q in batch["question"]]
        gens = model.generate(
            images,
            prompts,
            injection=injection,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        for g in gens:
            # Take the first whitespace-separated token on the first line — that's
            # the conventional CLEVR answer span; `clevr_exact_match` normalizes
            # punctuation / case / numeric words.
            line = g.strip().split("\n", 1)[0].strip()
            preds.append(line.split()[0] if line else "")
        golds.extend(batch["answer"])
        q_types.extend(batch["q_type"])
        seen += len(batch["answer"])
    n = min(seen, max_examples)
    return batch_clevr_accuracy(preds[:n], golds[:n], q_types[:n])


def apply_lora_to_decoder(decoder, rank: int, alpha: float):
    """Wrap every q_proj / v_proj nn.Linear inside `decoder`'s attention
    modules with `basics.lora.LoRALinear`.

    The §4 helper `apply_lora_to_attention` only targets `basics.model.Head`
    (our own ViT attention class); SmolLM2 uses HF's LLaMA-style attention,
    which exposes `q_proj` and `v_proj` as direct `nn.Linear` attributes on
    each attention layer. We walk modules and replace those two projections.

    A and B parameters are migrated to the base layer's device and dtype after
    wrapping, so the bf16-loaded decoder doesn't hit a mixed-dtype matmul.
    """
    import torch.nn as nn

    from basics.lora import LoRALinear

    for module in decoder.modules():
        if not (
            hasattr(module, "q_proj")
            and isinstance(module.q_proj, nn.Linear)
            and not isinstance(module.q_proj, LoRALinear)
            and hasattr(module, "v_proj")
            and isinstance(module.v_proj, nn.Linear)
            and not isinstance(module.v_proj, LoRALinear)
        ):
            continue
        for proj_name in ("q_proj", "v_proj"):
            base = getattr(module, proj_name)
            wrapped = LoRALinear(base, rank=rank, alpha=alpha)
            with torch.no_grad():
                wrapped.A.data = wrapped.A.data.to(
                    device=base.weight.device, dtype=base.weight.dtype
                )
                wrapped.B.data = wrapped.B.data.to(
                    device=base.weight.device, dtype=base.weight.dtype
                )
            setattr(module, proj_name, wrapped)
    return decoder


def apply_freeze_config(
    vit, projector, decoder, freeze_config: str, lora_rank: int = 8, lora_alpha: float = 16.0
) -> None:
    """A=projector only; B=+decoder LoRA; C=+full decoder; D=everything."""
    for p in vit.parameters():
        p.requires_grad = False
    for p in projector.parameters():
        p.requires_grad = True
    if freeze_config == "A":
        for p in decoder.parameters():
            p.requires_grad = False
    elif freeze_config == "B":
        for p in decoder.parameters():
            p.requires_grad = False
        # LoRALinear flips requires_grad off on the wrapped base, and the new
        # A / B parameters are trainable by default — so the only trainable
        # decoder tensors are the LoRA factors.
        apply_lora_to_decoder(decoder, rank=lora_rank, alpha=lora_alpha)
    elif freeze_config == "C":
        for p in decoder.parameters():
            p.requires_grad = True
    elif freeze_config == "D":
        for p in vit.parameters():
            p.requires_grad = True
        for p in decoder.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown freeze_config: {freeze_config!r}")


def train(args: argparse.Namespace, cfg: dict) -> dict:
    device = torch.device(args.device)

    # 1) CLEVR datasets (we use our own collate so build the datasets directly).
    train_ds = CLEVRMiniDataset(split="train", root=str(args.clevr_root), img_size=64)
    val_ds = CLEVRMiniDataset(split="val", root=str(args.clevr_root), img_size=64)

    # 2) CLIP-pretrained ViT — pull the ViT config from the checkpoint.
    vit_ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_cfg = vit_ckpt["config"]["vit"]
    vit = ViT(**vit_cfg)
    vit.load_state_dict(vit_ckpt["vit"])

    # 3) SmolLM2 decoder + tokenizer (bf16 + chosen attention impl).
    model_name = cfg["decoder"]["model_name"]
    attn_impl = args.attn_impl or cfg["decoder"].get("attn_implementation", "sdpa")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation=attn_impl
    )

    # 4) For interleaved injection, register <image> as a special token and
    #    grow the decoder's embedding table by one row.
    image_token_id = None
    if args.injection == "interleaved":
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": [IMAGE_TOKEN]}
        )
        if added > 0:
            decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 5) Projector + VLM.
    d_image = vit_cfg["d_model"]
    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=d_image, d_decoder=d_decoder, expansion=cfg["projector"]["expansion"]
    )
    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    ).to(device)

    # 6) Freeze configuration (default A: only projector trains).
    apply_freeze_config(model.vit, model.projector, model.decoder, args.freeze_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[{args.injection}/{args.mask_mode}/{args.freeze_config}] "
        f"trainable={trainable_params:,}  total={total_params:,}"
    )

    # 7) Loaders with answer-only collate.
    collate = build_collate(tokenizer, args.injection, args.max_len)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        drop_last=True,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate,
        pin_memory=True,
    )

    # 8) Optimizer + cosine schedule.
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    scheduler = LambdaLR(
        optimizer, cosine_warmup_lambda(cfg["optim"]["warmup_steps"], args.num_steps)
    )

    # 9) Visual-token count per example (used for the deliverable table).
    if args.injection == "cls":
        n_visual = 1
    else:
        n_visual = (vit_cfg["img_size"] // vit_cfg["patch_size"]) ** 2 + 1

    # 10) Train loop.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history = {"step": [], "train_loss": [], "val_acc": []}
    grad_accum = args.grad_accum
    train_iter = iter(train_dl)
    pbar = tqdm(total=args.num_steps, desc=f"[{args.injection}]")
    t_start = time.time()
    step = 0
    while step < args.num_steps:
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                batch = next(train_iter)
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attn,
                labels=labels,
                injection=args.injection,
                mask_mode=args.mask_mode,
            )
            loss = out["loss"] / grad_accum
            loss.backward()
            running += loss.item() * grad_accum
        optimizer.step()
        scheduler.step()
        step += 1
        avg_loss = running / grad_accum
        pbar.update(1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        if step % cfg["train"]["log_every"] == 0:
            history["step"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_acc"].append(None)

        eval_every = cfg["train"]["eval_every_steps"]
        if step % eval_every == 0 or step == args.num_steps:
            val_metrics = evaluate_clevr(
                model,
                val_dl,
                args.injection,
                cfg["train"]["eval_max_examples"],
                cfg["generation"]["max_new_tokens"],
                device,
            )
            history["step"].append(step)
            history["train_loss"].append(avg_loss)
            history["val_acc"].append(val_metrics["overall"])
            print(
                f"\nstep {step}: val_overall={val_metrics['overall']:.4f}  loss={avg_loss:.4f}"
            )
    pbar.close()

    train_time_s = time.time() - t_start
    sec_per_step = train_time_s / max(args.num_steps, 1)
    peak_mem_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )

    # Final eval pass (re-runs from the top of val so it's deterministic).
    final_metrics = evaluate_clevr(
        model,
        val_dl,
        args.injection,
        cfg["train"]["eval_max_examples"],
        cfg["generation"]["max_new_tokens"],
        device,
    )

    metrics = {
        "injection": args.injection,
        "mask_mode": args.mask_mode,
        "freeze_config": args.freeze_config,
        "lr": args.lr,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "n_visual": int(n_visual),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "final_val_acc": final_metrics["overall"],
        "final_val_metrics": final_metrics,
        "peak_mem_bytes": int(peak_mem_bytes),
        "peak_mem_mb": peak_mem_bytes / (1024 ** 2),
        "train_time_s": train_time_s,
        "sec_per_step": sec_per_step,
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"saved {args.output_dir / 'metrics.json'}")

    # Save every trainable tensor — works for any freeze config:
    #   A: only the projector state ends up in the dict.
    #   B: projector + LoRA A/B on q_proj/v_proj across every decoder layer.
    #   C: projector + full decoder weights.
    #   D: projector + ViT + full decoder.
    # The key under which a parameter is saved matches `model.named_parameters()`,
    # so reloading is a single named-parameter copy.
    trainable_state = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(
        {
            "trainable_state": trainable_state,
            "config": {
                "vit": vit_cfg,
                "decoder": cfg["decoder"],
                "projector": cfg["projector"],
                "image_token_id": image_token_id,
                "injection": args.injection,
                "mask_mode": args.mask_mode,
                "freeze_config": args.freeze_config,
                "lora_rank": 8,
                "lora_alpha": 16.0,
            },
            "metrics": {k: v for k, v in metrics.items() if k != "history"},
        },
        args.output_dir / "projector.pt",
    )

    return metrics


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------


def _default_run_dir(
    runs_root: Path, injection: str, mask_mode: str, freeze_config: str
) -> Path:
    return runs_root / f"vlm_{injection}_{mask_mode}_{freeze_config}"


def print_table(metrics_by_injection: dict[str, dict]) -> None:
    cols = [
        ("injection",   "injection",     14),
        ("n_visual",    "n_visual",       9),
        ("val_acc",     "final_val_acc",  9),
        ("peak_mem_MB", "peak_mem_mb",   12),
        ("sec_per_step","sec_per_step",  12),
    ]
    header = " | ".join(f"{name:>{w}}" for name, _, w in cols)
    print()
    print(header)
    print("-" * len(header))
    for inj in ("cls", "all_patches", "interleaved"):
        m = metrics_by_injection.get(inj)
        if m is None:
            continue
        row = []
        for name, key, w in cols:
            v = m[key]
            if name == "val_acc":
                s = f"{v:.4f}"
            elif name == "peak_mem_MB":
                s = f"{v:.1f}"
            elif name == "sec_per_step":
                s = f"{v:.3f}"
            else:
                s = str(v)
            row.append(f"{s:>{w}}")
        print(" | ".join(row))
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3.")
    p.add_argument("--clevr-root", type=Path, default=Path("data/clevr_mini"))
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode", choices=["causal", "image_bidir"], default="causal"
    )
    p.add_argument(
        "--freeze-config", choices=["A", "B", "C", "D"], default="A",
        help="A=projector only; B=+decoder LoRA; C=+full decoder; D=all three.",
    )
    p.add_argument("--num-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps. Default 1.")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument(
        "--attn-impl",
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Override decoder attn_implementation. Use sdpa when mask_mode=image_bidir.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--summarize", action="store_true",
                   help="Skip training; read metrics.json from each injection run and print table.")
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # YAML-backed defaults (CLI takes precedence).
    if args.num_steps is None:
        args.num_steps = cfg["train"]["num_steps"]
    if args.batch_size is None:
        args.batch_size = cfg["train"]["batch_size"]
    if args.lr is None:
        args.lr = cfg["optim"]["lr"]

    if args.summarize:
        metrics_by: dict[str, dict] = {}
        for inj in ("cls", "all_patches", "interleaved"):
            run_dir = _default_run_dir(
                args.runs_root, inj, args.mask_mode, args.freeze_config
            )
            mp = run_dir / "metrics.json"
            if mp.exists():
                metrics_by[inj] = json.loads(mp.read_text())
            else:
                print(f"[warn] no metrics at {mp}")
        print_table(metrics_by)
        return

    if args.output_dir is None:
        args.output_dir = _default_run_dir(
            args.runs_root, args.injection, args.mask_mode, args.freeze_config
        )

    metrics = train(args, cfg)
    print_table({args.injection: metrics})


if __name__ == "__main__":
    main()

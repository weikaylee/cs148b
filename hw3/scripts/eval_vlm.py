"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on held-out CLEVR examples, computes overall + per-q_type
exact-match accuracy on a larger pool, and dumps a balanced sample of correct /
incorrect cases (with the source image copied alongside) for the writeup's
qualitative discussion.

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_B/projector.pt \\
        --pretrained-vit runs/clip_eurosat/best.pt \\
        --num-examples 10 --max-eval 500 --save-images
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from basics.vit import ViT  # noqa: E402
from scripts.train_vlm import apply_lora_to_decoder  # noqa: E402
from vlm.data import CLEVRMiniDataset  # noqa: E402
from vlm.eval import batch_clevr_accuracy, clevr_exact_match  # noqa: E402
from vlm.model import VisionLanguageModel  # noqa: E402
from vlm.projector import VisionLanguageProjector  # noqa: E402


IMAGE_TOKEN = "<image>"


def _format_prompt(question: str, is_interleaved: bool) -> str:
    if is_interleaved:
        return f"{IMAGE_TOKEN} Question: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def build_model_for_eval(
    ckpt: dict, pretrained_vit_path: Path, attn_impl: str = "sdpa"
):
    """Reconstruct the VLM the checkpoint was trained from and load the saved
    trainable state on top.

    Supports both the current format (`trainable_state`) and the legacy format
    (`projector` only, which assumes freeze_config=A).
    """
    cfg = ckpt["config"]

    # 1) ViT — backbone weights live in the §3 checkpoint (frozen for A/B/C and
    #    overwritten from `trainable_state` for D).
    vit = ViT(**cfg["vit"])
    vit_ckpt = torch.load(pretrained_vit_path, map_location="cpu")
    vit.load_state_dict(vit_ckpt["vit"])

    # 2) Decoder + tokenizer.
    decoder_cfg = cfg["decoder"]
    tokenizer = AutoTokenizer.from_pretrained(decoder_cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(
        decoder_cfg["model_name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    # 3) Interleaved injection: re-register <image> and resize embeddings BEFORE
    #    loading trainable state, so the row layout matches what was saved.
    image_token_id = cfg.get("image_token_id")
    if cfg["injection"] == "interleaved" and image_token_id is None:
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        decoder.resize_token_embeddings(len(tokenizer))
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

    # 4) Projector (always trainable in every config).
    projector = VisionLanguageProjector(
        d_image=cfg["vit"]["d_model"],
        d_decoder=decoder.config.hidden_size,
        expansion=cfg["projector"]["expansion"],
    )

    # 5) For freeze_config=B, re-wrap q_proj/v_proj BEFORE building the VLM so
    #    the saved A/B parameter names match named_parameters().
    freeze_cfg = cfg.get("freeze_config", "A")
    if freeze_cfg == "B":
        apply_lora_to_decoder(
            decoder,
            rank=cfg.get("lora_rank", 8),
            alpha=cfg.get("lora_alpha", 16.0),
        )

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    )

    # 6) Load saved trainable state.
    if "trainable_state" in ckpt:
        state = ckpt["trainable_state"]
        own = dict(model.named_parameters())
        missing, loaded = [], 0
        for name, tensor in state.items():
            if name in own:
                with torch.no_grad():
                    own[name].data.copy_(tensor.to(own[name].dtype))
                loaded += 1
            else:
                missing.append(name)
        if missing:
            print(
                f"[warn] {len(missing)} saved params not found in model: "
                f"{missing[:3]}..."
            )
        print(f"loaded {loaded} trainable tensors from checkpoint")
    elif "projector" in ckpt:
        # Legacy projector-only checkpoint -> only freeze_config A makes sense.
        model.projector.load_state_dict(ckpt["projector"])
        print("loaded legacy projector-only checkpoint")
    else:
        raise KeyError(
            "checkpoint must contain 'trainable_state' or legacy 'projector' key"
        )

    return model, tokenizer


@torch.no_grad()
def generate_all(
    model: VisionLanguageModel,
    val_ds: CLEVRMiniDataset,
    injection: str,
    device: torch.device,
    max_eval: int,
    max_new_tokens: int,
) -> list[dict]:
    """Run greedy generation on the first `max_eval` val examples; return a
    list of result dicts (one per example)."""
    tok = model.tokenizer
    is_interleaved = injection == "interleaved"
    results: list[dict] = []
    n = min(max_eval, len(val_ds))
    pbar = tqdm(range(n), desc="generate")
    for idx in pbar:
        ex_raw = val_ds.examples[idx]
        sample = val_ds[idx]
        image = sample["image"].unsqueeze(0).to(device)
        prompt = _format_prompt(ex_raw["question"], is_interleaved)
        gen = model.generate(
            image,
            [prompt],
            injection=injection,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )[0]
        line = gen.strip().split("\n", 1)[0].strip()
        pred = line.split()[0] if line else ""
        gold = str(ex_raw["answer"])
        results.append(
            {
                "idx": idx,
                "image_file": ex_raw["image_file"],
                "question": ex_raw["question"],
                "gold": gold,
                "prediction": pred,
                "raw_generation": gen,
                "q_type": ex_raw.get("q_type", "other"),
                "correct": clevr_exact_match(pred, gold),
            }
        )
    pbar.close()
    return results


def sample_balanced(
    results: list[dict], n: int, seed: int = 0
) -> list[dict]:
    """Pick a mix of correct and incorrect predictions (~half each, topping up
    from whichever pool is larger if one is short)."""
    rng = random.Random(seed)
    correct = [r for r in results if r["correct"]]
    wrong = [r for r in results if not r["correct"]]
    n_c = min(n // 2, len(correct))
    n_w = min(n - n_c, len(wrong))
    if n_c + n_w < n:
        deficit = n - n_c - n_w
        more_c = min(deficit, len(correct) - n_c)
        n_c += more_c
        deficit -= more_c
        n_w += min(deficit, len(wrong) - n_w)
    sampled = rng.sample(correct, n_c) + rng.sample(wrong, n_w)
    rng.shuffle(sampled)
    return sampled


def write_outputs(
    sampled: list[dict],
    out_dir: Path,
    clevr_root: Path,
    save_images: bool,
) -> None:
    """Write examples.jsonl, examples.md (markdown table), and optionally copy
    the source images into out_dir/images/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        (out_dir / "images").mkdir(parents=True, exist_ok=True)

    jsonl = out_dir / "examples.jsonl"
    md_path = out_dir / "examples.md"

    with open(jsonl, "w") as fj, open(md_path, "w") as fm:
        fm.write(
            "| status | q_type | image | question | gold | prediction |\n"
            "|--------|--------|-------|----------|------|------------|\n"
        )
        for e in sampled:
            row = dict(e)
            if save_images:
                src = clevr_root / "images" / e["image_file"]
                dst = out_dir / "images" / e["image_file"]
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                row["image_path_local"] = f"images/{e['image_file']}"
            fj.write(json.dumps(row) + "\n")
            status = "✓" if e["correct"] else "✗"
            img_md = (
                f"![]({row['image_path_local']})"
                if save_images
                else e["image_file"]
            )
            fm.write(
                f"| {status} | {e['q_type']} | {img_md} | "
                f"{e['question']} | {e['gold']} | {e['prediction']} |\n"
            )
    print(f"wrote {jsonl}")
    print(f"wrote {md_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to projector.pt (or any trainable.pt) from training.")
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to the §3 CLIP-pretrained ViT checkpoint.")
    p.add_argument("--clevr-root", type=Path, default=Path("data/clevr_mini"))
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of (mixed correct/incorrect) examples to dump.")
    p.add_argument("--max-eval", type=int, default=500,
                   help="How many val examples to score for accuracy + sampling pool.")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--attn-impl", default="sdpa",
                   choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--save-images", action="store_true",
                   help="Copy source images into <output-dir>/images/.")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # 1) Load checkpoint + rebuild VLM.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(
        f"checkpoint: injection={cfg['injection']}  "
        f"mask_mode={cfg.get('mask_mode', '?')}  "
        f"freeze_config={cfg.get('freeze_config', '?')}"
    )
    model, _tokenizer = build_model_for_eval(ckpt, args.pretrained_vit, args.attn_impl)
    model.to(device).eval()

    # 2) Generate on max_eval val examples.
    val_ds = CLEVRMiniDataset(
        split=args.split, root=str(args.clevr_root), img_size=cfg["vit"]["img_size"]
    )
    results = generate_all(
        model,
        val_ds,
        injection=cfg["injection"],
        device=device,
        max_eval=args.max_eval,
        max_new_tokens=args.max_new_tokens,
    )

    # 3) Accuracy summary.
    overall = batch_clevr_accuracy(
        [r["prediction"] for r in results],
        [r["gold"] for r in results],
        [r["q_type"] for r in results],
    )
    n_correct = sum(1 for r in results if r["correct"])
    print(
        f"\nscored {len(results)} examples: "
        f"{n_correct}/{len(results)} correct "
        f"({n_correct / max(len(results), 1):.2%})"
    )
    print("per-q_type accuracy:")
    for k, v in overall.items():
        if k != "overall":
            print(f"  {k:>16}: {v:.4f}")

    # 4) Pick a balanced 10-example dump and write outputs.
    sampled = sample_balanced(results, args.num_examples, seed=args.seed)
    write_outputs(sampled, args.output_dir, args.clevr_root, args.save_images)

    # Also save the accuracy summary for the writeup.
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(args.checkpoint),
                "split": args.split,
                "max_eval": args.max_eval,
                "num_correct": n_correct,
                "overall_metrics": overall,
                "config": cfg,
            },
            f,
            indent=2,
        )

    # 5) Print a concise preview.
    print("\n=== qualitative examples ===")
    for e in sampled:
        status = "✓" if e["correct"] else "✗"
        print(f"{status} [{e['q_type']:>10}] {e['image_file']}")
        print(f"    Q: {e['question']}")
        print(
            f"    gold: {e['gold']!r:>10}   pred: {e['prediction']!r:>10}   "
            f"raw: {e['raw_generation']!r}"
        )


if __name__ == "__main__":
    main()

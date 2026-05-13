# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

CS/EE 148B HW3 starter code for building a Vision-Language Model from scratch in five sections:
§2 ViT → §3 CLIP pretraining (EuroSAT) → §4 LoRA fine-tuning (RESISC45) → §5 VLM on CLEVR → §6 RoPE ablations.

The writeup (`hw3.pdf`) is the source of truth for what each problem requires.

## Commands

Dependency management is via `uv`. All commands assume the `hw3/` directory as cwd.

```bash
# Initial setup
uv sync --extra test           # core + pytest
uv sync --extra flash          # adds flash-attn (needed for §5 VLM training; Linux/CUDA only)

# Tests — staff tests live in tests/test_*.py and import tests/adapters.py
uv run pytest                                # all
uv run pytest -k test_patch_embeddings       # single test by name substring
uv run pytest tests/test_vit.py              # single file

# Experiment scripts (each takes a YAML config + CLI overrides)
uv run python scripts/pretrain_clip.py    --config configs/clip_eurosat.yaml
uv run python scripts/finetune_resisc.py  --config configs/lora_resisc.yaml --method lora --rank 8 --pretrained <vit.pt>
uv run python scripts/train_vlm.py        --config configs/vlm_clevr.yaml   --injection all_patches --mask-mode image_bidir --pretrained-vit <vit.pt>
uv run python scripts/eval_vlm.py         --checkpoint runs/<...>/best.pt

# CLEVR dataset (must run before §5)
uv run python scripts/download_clevr.py
```

## Architecture

### The PROVIDED vs TODO split is load-bearing — do not modify provided files

These files contain working code and define the interfaces that TODOs must conform to. **Do not edit them**, even if a fix seems obvious — the staff tests assume their behavior:

- `basics/model.py` — `Head`, `MultiHeadAttention`, `MLP`, `Block`. The `is_decoder` flag toggles causal vs bidirectional attention; the same `Block` is reused for the ViT encoder (`is_decoder=False`) and the VLM language decoder path (`is_decoder=True`).
- `basics/text_encoder.py` — `FrozenTextEncoder` (sentence-transformers wrapper). Always in eval mode; `.embedding_dim` exposes output dim.
- `vlm/masking.py` — `build_causal_mask`, `build_image_bidir_mask`. Returns (1,1,T,T) additive masks where `torch.finfo(dtype).min` blocks attention. The "image_bidir" layout is `[v_1..v_N, t_1..t_M]`: visual tokens attend bidirectionally among themselves but cannot see text; text attends causally to all visuals + past text.
- `vlm/data.py` — EuroSAT / RESISC45 / CLEVR dataset loaders.
- `vlm/eval.py` — zero-shot accuracy and CLEVR exact-match scoring.

TODO files (have `raise NotImplementedError` placeholders with hints in docstrings):

- §2: `basics/vit.py` — `PatchEmbeddings`, `ViT`. The writeup hints add a `return_all_tokens=True` flag later for §5.
- §3: `vlm/clip.py` — `ProjectionHeads`, `clip_loss`. Logit-scale clamping is done in the *training loop*, not inside `clip_loss`.
- §4: `basics/lora.py` — `LoRALinear`, `apply_lora_to_attention`. Walks `model.named_modules()` and swaps `q_proj`/`v_proj` inside every `basics.model.Head`.
- §5: `vlm/projector.py` (MLP projector) and `vlm/model.py` (`VisionLanguageModel`).
- §6: `basics/rope.py` — `RoPE1D`, `RoPE2D`. Applied to Q and K (not V) before the attention dot product.

### Test infrastructure: the `adapters.py` indirection

Tests don't import student modules directly — they go through `tests/adapters.py`, which has `run_*` shim functions. **Only edit the bodies of these `run_*` functions; do not change their signatures.** The current bodies already import and call the expected classes; usually you only need to keep them in sync if you rename or refactor.

### Section interdependencies (checkpoint flow)

- §3 produces a CLIP-pretrained ViT checkpoint → consumed by §4 (`--pretrained`) and §5 (`--pretrained-vit`).
- §4 produces RESISC accuracy / param-count / peak-memory numbers for comparing LoRA vs full FT vs linear probe.
- §5 freeze-configs A/B/C/D: A=projector only, B=+decoder LoRA (reuses §4's `apply_lora_to_attention`), C=+full decoder, D=all three trainable.

### VLM injection modes (§5)

`VisionLanguageModel.forward` switches on `injection`:
- `"cls"` — prepend a single visual token (ViT CLS) → 1 visual position.
- `"all_patches"` — prepend CLS + all patches → N+1 visual positions.
- `"interleaved"` — locate a special `<image>` token in `input_ids` and replace it with the patch-embedding sequence.

`mask_mode` is orthogonal: `"causal"` uses HF's default; `"image_bidir"` builds a custom 4D mask via `vlm/masking.py` and passes it to the decoder. Visual-token positions in `labels` must be set to `-100` before being passed to HF so the loss ignores them.

### Decoder choice (§5)

SmolLM2-360M-Instruct loaded in bf16 with FlashAttention-2. `d_decoder = 960`. The projector MLP maps `d_image → 4*d_image → 960`.

## Conventions

- Image inputs are 64×64, ImageNet-normalized (see `default_image_transform` in `vlm/data.py`).
- Configs are YAML in `configs/`; scripts both consume the config and accept CLI overrides for ablation knobs (e.g., `--injection`, `--rank`).
- Run outputs go to `runs/<experiment>/` (gitignored). Checkpoints are `*.pt`.
- Lint: ruff (line-length 100, py310 target); E501 and E741 ignored.
- Python 3.10–3.12; torch 2.4+.

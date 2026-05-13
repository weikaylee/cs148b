"""§2 (vit_patch_size) — effect of patch size on token count and forward time.

Times a ViT (d_model=384, num_heads=6, num_blocks=6) forward pass on a batch of
16 images at 224x224 for patch sizes P in {8, 16, 32}. Reports N = (224/P)^2
and mean ± std wall-clock time over 20 steps after 5 warmup steps, with proper
device synchronization.

Usage:
    uv run python scripts/patch_size_benchmark.py            # auto device
    uv run python scripts/patch_size_benchmark.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `basics` importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from basics.vit import ViT  # noqa: E402


IMG_SIZE = 224
BATCH = 16
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
PATCH_SIZES = (8, 16, 32)
WARMUP_STEPS = 5
TIMED_STEPS = 20


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Block until queued kernels on `device` have completed."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def benchmark(patch_size: int, device: torch.device) -> tuple[int, float, float]:
    model = (
        ViT(
            img_size=IMG_SIZE,
            patch_size=patch_size,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            num_blocks=NUM_BLOCKS,
            dropout=0.0,
        )
        .to(device)
        .eval()
    )
    x = torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE, device=device)
    n_patches = (IMG_SIZE // patch_size) ** 2

    for _ in range(WARMUP_STEPS):
        _ = model(x)
    sync(device)

    times_s = []
    for _ in range(TIMED_STEPS):
        sync(device)
        t0 = time.perf_counter()
        _ = model(x)
        sync(device)
        times_s.append(time.perf_counter() - t0)

    t = torch.tensor(times_s)
    return n_patches, t.mean().item(), t.std().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(
        f"config: img_size={IMG_SIZE}, batch={BATCH}, d_model={D_MODEL}, "
        f"num_heads={NUM_HEADS}, num_blocks={NUM_BLOCKS}"
    )
    print(f"timing: {WARMUP_STEPS} warmup + {TIMED_STEPS} measured steps")
    print()

    # Part (1): print N for each P so the table doubles as an answer.
    print(f"{'P':>4} | {'N':>5} | {'mean (ms)':>11} | {'std (ms)':>10}")
    print("-" * 42)
    for patch_size in PATCH_SIZES:
        n, mean_s, std_s = benchmark(patch_size, device)
        print(f"{patch_size:>4} | {n:>5} | {mean_s * 1000:>11.3f} | {std_s * 1000:>10.3f}")


if __name__ == "__main__":
    main()

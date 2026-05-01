from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_eval_history(history_path: Path) -> tuple[list[int], list[float], list[float]]:
    with history_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    eval_rows = [row for row in rows if row.get("phase") == "eval"]
    if not eval_rows:
        raise ValueError(f"No eval rows found in {history_path}")

    steps = [int(row["step"]) for row in eval_rows]
    train_loss = [float(row["train_loss"]) for row in eval_rows]
    val_loss = [float(row["val_loss"]) for row in eval_rows]
    return steps, train_loss, val_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot Sinusoidal PE vs NoPE learning curves.")
    p.add_argument(
        "--sinusoidal-history",
        type=Path,
        default=Path("checkpoints/final_single_tuned_online/training_history.json"),
        help="Path to training_history.json for the sinusoidal PE run.",
    )
    p.add_argument(
        "--nope-history",
        type=Path,
        default=Path("checkpoints/final_nope_tuned_online/training_history.json"),
        help="Path to training_history.json for the no-position-encoding run.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/pe_vs_nope_learning_curve.png"),
        help="Output image path.",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default="sinusoidal-vs-nope-curve")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sin_steps, sin_train, sin_val = _load_eval_history(args.sinusoidal_history)
    nope_steps, nope_train, nope_val = _load_eval_history(args.nope_history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(sin_steps, sin_val, label="Sinusoidal PE (val)", linewidth=2)
    plt.plot(nope_steps, nope_val, label="NoPE (val)", linewidth=2)
    plt.plot(sin_steps, sin_train, "--", label="Sinusoidal PE (train)", alpha=0.7)
    plt.plot(nope_steps, nope_train, "--", label="NoPE (train)", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Learning Curve: Sinusoidal PE vs NoPE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close()

    print(f"Saved curve to {args.output}")

    if args.wandb_project is not None:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "sinusoidal_history": str(args.sinusoidal_history),
                    "nope_history": str(args.nope_history),
                    "output": str(args.output),
                },
            )
            if run is not None:
                wandb.log({"pe_vs_nope_curve": wandb.Image(str(args.output))})
                run.finish()
        except Exception as exc:
            print(f"wandb logging skipped: {exc}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
import statistics
import sys
import timeit
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Literal, TypedDict

import torch
try:
    import basics.basics.model as basics_model_module
    from basics.basics.model import BasicsTransformerLM
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basics"))
    import basics.model as basics_model_module
    from basics.model import BasicsTransformerLM


Mode = Literal["forward", "backward", "optimizer", "train-step"]


class BenchmarkMetrics(TypedDict):
    avg_ms: float
    std_ms: float
    timings_ms: list[float]
    status: str
    peak_mem_bytes: int


class AttentionBenchmarkMetrics(TypedDict):
    d_model: int
    sequence_length: int
    forward_avg_ms: float
    backward_avg_ms: float
    mem_before_backward_bytes: int
    saved_for_backward_bytes: int
    status: str


MODEL_SPECS = {
    "small": {"d_model": 512, "d_ff": 2048, "num_layers": 8, "num_heads": 8},
    "medium": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "large": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
}
CONTEXT_LENGTHS = (32, 64, 128, 256)
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
ROPE_THETA = 10_000.0
ATTENTION_BATCH_SIZE = 8
ATTENTION_HEAD_COUNTS = (1,)
ATTENTION_HEAD_DIMS = (16, 32, 64, 128)
ATTENTION_SEQUENCE_LENGTHS = (64, 128, 256, 512, 1024)
ATTENTION_FORWARD_ITERS = 100
ATTENTION_BACKWARD_ITERS = 100
ATTENTION_WARMUP_ITERS = 10


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def precision_context(device: torch.device, use_bf16: bool):
    if use_bf16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def maybe_compile_callable(fn):
    try:
        return torch.compile(fn)
    except Exception as exc:
        print(f"Warning: torch.compile failed for callable {getattr(fn, '__name__', type(fn).__name__)}: {exc}")
        return fn


def build_model(
    model_size: str,
    context_length: int,
    vocab_size: int,
    rope_theta: float,
    device: torch.device,
    compile_model: bool = False,
) -> BasicsTransformerLM:
    spec = MODEL_SPECS[model_size]
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=spec["d_model"],
        num_layers=spec["num_layers"],
        num_heads=spec["num_heads"],
        d_ff=spec["d_ff"],
        rope_theta=rope_theta,
    ).to(device)
    if compile_model:
        model = maybe_compile_callable(model)
    return model


def make_batch(vocab_size: int, batch_size: int, context_length: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)


def make_attention_inputs(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (batch_size, num_heads, sequence_length, head_dim)
    q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=requires_grad)
    k = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=requires_grad)
    v = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=requires_grad)
    return q, k, v


def build_causal_mask(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.tril(torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, sequence_length, sequence_length)


def estimate_attention_saved_for_backward_bytes(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
    dtype_bytes: int = 4,
) -> int:
    attention_matrix_bytes = batch_size * num_heads * sequence_length * sequence_length * dtype_bytes
    qkv_bytes = 3 * batch_size * num_heads * sequence_length * head_dim * dtype_bytes
    return attention_matrix_bytes + qkv_bytes


def benchmark_attention_case(
    attention_fn,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    sequence_length: int,
    device: torch.device,
    use_nvtx: bool,
) -> dict[str, float | int | str]:
    saved_for_backward_bytes = estimate_attention_saved_for_backward_bytes(
        batch_size=batch_size,
        num_heads=num_heads,
        sequence_length=sequence_length,
        head_dim=head_dim,
        dtype_bytes=4,
    )

    status = "ok"
    forward_avg_ms = float("nan")
    backward_avg_ms = float("nan")
    mem_before_backward_bytes = 0
    mask = build_causal_mask(batch_size, num_heads, sequence_length, device)

    try:
        q, k, v = make_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            sequence_length=sequence_length,
            head_dim=head_dim,
            device=device,
            requires_grad=False,
        )

        for _ in range(ATTENTION_WARMUP_ITERS):
            synchronize_cuda(device)
            with torch.no_grad():
                _ = attention_fn(q, k, v, mask=mask)
            synchronize_cuda(device)

        forward_times_ms: list[float] = []
        for _ in range(ATTENTION_FORWARD_ITERS):
            synchronize_cuda(device)
            t0 = timeit.default_timer()
            with torch.no_grad():
                _ = attention_fn(q, k, v, mask=mask)
            synchronize_cuda(device)
            t1 = timeit.default_timer()
            forward_times_ms.append((t1 - t0) * 1000.0)

        forward_avg_ms = statistics.mean(forward_times_ms) if forward_times_ms else float("nan")

        if device.type == "cuda":
            torch.cuda.empty_cache()
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        q, k, v = make_attention_inputs(
            batch_size=batch_size,
            num_heads=num_heads,
            sequence_length=sequence_length,
            head_dim=head_dim,
            device=device,
            requires_grad=True,
        )

        for _ in range(ATTENTION_WARMUP_ITERS):
            q.grad = None
            k.grad = None
            v.grad = None
            with nvtx_range("attention_forward", use_nvtx):
                output = attention_fn(q, k, v, mask=mask)
                loss = output.sum()
            with nvtx_range("attention_backward", use_nvtx):
                loss.backward()
            synchronize_cuda(device)

        q.grad = None
        k.grad = None
        v.grad = None
        with nvtx_range("attention_forward", use_nvtx):
            output = attention_fn(q, k, v, mask=mask)
            loss = output.sum()
        synchronize_cuda(device)
        if device.type == "cuda":
            mem_before_backward_bytes = int(torch.cuda.memory_allocated())

        backward_times_ms: list[float] = []
        for _ in range(ATTENTION_BACKWARD_ITERS):
            q.grad = None
            k.grad = None
            v.grad = None
            synchronize_cuda(device)
            t0 = timeit.default_timer()
            with nvtx_range("attention_backward", use_nvtx):
                output = attention_fn(q, k, v, mask=mask)
                loss = output.sum()
                loss.backward()
            synchronize_cuda(device)
            t1 = timeit.default_timer()
            backward_times_ms.append((t1 - t0) * 1000.0)

        backward_avg_ms = statistics.mean(backward_times_ms) if backward_times_ms else float("nan")

    except torch.cuda.OutOfMemoryError:
        status = "oom"
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "forward_avg_ms": forward_avg_ms,
        "backward_avg_ms": backward_avg_ms,
        "mem_before_backward_bytes": mem_before_backward_bytes,
        "saved_for_backward_bytes": saved_for_backward_bytes,
        "status": status,
    }


def _forward_pass(model: BasicsTransformerLM, batch: torch.Tensor, use_nvtx: bool) -> torch.Tensor:
    with nvtx_range("forward", use_nvtx):
        return model(batch)


def _backward_pass(loss: torch.Tensor, use_nvtx: bool) -> None:
    with nvtx_range("backward", use_nvtx):
        loss.backward()


def _optimizer_step(optimizer: torch.optim.Optimizer, use_nvtx: bool) -> None:
    with nvtx_range("optimizer_step", use_nvtx):
        optimizer.step()


def _compute_loss(logits: torch.Tensor, use_nvtx: bool) -> torch.Tensor:
    with nvtx_range("loss", use_nvtx):
        return logits.float().mean()


def benchmark_mode(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mode: Mode,
    warmup_steps: int,
    measure_steps: int,
    device: torch.device,
    use_nvtx: bool,
    use_bf16: bool,
    memory_profile: bool = False,
    snapshot_path: str | None = None,
) -> BenchmarkMetrics:
    step_times_ms: list[float] = []

    if mode == "forward":
        model.eval()
    else:
        model.train()

    try:
        if memory_profile and device.type != "cuda":
            print("Warning: --memory-profile requested but CUDA is not available; no memory snapshot will be recorded.")
        # Optionally start recording CUDA memory history before the warmup phase
        if memory_profile and device.type == "cuda":
            mem_mod = getattr(torch.cuda, "memory", None)
            record_fn = getattr(mem_mod, "_record_memory_history", None) if mem_mod is not None else None
            if callable(record_fn):
                try:
                    record_fn(max_entries=1000000)
                    print(f"Started CUDA memory history recording (max_entries=1000000)")
                except TypeError:
                    # Fallback if signature differs
                    try:
                        record_fn(True)
                    except Exception:
                        print("Warning: unable to start CUDA memory history recording (signature mismatch)")
            else:
                print("Warning: torch.cuda.memory._record_memory_history not available on this PyTorch build")

        for _ in range(warmup_steps):
            if mode == "forward":
                with torch.no_grad():
                    with precision_context(device, use_bf16):
                        _ = _forward_pass(model, batch, use_nvtx)
            elif mode == "backward":
                model.zero_grad(set_to_none=True)
                with precision_context(device, use_bf16):
                    logits = _forward_pass(model, batch, use_nvtx)
                    loss = logits.float().mean()
                _backward_pass(loss, use_nvtx)
            elif mode == "optimizer":
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, use_bf16):
                    logits = _forward_pass(model, batch, use_nvtx)
                    loss = _compute_loss(logits, use_nvtx)
                _backward_pass(loss, use_nvtx)
                _optimizer_step(optimizer, use_nvtx)
            else:
                optimizer.zero_grad(set_to_none=True)
                with nvtx_range("train_step", use_nvtx):
                    with precision_context(device, use_bf16):
                        logits = _forward_pass(model, batch, use_nvtx)
                        loss = _compute_loss(logits, use_nvtx)
                    _backward_pass(loss, use_nvtx)
                    _optimizer_step(optimizer, use_nvtx)
            synchronize_cuda(device)

        # Reset peak memory statistics so peak reflects the measurement window only.
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        for _ in range(measure_steps):
            if mode == "forward":
                synchronize_cuda(device)
                t0 = timeit.default_timer()
                with torch.no_grad(), nvtx_range("inference_step", use_nvtx):
                    with precision_context(device, use_bf16):
                        _ = _forward_pass(model, batch, use_nvtx)
                synchronize_cuda(device)
                t1 = timeit.default_timer()
            elif mode == "backward":
                model.zero_grad(set_to_none=True)
                with precision_context(device, use_bf16):
                    logits = _forward_pass(model, batch, use_nvtx)
                    loss = _compute_loss(logits, use_nvtx)
                synchronize_cuda(device)
                t0 = timeit.default_timer()
                _backward_pass(loss, use_nvtx)
                synchronize_cuda(device)
                t1 = timeit.default_timer()
            elif mode == "optimizer":
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, use_bf16):
                    logits = _forward_pass(model, batch, use_nvtx)
                    loss = _compute_loss(logits, use_nvtx)
                _backward_pass(loss, use_nvtx)
                synchronize_cuda(device)
                t0 = timeit.default_timer()
                _optimizer_step(optimizer, use_nvtx)
                synchronize_cuda(device)
                t1 = timeit.default_timer()
            else:
                optimizer.zero_grad(set_to_none=True)
                synchronize_cuda(device)
                t0 = timeit.default_timer()
                with nvtx_range("train_step", use_nvtx):
                    with precision_context(device, use_bf16):
                        logits = _forward_pass(model, batch, use_nvtx)
                        loss = _compute_loss(logits, use_nvtx)
                    _backward_pass(loss, use_nvtx)
                    _optimizer_step(optimizer, use_nvtx)
                synchronize_cuda(device)
                t1 = timeit.default_timer()

            step_times_ms.append((t1 - t0) * 1000.0)

        avg_ms = statistics.mean(step_times_ms) if step_times_ms else float("nan")
        std_ms = statistics.stdev(step_times_ms) if len(step_times_ms) > 1 else 0.0
        # Optionally dump memory snapshot after measurements
        if memory_profile and device.type == "cuda":
            mem_mod = getattr(torch.cuda, "memory", None)
            dump_fn = getattr(mem_mod, "_dump_snapshot", None) if mem_mod is not None else None
            record_fn = getattr(mem_mod, "_record_memory_history", None) if mem_mod is not None else None
            out_path = snapshot_path or "memory_snapshot.pickle"
            if callable(dump_fn):
                try:
                    dump_fn(out_path)
                    print(f"Wrote memory snapshot to {out_path}")
                except Exception as e:
                    print(f"Warning: failed to dump memory snapshot: {e}")
            else:
                print("Warning: torch.cuda.memory._dump_snapshot not available on this PyTorch build")

            # Stop recording history (restore default)
            if callable(record_fn):
                try:
                    record_fn(enabled=None)
                except TypeError:
                    try:
                        record_fn(None)
                    except Exception:
                        pass
        peak_mem = 0
        if device.type == "cuda":
            try:
                peak_mem = int(torch.cuda.max_memory_allocated())
            except Exception:
                peak_mem = 0

        return {
            "avg_ms": avg_ms,
            "std_ms": std_ms,
            "timings_ms": step_times_ms,
            "status": "ok",
            "peak_mem_bytes": peak_mem,
        }
    except torch.cuda.OutOfMemoryError:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "avg_ms": float("nan"),
            "std_ms": float("nan"),
            "timings_ms": [],
            "status": "oom",
            "peak_mem_bytes": 0,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NVTX profiling benchmark for BasicsTransformerLM")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), default=None, help="If omitted, run all model sizes")
    parser.add_argument("--context-length", type=int, default=None, help="If omitted, run 32/64/128/256")
    parser.add_argument("--timing", choices=["forward", "backward", "optimizer", "train-step", "all"], default="all")
    parser.add_argument(
        "--compare-softmax-matmul",
        action="store_true",
        help="During forward pass, compare softmax runtime vs attention matmul runtime and FLOPs.",
    )
    parser.add_argument(
        "--compare-forward-train-step",
        action="store_true",
        help="Run and print forward-only vs full AdamW train-step metrics for side-by-side Nsight comparison.",
    )
    parser.add_argument(
        "--compare-bf16",
        action="store_true",
        help="Compare FP32 vs BF16 runtimes for forward and backward across model sizes.",
    )
    parser.add_argument("--use-bf16", action="store_true", help="Use BF16 autocast (CUDA only) for regular profiling modes.")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rope_theta", type=float, default=ROPE_THETA)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=1)
    parser.add_argument("--disable-nvtx", action="store_true", help="Disable NVTX ranges")
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Record CUDA memory history and dump a snapshot for PyTorch memory viz (CUDA only).",
    )
    parser.add_argument(
        "--benchmark-attention-scaling",
        action="store_true",
        help="Benchmark the standalone attention implementation over the requested d_model and sequence-length grid.",
    )
    parser.add_argument(
        "--compare-compiled-attention",
        action="store_true",
        help="Compare torch.compile() vs eager attention on the same batch/head/sequence grid.",
    )
    parser.add_argument(
        "--compare-compiled-transformer",
        action="store_true",
        help="Compare torch.compile() vs eager Transformer runs for forward/backward/optimizer/train-step.",
    )
    return parser


def compare_softmax_vs_matmul(args: argparse.Namespace) -> dict[str, float]:
    device = get_device()
    model_size = args.model_size or "small"
    context_length = args.context_length or 128
    use_nvtx = not args.disable_nvtx

    spec = MODEL_SPECS[model_size]
    model = build_model(model_size, context_length, args.vocab_size, args.rope_theta, device)
    batch = make_batch(args.vocab_size, args.batch_size, context_length, device)

    original_attention = basics_model_module.scaled_dot_product_attention
    timing = {
        "qk_ms": 0.0,
        "softmax_ms": 0.0,
        "av_ms": 0.0,
        "calls": 0,
        "enabled": False,
    }

    def instrumented_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]

        with nvtx_range("attn_qk_matmul", use_nvtx):
            synchronize_cuda(device)
            t0 = timeit.default_timer()
            attention_scores = basics_model_module.einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
            synchronize_cuda(device)
            t1 = timeit.default_timer()

        with nvtx_range("attn_softmax", use_nvtx):
            synchronize_cuda(device)
            t2 = timeit.default_timer()
            attention_weights = basics_model_module.softmax(attention_scores, dim=-1)
            synchronize_cuda(device)
            t3 = timeit.default_timer()

        with nvtx_range("attn_av_matmul", use_nvtx):
            synchronize_cuda(device)
            t4 = timeit.default_timer()
            output = basics_model_module.einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
            synchronize_cuda(device)
            t5 = timeit.default_timer()

        if timing["enabled"]:
            timing["qk_ms"] += (t1 - t0) * 1000.0
            timing["softmax_ms"] += (t3 - t2) * 1000.0
            timing["av_ms"] += (t5 - t4) * 1000.0
            timing["calls"] += 1

        return output

    try:
        basics_model_module.scaled_dot_product_attention = instrumented_attention

        model.eval()
        for _ in range(args.warmup_steps):
            with torch.no_grad(), nvtx_range("inference_step", use_nvtx):
                _ = model(batch)

        synchronize_cuda(device)
        total_t0 = timeit.default_timer()
        timing["enabled"] = True
        for _ in range(args.measure_steps):
            with torch.no_grad(), nvtx_range("inference_step", use_nvtx):
                _ = model(batch)
        timing["enabled"] = False
        synchronize_cuda(device)
        total_t1 = timeit.default_timer()
    finally:
        basics_model_module.scaled_dot_product_attention = original_attention

    matmul_ms = timing["qk_ms"] + timing["av_ms"]
    softmax_ms = timing["softmax_ms"]
    attn_total_ms = matmul_ms + softmax_ms
    total_forward_ms = (total_t1 - total_t0) * 1000.0

    batch = args.batch_size
    seq = context_length
    heads = spec["num_heads"]
    d_head = spec["d_model"] // spec["num_heads"]
    layers = spec["num_layers"]
    calls_expected = layers * args.measure_steps

    # Approximate FLOPs in one attention call (QK^T + A@V, and softmax reductions/pointwise ops).
    matmul_flops_per_call = 4 * batch * heads * seq * seq * d_head
    softmax_flops_per_call = 4 * batch * heads * seq * seq

    matmul_flops_total = matmul_flops_per_call * calls_expected
    softmax_flops_total = softmax_flops_per_call * calls_expected

    runtime_ratio = (matmul_ms / softmax_ms) if softmax_ms > 0 else float("inf")
    flops_ratio = (matmul_flops_total / softmax_flops_total) if softmax_flops_total > 0 else float("inf")
    matmul_runtime_fraction = (matmul_ms / attn_total_ms) if attn_total_ms > 0 else float("nan")
    softmax_runtime_fraction = (softmax_ms / attn_total_ms) if attn_total_ms > 0 else float("nan")

    print("Attention kernel comparison (forward pass):")
    print(f"  model_size={model_size}, context_length={context_length}, batch_size={args.batch_size}")
    print(f"  measured_attention_calls={timing['calls']} (expected≈{calls_expected})")
    print(f"  qk_matmul_ms={timing['qk_ms']:.3f}")
    print(f"  av_matmul_ms={timing['av_ms']:.3f}")
    print(f"  softmax_ms={softmax_ms:.3f}")
    print(f"  total_attention_profiled_ms={attn_total_ms:.3f}")
    print(f"  total_forward_window_ms={total_forward_ms:.3f}")
    print(f"  matmul_runtime_fraction={matmul_runtime_fraction:.4f}")
    print(f"  softmax_runtime_fraction={softmax_runtime_fraction:.4f}")
    print(f"  matmul_vs_softmax_runtime_ratio={runtime_ratio:.3f}")
    print(f"  matmul_vs_softmax_flops_ratio={flops_ratio:.3f}")

    return {
        "qk_ms": timing["qk_ms"],
        "av_ms": timing["av_ms"],
        "softmax_ms": softmax_ms,
        "matmul_runtime_fraction": matmul_runtime_fraction,
        "softmax_runtime_fraction": softmax_runtime_fraction,
        "runtime_ratio": runtime_ratio,
        "flops_ratio": float(flops_ratio),
    }


def compare_bf16_by_model_size(args: argparse.Namespace) -> dict[str, dict[str, dict[str, BenchmarkMetrics]]]:
    device = get_device()
    use_nvtx = not args.disable_nvtx
    if device.type != "cuda":
        print("Warning: CUDA not available; BF16 autocast is disabled and results reflect full precision execution.")

    model_sizes = [args.model_size] if args.model_size else ["small", "medium", "large"]
    context_length = args.context_length or 128
    modes: list[Mode] = ["forward", "backward"]

    results: dict[str, dict[str, dict[str, BenchmarkMetrics]]] = {}

    for model_size in model_sizes:
        print(f"BF16 comparison for model_size={model_size}, context_length={context_length}")
        results[model_size] = {"fp32": {}, "bf16": {}}

        for precision_name, use_bf16 in (("fp32", False), ("bf16", True)):
            model = build_model(model_size, context_length, args.vocab_size, args.rope_theta, device)
            batch = make_batch(args.vocab_size, args.batch_size, context_length, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            for mode in modes:
                metrics = benchmark_mode(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    mode=mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    device=device,
                    use_nvtx=use_nvtx,
                    use_bf16=use_bf16,
                    memory_profile=False,
                    snapshot_path=None,
                )
                results[model_size][precision_name][mode] = metrics
                if metrics["status"] == "oom":
                    print(f"  {precision_name} {mode}: OOM")
                else:
                    print(f"  {precision_name} {mode}: avg={metrics['avg_ms']:.3f}ms std={metrics['std_ms']:.3f}ms")

        fp32_fwd = results[model_size]["fp32"]["forward"]
        bf16_fwd = results[model_size]["bf16"]["forward"]
        fp32_bwd = results[model_size]["fp32"]["backward"]
        bf16_bwd = results[model_size]["bf16"]["backward"]
        if (
            fp32_fwd["status"] == "ok"
            and bf16_fwd["status"] == "ok"
            and fp32_bwd["status"] == "ok"
            and bf16_bwd["status"] == "ok"
            and bf16_fwd["avg_ms"] > 0
            and bf16_bwd["avg_ms"] > 0
        ):
            fwd_speedup = fp32_fwd["avg_ms"] / bf16_fwd["avg_ms"]
            bwd_speedup = fp32_bwd["avg_ms"] / bf16_bwd["avg_ms"]
            print(f"  speedup: forward={fwd_speedup:.3f}x, backward={bwd_speedup:.3f}x")

    print("Trend summary (interpretation):")
    print("  Compare speedup values across small→medium→large; larger models often see larger BF16 gains on CUDA tensor-core hardware.")
    print("  Backward can differ from forward because gradient kernels and optimizer-related work have different precision sensitivity.")
    return results


def benchmark_attention_scaling(args: argparse.Namespace | None = None) -> list[AttentionBenchmarkMetrics]:
    device = get_device()
    use_nvtx = not (args.disable_nvtx if args is not None else False)

    if device.type != "cuda":
        print("Warning: attention scaling benchmark is intended for CUDA; memory measurements will be zero on CPU.")

    print("Attention scaling benchmark: batch_size=8, num_heads=1")
    print("Rows report forward/backward averages over 100 passes and memory in use before backward starts.")
    print("| d_model | seq_len | forward_ms | backward_ms | mem_before_backward_MB | saved_for_backward_MB | status |")
    print("| --- | --- | ---: | ---: | ---: | ---: | --- |")

    results: list[AttentionBenchmarkMetrics] = []

    for head_dim in ATTENTION_HEAD_DIMS:
        for sequence_length in ATTENTION_SEQUENCE_LENGTHS:
            batch_size = ATTENTION_BATCH_SIZE
            num_heads = ATTENTION_HEAD_COUNTS[0]
            saved_for_backward_bytes = estimate_attention_saved_for_backward_bytes(
                batch_size=batch_size,
                num_heads=num_heads,
                sequence_length=sequence_length,
                head_dim=head_dim,
                dtype_bytes=4,
            )
            status = "ok"
            forward_avg_ms = float("nan")
            backward_avg_ms = float("nan")
            mem_before_backward_bytes = 0

            mask = build_causal_mask(batch_size, num_heads, sequence_length, device)

            try:
                # Warm up the forward path.
                q, k, v = make_attention_inputs(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    device=device,
                    requires_grad=False,
                )
                for _ in range(ATTENTION_WARMUP_ITERS):
                    synchronize_cuda(device)
                    with torch.no_grad():
                        _ = basics_model_module.scaled_dot_product_attention(q, k, v, mask=mask)
                    synchronize_cuda(device)

                forward_times_ms: list[float] = []
                for _ in range(ATTENTION_FORWARD_ITERS):
                    synchronize_cuda(device)
                    t0 = timeit.default_timer()
                    with torch.no_grad():
                        _ = basics_model_module.scaled_dot_product_attention(q, k, v, mask=mask)
                    synchronize_cuda(device)
                    t1 = timeit.default_timer()
                    forward_times_ms.append((t1 - t0) * 1000.0)

                forward_avg_ms = statistics.mean(forward_times_ms) if forward_times_ms else float("nan")

                # Warm up backward on a separate graph, then measure memory before the timed backward loop.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass

                q, k, v = make_attention_inputs(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    device=device,
                    requires_grad=True,
                )

                for _ in range(ATTENTION_WARMUP_ITERS):
                    q.grad = None
                    k.grad = None
                    v.grad = None
                    with nvtx_range("attention_forward", use_nvtx):
                        output = basics_model_module.scaled_dot_product_attention(q, k, v, mask=mask)
                        loss = output.sum()
                    with nvtx_range("attention_backward", use_nvtx):
                        loss.backward()
                    synchronize_cuda(device)

                q.grad = None
                k.grad = None
                v.grad = None
                with nvtx_range("attention_forward", use_nvtx):
                    output = basics_model_module.scaled_dot_product_attention(q, k, v, mask=mask)
                    loss = output.sum()
                synchronize_cuda(device)
                if device.type == "cuda":
                    mem_before_backward_bytes = int(torch.cuda.memory_allocated())

                backward_times_ms: list[float] = []
                for _ in range(ATTENTION_BACKWARD_ITERS):
                    q.grad = None
                    k.grad = None
                    v.grad = None
                    synchronize_cuda(device)
                    t0 = timeit.default_timer()
                    with nvtx_range("attention_backward", use_nvtx):
                        loss.backward(retain_graph=True)
                    synchronize_cuda(device)
                    t1 = timeit.default_timer()
                    backward_times_ms.append((t1 - t0) * 1000.0)

                backward_avg_ms = statistics.mean(backward_times_ms) if backward_times_ms else float("nan")

            except torch.cuda.OutOfMemoryError:
                status = "oom"
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            result: AttentionBenchmarkMetrics = {
                "d_model": head_dim,
                "sequence_length": sequence_length,
                "forward_avg_ms": forward_avg_ms,
                "backward_avg_ms": backward_avg_ms,
                "mem_before_backward_bytes": mem_before_backward_bytes,
                "saved_for_backward_bytes": saved_for_backward_bytes,
                "status": status,
            }
            results.append(result)

            if status == "oom":
                print(f"| {head_dim} | {sequence_length} | OOM | OOM | OOM | OOM | oom |")
            else:
                mem_before_backward_mb = mem_before_backward_bytes / (1024 ** 2)
                saved_for_backward_mb = saved_for_backward_bytes / (1024 ** 2)
                print(
                    f"| {head_dim} | {sequence_length} | {forward_avg_ms:.3f} | {backward_avg_ms:.3f} | "
                    f"{mem_before_backward_mb:.3f} | {saved_for_backward_mb:.3f} | ok |"
                )

    print("Notes:")
    print("- `mem_before_backward_MB` is the live CUDA memory after the forward pass and before any backward kernel runs.")
    print("- `saved_for_backward_MB` is a simple activation estimate: Q/K/V plus the attention matrix term, which scales as O(batch * heads * seq_len^2).")
    print("- To remove the backward memory cost, use attention kernels that do not materialize the full attention matrix (for example, FlashAttention-style fused kernels or recomputation/checkpointing).")
    return results


def benchmark_compiled_attention_compare(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    device = get_device()
    use_nvtx = not args.disable_nvtx
    attention_fn = basics_model_module.scaled_dot_product_attention
    compiled_attention_fn = maybe_compile_callable(basics_model_module.scaled_dot_product_attention)

    if device.type != "cuda":
        print("Warning: compiled attention comparison is intended for CUDA; timings may not reflect GPU performance.")

    print("Compiled attention comparison (batch_size=8, num_heads=1):")
    print("| d_model | seq_len | eager_forward_ms | compiled_forward_ms | eager_backward_ms | compiled_backward_ms | forward_speedup | backward_speedup | status |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    results: list[dict[str, float | int | str]] = []

    for head_dim in ATTENTION_HEAD_DIMS:
        for sequence_length in ATTENTION_SEQUENCE_LENGTHS:
            eager_metrics = benchmark_attention_case(
                attention_fn=attention_fn,
                batch_size=ATTENTION_BATCH_SIZE,
                num_heads=ATTENTION_HEAD_COUNTS[0],
                head_dim=head_dim,
                sequence_length=sequence_length,
                device=device,
                use_nvtx=use_nvtx,
            )
            compiled_metrics = benchmark_attention_case(
                attention_fn=compiled_attention_fn,
                batch_size=ATTENTION_BATCH_SIZE,
                num_heads=ATTENTION_HEAD_COUNTS[0],
                head_dim=head_dim,
                sequence_length=sequence_length,
                device=device,
                use_nvtx=use_nvtx,
            )

            status = "ok"
            if eager_metrics["status"] != "ok" or compiled_metrics["status"] != "ok":
                status = "oom"

            row = {
                "d_model": head_dim,
                "sequence_length": sequence_length,
                "eager_forward_ms": float(eager_metrics["forward_avg_ms"]),
                "compiled_forward_ms": float(compiled_metrics["forward_avg_ms"]),
                "eager_backward_ms": float(eager_metrics["backward_avg_ms"]),
                "compiled_backward_ms": float(compiled_metrics["backward_avg_ms"]),
                "status": status,
            }
            results.append(row)

            if status != "ok":
                print(f"| {head_dim} | {sequence_length} | OOM | OOM | OOM | OOM | OOM | OOM | oom |")
                continue

            forward_speedup = row["eager_forward_ms"] / row["compiled_forward_ms"] if row["compiled_forward_ms"] > 0 else float("nan")
            backward_speedup = row["eager_backward_ms"] / row["compiled_backward_ms"] if row["compiled_backward_ms"] > 0 else float("nan")
            print(
                f"| {head_dim} | {sequence_length} | {row['eager_forward_ms']:.3f} | {row['compiled_forward_ms']:.3f} | "
                f"{row['eager_backward_ms']:.3f} | {row['compiled_backward_ms']:.3f} | {forward_speedup:.3f} | {backward_speedup:.3f} | ok |"
            )

    print("Notes:")
    print("- `compiled_*` timings come from `torch.compile` on the standalone attention function, using the same batch/head/sequence grid as the eager benchmark.")
    print("- Backward timing reuses the same scalar loss graph with `retain_graph=True`, matching the profiling setup used in the attention benchmark.")
    return results


def benchmark_compiled_transformer_compare(args: argparse.Namespace) -> dict[str, dict[int, dict[str, dict[str, float | int | str]]]]:
    device = get_device()
    use_nvtx = not args.disable_nvtx
    model_sizes = [args.model_size] if args.model_size else list(MODEL_SPECS)
    context_lengths = [args.context_length] if args.context_length else list(CONTEXT_LENGTHS)
    modes: list[Mode] = ["forward", "backward", "optimizer", "train-step"]

    if device.type != "cuda":
        print("Warning: compiled Transformer comparison is intended for CUDA; timings may not reflect GPU performance.")

    print("Compiled Transformer comparison:")
    print("| model_size | context_len | mode | vanilla_ms | compiled_ms | speedup | status |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- |")

    results: dict[str, dict[int, dict[str, dict[str, float | int | str]]]] = {}

    for model_size in model_sizes:
        results[model_size] = {}
        for context_length in context_lengths:
            print(f"Benchmarking model_size={model_size}, context_length={context_length}")
            vanilla_model = build_model(model_size, context_length, args.vocab_size, args.rope_theta, device, compile_model=False)
            compiled_model = build_model(model_size, context_length, args.vocab_size, args.rope_theta, device, compile_model=True)
            vanilla_optimizer = torch.optim.AdamW(vanilla_model.parameters(), lr=1e-3)
            compiled_optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-3)
            batch = make_batch(args.vocab_size, args.batch_size, context_length, device)

            results[model_size][context_length] = {}
            for mode in modes:
                vanilla_metrics = benchmark_mode(
                    model=vanilla_model,
                    batch=batch,
                    optimizer=vanilla_optimizer,
                    mode=mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    device=device,
                    use_nvtx=use_nvtx,
                    use_bf16=args.use_bf16,
                )
                compiled_metrics = benchmark_mode(
                    model=compiled_model,
                    batch=batch,
                    optimizer=compiled_optimizer,
                    mode=mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    device=device,
                    use_nvtx=use_nvtx,
                    use_bf16=args.use_bf16,
                )

                status = "ok" if vanilla_metrics["status"] == "ok" and compiled_metrics["status"] == "ok" else "oom"
                row = {
                    "vanilla_ms": float(vanilla_metrics["avg_ms"]),
                    "compiled_ms": float(compiled_metrics["avg_ms"]),
                    "speedup": float(vanilla_metrics["avg_ms"] / compiled_metrics["avg_ms"]) if compiled_metrics["avg_ms"] > 0 else float("nan"),
                    "status": status,
                }
                results[model_size][context_length][mode] = row

                if status != "ok":
                    print(f"| {model_size} | {context_length} | {mode} | OOM | OOM | OOM | oom |")
                else:
                    print(
                        f"| {model_size} | {context_length} | {mode} | {row['vanilla_ms']:.3f} | {row['compiled_ms']:.3f} | {row['speedup']:.3f} | ok |"
                    )

    print("Notes:")
    print("- `compiled_ms` comes from `torch.compile` applied to the full Transformer module before constructing the optimizer.")
    print("- The table includes forward, backward, optimizer, and full train-step modes so you can compare the performance impact at different levels of the training loop.")
    return results


def profile_grid(args: argparse.Namespace) -> dict[str, dict[int, dict[str, BenchmarkMetrics]]]:
    device = get_device()
    model_sizes = [args.model_size] if args.model_size else ["small", "medium", "large"]
    context_lengths = [args.context_length] if args.context_length else list(CONTEXT_LENGTHS)
    if args.compare_forward_train_step:
        modes: list[Mode] = ["forward", "train-step"]
    else:
        modes = [args.timing] if args.timing != "all" else ["forward", "backward", "optimizer", "train-step"]
    use_nvtx = not args.disable_nvtx

    print(f"device={device}, nvtx_enabled={use_nvtx}")
    results: dict[str, dict[int, dict[str, BenchmarkMetrics]]] = {}

    for model_size in model_sizes:
        results[model_size] = {}
        for context_length in context_lengths:
            print(f"Running model_size={model_size}, context_length={context_length}")
            model = build_model(model_size, context_length, args.vocab_size, args.rope_theta, device)
            batch = make_batch(args.vocab_size, args.batch_size, context_length, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

            results[model_size][context_length] = {}
            for mode in modes:
                # Configure memory profiling for the large model only when requested.
                if args.memory_profile and model_size == "large":
                    snapshot_path = f"memory_snapshot_{model_size}_{context_length}_{mode}.pickle"
                    mem_profile_flag = True
                else:
                    snapshot_path = None
                    mem_profile_flag = False

                metrics = benchmark_mode(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    mode=mode,
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    device=device,
                    use_nvtx=use_nvtx,
                    use_bf16=args.use_bf16,
                    memory_profile=mem_profile_flag,
                    snapshot_path=snapshot_path,
                )
                results[model_size][context_length][mode] = metrics
                if metrics["status"] == "oom":
                    print(f"  {mode}: OOM")
                else:
                    peak_mb = metrics.get("peak_mem_bytes", 0) / (1024 ** 2)
                    print(
                        f"  {mode}: avg={metrics['avg_ms']:.3f}ms std={metrics['std_ms']:.3f}ms peak={peak_mb:.3f} MB"
                    )

            if args.compare_forward_train_step:
                fwd = results[model_size][context_length].get("forward")
                train = results[model_size][context_length].get("train-step")
                if fwd and train and fwd["status"] == "ok" and train["status"] == "ok":
                    ratio = train["avg_ms"] / fwd["avg_ms"] if fwd["avg_ms"] > 0 else float("nan")
                    print(
                        f"  compare: train_step/forward avg-time ratio = {ratio:.3f}x "
                        f"(use Nsight kernel summary to compare matmul and other kernel fractions)"
                    )

    return results


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.compare_compiled_attention:
        benchmark_compiled_attention_compare(args)
        return
    if args.compare_compiled_transformer:
        benchmark_compiled_transformer_compare(args)
        return
    if args.benchmark_attention_scaling:
        benchmark_attention_scaling(args)
        return
    if args.compare_bf16:
        compare_bf16_by_model_size(args)
        return
    if args.compare_softmax_matmul:
        compare_softmax_vs_matmul(args)
        return
    profile_grid(args)


if __name__ == "__main__":
    main()
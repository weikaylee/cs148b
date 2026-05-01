from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    # Tokenize without special tokens to match test expectations
    batch_input_ids = []
    batch_labels = []
    batch_response_mask = []

    # Build full sequences (prompt + output) per example
    full_sequences = [tokenizer.encode(p, add_special_tokens=False) + tokenizer.encode(o, add_special_tokens=False) for p, o in zip(prompt_strs, output_strs)]

    # Sequence lengths for shifted labels (drop first or last accordingly)
    seq_lens = [len(seq) - 1 for seq in full_sequences]
    max_len = max(seq_lens) if seq_lens else 0

    for (p, o), full_seq in zip(zip(prompt_strs, output_strs), full_sequences):
        prompt_ids = tokenizer.encode(p, add_special_tokens=False)
        output_ids = tokenizer.encode(o, add_special_tokens=False)

        # input ids are full_seq[:-1], labels are full_seq[1:]
        input_ids = full_seq[:-1]
        labels = full_seq[1:]

        pad_len = max_len - len(input_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0)

        input_ids_padded = input_ids + [pad_id] * pad_len
        labels_padded = labels + [pad_id] * pad_len

        # response mask aligns with labels: True where label token corresponds to response tokens
        prompt_len = len(prompt_ids)
        response_len = len(output_ids)
        # labels correspond to tokens shifted by 1, so label positions that are response tokens
        # start at (prompt_len - 1) and cover response_len positions
        response_mask = [False] * (prompt_len - 1)
        response_mask += [True] * response_len
        # pad remainder
        response_mask += [False] * pad_len

        batch_input_ids.append(input_ids_padded)
        batch_labels.append(labels_padded)
        batch_response_mask.append(response_mask)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
        "response_mask": torch.tensor(batch_response_mask),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    # logits: (..., vocab)
    # Use a numerically stable formulation via logsumexp.
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_probs = logits - log_z
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy

# this tisthe log likelihood of the response. given the input, the model outputs some prediction 
# for each input over each score. we convert those scores into log porbaiblities. we get the probiablity that the correct token was chosen, and sum!
# remmeber, the model outputs (batch, num_input_tokens, vocab_size), predicting the next token after seeing num_input_tokens[t] 
def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    # Run model to obtain logits
    out = model(input_ids)
    logits = out.logits

    log_probs = torch.log_softmax(logits, dim=-1)
    # gather log-probs for the provided labels
    label_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, Tensor] = {"log_probs": label_log_probs}
    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)
    return result


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    # Ensure mask is broadcastable and numeric
    mask_float = mask.to(dtype=tensor.dtype)
    masked = tensor * mask_float
    if dim is None:
        summ = masked.sum()
    else:
        summ = masked.sum(dim=dim)
    return summ / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("rollout_responses and repeated_ground_truths must have the same length")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if len(rollout_responses) % group_size != 0:
        raise ValueError("rollout_responses length must be divisible by group_size")

    reward_values = []
    format_rewards = []
    answer_rewards = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        info = reward_fn(response, gt)
        reward_values.append(float(info.get("reward", 0.0)))
        format_rewards.append(float(info.get("format_reward", 0.0)))
        answer_rewards.append(float(info.get("answer_reward", 0.0)))

    raw_rewards = torch.tensor(reward_values, dtype=torch.float32)
    grouped = raw_rewards.view(-1, group_size)
    group_mean = grouped.mean(dim=1, keepdim=True)
    centered = grouped - group_mean

    if normalize_by_std:
        group_std = grouped.std(dim=1, keepdim=True, unbiased=False)
        normalized = centered / (group_std + advantage_eps)
    else:
        normalized = centered

    advantages = normalized.reshape(-1)

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "std_reward": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() else 0.0,
        "mean_format_reward": float(torch.tensor(format_rewards).mean().item()) if format_rewards else 0.0,
        "mean_answer_reward": float(torch.tensor(answer_rewards).mean().item()) if answer_rewards else 0.0,
    }

    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - cliprange, 1 + cliprange)
    broadcast_advantages = advantages.expand_as(policy_log_probs)
    unclipped = ratios * broadcast_advantages
    clipped = clipped_ratios * broadcast_advantages
    loss = -torch.minimum(unclipped, clipped)
    # A token is clipped when the clipped term is strictly lower than the unclipped term.
    is_clipped = clipped < unclipped
    return loss, {"is_clipped": is_clipped}


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    logs: list[dict[str, Any]] = []

    response_lengths = [len(r.split()) for r in responses]
    correct_mask = [bool(info.get("answer_reward", 0.0) >= 1.0) for info in reward_infos]

    avg_len = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
    correct_lengths = [l for l, ok in zip(response_lengths, correct_mask) if ok]
    incorrect_lengths = [l for l, ok in zip(response_lengths, correct_mask) if not ok]
    avg_len_correct = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0.0
    avg_len_incorrect = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0.0

    for idx, (prompt, response, gt, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos)
    ):
        entry: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "ground_truth": gt,
            "reward_info": reward_info,
            "avg_response_length": avg_len,
            "avg_response_length_correct": avg_len_correct,
            "avg_response_length_incorrect": avg_len_incorrect,
            "response_length": response_lengths[idx],
        }
        if token_entropies is not None:
            entry["avg_token_entropy"] = token_entropies[idx]
        logs.append(entry)

    return logs


def _generate_rollouts(
    policy: torch.nn.Module,
    tokenizer,
    prompt_strs: list[str],
    device: torch.device,
    temperature: float,
    min_new_tokens: int,
    max_new_tokens: int,
    stop_str: str = "</answer>",
) -> list[str]:
    """Generate responses from the policy using HF generation (no vLLM dependency)."""
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    encoded = tokenizer(
        prompt_strs,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    policy.eval()
    with torch.no_grad():
        out_ids = policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    policy.train()
    tokenizer.padding_side = orig_padding_side

    responses = []
    for ids in out_ids:
        text = tokenizer.decode(ids[prompt_len:], skip_special_tokens=True)
        if stop_str in text:
            text = text[: text.index(stop_str) + len(stop_str)]
        responses.append(text)
    return responses


def _validate(
    policy: torch.nn.Module,
    tokenizer,
    val_examples: list[dict],
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: str,
    device: torch.device,
    val_size: int,
    val_sampling_max_tokens: int,
) -> dict[str, float]:
    """Greedy-decode a validation slice and return mean rewards."""
    examples = val_examples[:val_size]
    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]
    ground_truths = [ex["answer"] for ex in examples]

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_responses: list[str] = []
    chunk = 16
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(prompts), chunk):
            enc = tokenizer(
                prompts[start : start + chunk],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            plen = ids.shape[1]
            out = policy.generate(
                input_ids=ids,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=val_sampling_max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for row in out:
                text = tokenizer.decode(row[plen:], skip_special_tokens=True)
                if "</answer>" in text:
                    text = text[: text.index("</answer>") + len("</answer>")]
                all_responses.append(text)
    policy.train()
    tokenizer.padding_side = orig_padding_side

    rewards = [reward_fn(r, gt) for r, gt in zip(all_responses, ground_truths)]
    mean_reward = sum(float(r.get("reward", 0.0)) for r in rewards) / len(rewards)
    mean_format = sum(float(r.get("format_reward", 0.0)) for r in rewards) / len(rewards)
    mean_answer = sum(float(r.get("answer_reward", 0.0)) for r in rewards) / len(rewards)
    return {"val_reward": mean_reward, "val_format_reward": mean_format, "val_answer_reward": mean_answer}


def train_grpo(
    policy: torch.nn.Module,
    tokenizer,
    train_examples: list[dict],
    val_examples: list[dict],
    reward_fn: Callable[[str, str], dict[str, float]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prompt_template: str,
    n_grpo_steps: int = 8,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    advantage_eps: float = 1e-6,
    normalize_by_std: bool = True,
    val_interval: int = 5,
    val_size: int = 256,
    val_sampling_max_tokens: int = 512,
    output_dir: Path | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    random.seed(seed)

    assert train_batch_size % gradient_accumulation_steps == 0
    micro_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0
    n_prompts_per_rollout = rollout_batch_size // group_size

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []

    policy.train()
    for step in range(n_grpo_steps):
        # ── 1. Sample prompts and repeat for group rollouts ────────────────────
        sampled = random.sample(train_examples, n_prompts_per_rollout)
        # Each prompt repeated group_size times (contiguous groups)
        repeated_examples = [ex for ex in sampled for _ in range(group_size)]
        prompt_strs = [prompt_template.format(question=ex["question"]) for ex in repeated_examples]
        ground_truths = [ex["answer"] for ex in repeated_examples]

        # ── 2. Generate rollouts ────────────────────────────────────────────────
        responses = _generate_rollouts(
            policy=policy,
            tokenizer=tokenizer,
            prompt_strs=prompt_strs,
            device=device,
            temperature=sampling_temperature,
            min_new_tokens=sampling_min_tokens,
            max_new_tokens=sampling_max_tokens,
        )

        # ── 3. Rewards and advantages ──────────────────────────────────────────
        advantages, _, reward_meta = compute_group_normalized_rewards(
            reward_fn=reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )
        # advantages: (rollout_batch_size,) → (rollout_batch_size, 1) for broadcasting
        advantages = advantages.unsqueeze(1).to(device)

        # ── 4. Tokenize rollout batch ─────────────────────────────────────────
        tokenized = tokenize_prompt_and_output(prompt_strs, responses, tokenizer)
        all_input_ids = tokenized["input_ids"]       # (B, L)
        all_labels = tokenized["labels"]             # (B, L)
        all_response_mask = tokenized["response_mask"]  # (B, L)

        # ── 5. Cache old log probs (no gradient) ─────────────────────────────
        policy.eval()
        with torch.no_grad():
            old_result = get_response_log_probs(
                model=policy,
                input_ids=all_input_ids.to(device),
                labels=all_labels.to(device),
                return_token_entropy=False,
            )
        old_log_probs = old_result["log_probs"].detach().cpu()
        policy.train()

        # ── 6. Train for epochs_per_rollout_batch epochs ──────────────────────
        step_losses: list[float] = []
        grad_norm = 0.0
        for _ in range(epochs_per_rollout_batch):
            indices = list(range(rollout_batch_size))
            random.shuffle(indices)

            optimizer.zero_grad()
            for mb_start in range(0, rollout_batch_size, micro_batch_size):
                mb_idx = indices[mb_start : mb_start + micro_batch_size]

                mb_input = all_input_ids[mb_idx].to(device)
                mb_labels = all_labels[mb_idx].to(device)
                mb_mask = all_response_mask[mb_idx].to(device)
                mb_adv = advantages[mb_idx]
                mb_old = old_log_probs[mb_idx].to(device)

                policy_result = get_response_log_probs(
                    model=policy,
                    input_ids=mb_input,
                    labels=mb_labels,
                    return_token_entropy=False,
                )
                mb_log_probs = policy_result["log_probs"]

                loss, _ = grpo_microbatch_train_step(
                    policy_log_probs=mb_log_probs,
                    response_mask=mb_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    advantages=mb_adv,
                    old_log_probs=mb_old,
                    cliprange=cliprange,
                )
                step_losses.append(loss.item())

            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
            optimizer.step()

        # ── 7. Log step metrics ────────────────────────────────────────────────
        step_log: dict[str, Any] = {
            "step": step,
            "loss": sum(step_losses) / len(step_losses) if step_losses else 0.0,
            "grad_norm": grad_norm,
            **reward_meta,
        }
        history.append(step_log)
        print(
            f"[step {step:3d}] loss={step_log['loss']:.4f}  "
            f"grad_norm={grad_norm:.3f}  "
            f"reward={reward_meta['mean_reward']:.3f}  "
            f"answer={reward_meta['mean_answer_reward']:.3f}"
        )

        # ── 8. Periodic validation ─────────────────────────────────────────────
        if (step + 1) % val_interval == 0 or step == n_grpo_steps - 1:
            val_metrics = _validate(
                policy=policy,
                tokenizer=tokenizer,
                val_examples=val_examples,
                reward_fn=reward_fn,
                prompt_template=prompt_template,
                device=device,
                val_size=val_size,
                val_sampling_max_tokens=val_sampling_max_tokens,
            )
            val_log = {"step": step, **val_metrics}
            val_history.append(val_log)
            print(
                f"  [val  step {step:3d}] "
                f"reward={val_metrics['val_reward']:.3f}  "
                f"answer={val_metrics['val_answer_reward']:.3f}"
            )
            if output_dir is not None:
                import json
                Path(output_dir, f"val_step{step:04d}.json").write_text(
                    json.dumps(val_log, indent=2)
                )

    return {"history": history, "val_history": val_history}

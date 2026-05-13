"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from vlm.masking import build_image_bidir_mask

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]

# HF cross-entropy ignores positions with this label, so we use it to mask both
# (a) visual-token positions (the model shouldn't be asked to predict images)
# and (b) prompt/padding positions for answer-only VQA training.
IGNORE_INDEX = -100


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder.

    Args:
        vit:       Your CLIP-pretrained ViT from §3.
        projector: vlm.projector.VisionLanguageProjector instance.
        decoder:   HuggingFace causal LM (e.g., SmolLM2-360M-Instruct) loaded
                   in bf16 with FlashAttention-2.
        tokenizer: Matching HF tokenizer.
        image_token_id: Token ID corresponding to the special <image> placeholder
                        in interleaved mode (None for cls / all_patches modes).

    Forward:
        images:         (B, 3, H, W) float tensor.
        input_ids:      (B, T) tokenized text.
        attention_mask: (B, T) text attention mask from the tokenizer.
        labels:         (B, T) for loss computation, or None for inference.
                        Visual-token positions are added internally; the caller
                        only needs to mask question/padding positions with -100
                        if doing answer-only training.
        injection:      One of "cls", "all_patches", "interleaved".
        mask_mode:      One of "causal", "image_bidir".

    Returns:
        A dict with at least:
          - "loss":   scalar (only if labels was provided).
          - "logits": (B, T_total, vocab_size).
    """

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # NOTE: rememeber, attention mask gets added to qt / sqrt() term before softmax.
    # so, if you want to mask out a value, add -inf so that it gets pusehd to 0 during softmax. 
    # for vals you want to keep, add 0! yippeeee. 
    def _encode_visual(
        self, images: torch.Tensor, injection: InjectionMode
    ) -> torch.Tensor:
        """Run ViT + projector. Returns (B, n_visual, d_decoder)."""
        if injection == "cls":
            feats = self.vit(images)  # (B, d_image)
        else:
            # all_patches / interleaved both need the full visual-token sequence.
            feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_image)
        return self.projector(feats)  # projector promotes 2D -> (B, 1, d_decoder)

    def _prepare_inputs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
        injection: InjectionMode,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Stitch visual tokens into the text stream and extend attention_mask
        and labels accordingly.

        Returns (inputs_embeds, attention_mask_2d, labels_or_None, n_visual).
        """
        B = input_ids.shape[0]
        visual_embeds = self._encode_visual(images, injection)  # (B, n_visual, d_decoder)
        n_visual = visual_embeds.shape[1]

        # The decoder's own input embedding maps text token IDs into its
        # embedding space; reuse it so visual + text tokens live in the same dim.
        text_embeds = self.decoder.get_input_embeddings()(input_ids)  # (B, T, d_decoder)
        # Match the decoder's dtype (often bf16) so torch.cat doesn't error if
        # the projector still emits fp32.
        visual_embeds = visual_embeds.to(text_embeds.dtype)
        device = visual_embeds.device

        if injection in ("cls", "all_patches"):
            # Layout: [v_1, ..., v_{n_visual}, t_1, ..., t_T]
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            v_attn = torch.ones(
                B, n_visual, dtype=attention_mask.dtype, device=device
            )
            attn_out = torch.cat([v_attn, attention_mask], dim=1)
            if labels is not None:
                v_labels = torch.full(
                    (B, n_visual), IGNORE_INDEX, dtype=labels.dtype, device=device
                )
                labels_out = torch.cat([v_labels, labels], dim=1)
            else:
                labels_out = None

        elif injection == "interleaved":
            if self.image_token_id is None:
                raise ValueError(
                    "interleaved injection requires image_token_id on the model."
                )
            # Per-example splice: each example must contain exactly one <image>
            # token, which is replaced by the n_visual projected patch tokens.
            new_embeds, new_attn, new_labels = [], [], []
            v_attn = torch.ones(n_visual, dtype=attention_mask.dtype, device=device)
            v_labels = (
                torch.full((n_visual,), IGNORE_INDEX, dtype=labels.dtype, device=device)
                if labels is not None
                else None
            )
            for b in range(B):
                pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=True)[0]
                if pos.numel() != 1:
                    raise ValueError(
                        f"interleaved: expected exactly one image_token_id in "
                        f"example {b}; got {pos.numel()}."
                    )
                i = int(pos.item())
                new_embeds.append(
                    torch.cat(
                        [text_embeds[b, :i], visual_embeds[b], text_embeds[b, i + 1 :]],
                        dim=0,
                    )
                )
                new_attn.append(
                    torch.cat(
                        [attention_mask[b, :i], v_attn, attention_mask[b, i + 1 :]],
                        dim=0,
                    )
                )
                if labels is not None:
                    new_labels.append(
                        torch.cat([labels[b, :i], v_labels, labels[b, i + 1 :]], dim=0)
                    )
            inputs_embeds = torch.stack(new_embeds, dim=0)
            attn_out = torch.stack(new_attn, dim=0)
            labels_out = torch.stack(new_labels, dim=0) if labels is not None else None

        else:
            raise ValueError(f"unknown injection mode: {injection!r}")

        return inputs_embeds, attn_out, labels_out, n_visual

    # ------------------------------------------------------------------
    # Forward / generate
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        inputs_embeds, attn_2d, new_labels, n_visual = self._prepare_inputs(
            images, input_ids, attention_mask, labels, injection
        )
        T_total = inputs_embeds.shape[1]

        if mask_mode == "image_bidir":
            # build_image_bidir_mask assumes the [v..., t...] layout, which is
            # only produced by cls / all_patches. For interleaved we keep causal.
            if injection == "interleaved":
                raise ValueError(
                    "image_bidir is defined for [visual, text] layouts; "
                    "use mask_mode='causal' with injection='interleaved'."
                )
            device, dtype = inputs_embeds.device, inputs_embeds.dtype
            n_text = T_total - n_visual
            bidir = build_image_bidir_mask(n_visual, n_text, device, dtype)  # (1,1,T,T)
            # Add a per-batch padding-column bias so no query attends to padded
            # text positions. attn_2d covers both the visual prefix and the text.
            zero = torch.zeros((), dtype=dtype, device=device)
            neg = torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
            pad_additive = torch.where(attn_2d.bool(), zero, neg)[:, None, None, :]
            attn_for_hf = bidir + pad_additive  # (B, 1, T_total, T_total)
        else:
            attn_for_hf = attn_2d

        kwargs = dict(inputs_embeds=inputs_embeds, attention_mask=attn_for_hf)
        if new_labels is not None:
            kwargs["labels"] = new_labels
        out = self.decoder(**kwargs)
        return {
            "loss": out.loss if new_labels is not None else None,
            "logits": out.logits,
        }

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text continuations conditioned on images + prompts.

        Uses causal masking during generation (KV cache + bidir image attention
        don't compose cleanly post-prefill). The visual tokens are placed into
        `inputs_embeds`, and HF's generate handles the autoregressive loop.

        Returns a list of decoded continuations (no input echo).
        """
        was_training = self.training
        self.eval()
        try:
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            enc = {k: v.to(images.device) for k, v in enc.items()}
            inputs_embeds, attn_2d, _, _ = self._prepare_inputs(
                images=images,
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=None,
                injection=injection,
            )
            out_ids = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_2d,
                max_new_tokens=max_new_tokens,
                **gen_kwargs,
            )
            return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        finally:
            self.train(was_training)

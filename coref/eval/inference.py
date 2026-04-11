"""
inference.py — Controlled inference engine for CorefInst.

Key idea (Section 3.3 of the paper):
  • The model's input contains multiple #MASK tokens — one per mention.
  • Instead of generating the entire output in one shot, we predict each
    #MASK token's cluster number sequentially, feeding every previous
    prediction back into the context before predicting the next one.

Optimised implementation — incremental KV-cache reuse:
  • The prefix (instruction + masked_input + first segment + '#') is
    processed ONCE per frame in a single forward pass.
  • Each successive MASK only runs the model over the tiny delta tokens
    (predicted digit(s) + next segment text + next '#'), NOT the full prompt.
  • Per-MASK prefill cost: O(~10 tokens) instead of O(full_ctx_len).
  • Uses model() directly instead of model.generate(), which avoids the
    slow Transformers 5.5 generate() code path entirely.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prefix_str(tokenizer, instruction: str, masked_input: str) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": masked_input},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _tokenize_left_truncated(tokenizer, text: str, max_len: int, device):
    orig = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
    ).to(device)
    tokenizer.truncation_side = orig
    return enc


def _model_step(model, input_ids, attention_mask, past_key_values):
    """One forward pass; returns (logits at last position, updated past_key_values)."""
    # Unsloth's patched forward requires explicit position_ids when past_key_values
    # is provided — it cannot infer them from the KV cache length on its own.
    if past_key_values is not None:
        kv_len = past_key_values[0][0].shape[2]
        new_len = input_ids.shape[1]
        position_ids = torch.arange(
            kv_len, kv_len + new_len, device=input_ids.device
        ).unsqueeze(0)
    else:
        position_ids = None

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    return out.logits[:, -1, :], out.past_key_values


# ---------------------------------------------------------------------------
# Main controlled-inference routine (one frame)
# ---------------------------------------------------------------------------

def controlled_inference(
    model,
    tokenizer,
    instruction: str,
    masked_input: str,
    device: torch.device,
    max_cluster_id: int = 200,
    max_seq_length: int = 4096,
) -> Tuple[str, List[int]]:
    """
    Controlled inference with incremental KV-cache reuse.

    1. Tokenise [prefix + segments[0] + '#'] and prime the KV cache once.
    2. For each MASK i:
         a. Greedy-decode up to 4 digit tokens from the last logit vector.
         b. Record predicted cluster number.
         c. Process [segments[i+1] + '#'] incrementally (tiny delta).
    3. Reconstruct and return the output string.
    """
    segments = re.split(r"[#@]MASK", masked_input)
    n_masks = len(segments) - 1

    if n_masks == 0:
        return masked_input, []

    # ── Prime KV with [prefix + segments[0] + '#'] ──────────────────────────
    prefix_str  = _build_prefix_str(tokenizer, instruction, masked_input)
    initial_str = prefix_str + segments[0] + "#"

    enc = _tokenize_left_truncated(
        tokenizer, initial_str, max_seq_length - 4, device
    )

    logits, kv = _model_step(model, enc["input_ids"], enc["attention_mask"], None)
    kv_len = enc["input_ids"].shape[1]

    predicted_locals: List[int] = []

    for i in range(n_masks):
        # ── Greedy-decode up to 4 digit tokens ──────────────────────────────
        pred_ids:   List[int] = []
        cur_kv     = kv
        cur_len    = kv_len
        cur_logits = logits

        for _ in range(4):
            next_id  = int(cur_logits.argmax(dim=-1).item())
            next_tok = tokenizer.decode([next_id]).strip()

            if not re.match(r"^\d", next_tok):
                break

            pred_ids.append(next_id)
            tok_t = torch.tensor([[next_id]], device=device)
            mask  = torch.ones(1, cur_len + 1, device=device)
            cur_logits, cur_kv = _model_step(model, tok_t, mask, cur_kv)
            cur_len += 1

        # ── Parse cluster number ─────────────────────────────────────────────
        raw      = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        m        = re.match(r"^\d+", raw)
        pred_num = min(int(m.group()), max_cluster_id) if m else 0
        predicted_locals.append(pred_num)

        # ── Extend KV with [segments[i+1] + '#'] for next MASK ──────────────
        if i + 1 < n_masks:
            chunk     = segments[i + 1] + "#"
            chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)

            if chunk_ids:
                chunk_t = torch.tensor([chunk_ids], device=device)
                mask    = torch.ones(1, cur_len + len(chunk_ids), device=device)
                logits, kv = _model_step(model, chunk_t, mask, cur_kv)
                kv_len     = cur_len + len(chunk_ids)
            else:
                logits, kv, kv_len = cur_logits, cur_kv, cur_len

    # ── Reconstruct output string ────────────────────────────────────────────
    parts = [segments[0]]
    for j, num in enumerate(predicted_locals):
        parts.append(f"#{num}")
        parts.append(segments[j + 1])

    return "".join(parts), predicted_locals


# ---------------------------------------------------------------------------
# Batch inference over a list of FrameExamples
# ---------------------------------------------------------------------------

def run_inference_on_examples(
    model,
    tokenizer,
    examples: list,
    device: Optional[torch.device] = None,
    max_cluster_id: int = 200,
    max_seq_length: int = 4096,
    verbose: bool = False,
) -> List[dict]:
    """
    Run controlled inference on a list of FrameExamples.

    Each item in *examples* must have the fields:
        doc_id, instruction, input (masked), before_mentions, after_mentions,
        before_sent_indices, after_sent_indices

    Returns a list of result dicts with 'predicted_local_no' added to each mention.
    """
    import json

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for idx, ex in enumerate(examples):
        if hasattr(ex, "masked_input"):
            instruction     = ex.instruction
            masked_input    = ex.masked_input
            doc_id          = ex.doc_id
            before_mentions = ex.before_mentions
            after_mentions  = ex.after_mentions
            before_si       = ex.before_sent_indices
            after_si        = ex.after_sent_indices
        else:
            instruction     = ex["instruction"]
            masked_input    = ex["input"]
            doc_id          = ex["doc_id"]
            before_mentions = json.loads(ex["before_mentions"]) if isinstance(ex["before_mentions"], str) else ex["before_mentions"]
            after_mentions  = json.loads(ex["after_mentions"])  if isinstance(ex["after_mentions"],  str) else ex["after_mentions"]
            before_si       = json.loads(ex["before_sent_indices"]) if isinstance(ex["before_sent_indices"], str) else ex["before_sent_indices"]
            after_si        = json.loads(ex["after_sent_indices"])  if isinstance(ex["after_sent_indices"],  str) else ex["after_sent_indices"]

        if verbose and idx % 50 == 0:
            print(f"  Inference [{idx}/{len(examples)}] doc={doc_id} …")

        output_text, predicted_locals = controlled_inference(
            model, tokenizer, instruction, masked_input, device,
            max_cluster_id, max_seq_length,
        )

        all_mentions = list(before_mentions) + list(after_mentions)
        for k, mention in enumerate(all_mentions):
            m_dict = dict(mention)
            m_dict["predicted_local_no"] = predicted_locals[k] if k < len(predicted_locals) else k
            all_mentions[k] = m_dict

        n_before = len(before_mentions)
        results.append({
            "doc_id":              doc_id,
            "before_sent_indices": before_si,
            "after_sent_indices":  after_si,
            "before_mentions":     all_mentions[:n_before],
            "after_mentions":      all_mentions[n_before:],
            "output_text":         output_text,
        })

    return results

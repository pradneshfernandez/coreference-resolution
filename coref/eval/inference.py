"""
inference.py — Controlled inference engine for CorefInst.

Key idea (Section 3.3 of the paper):
  • The model's input contains multiple #MASK tokens — one per mention.
  • Instead of generating the entire output in one shot, we predict each
    #MASK token's cluster number sequentially, feeding every previous
    prediction back into the context before predicting the next one.

Optimised implementation — incremental KV-cache reuse:
  • The prefix (instruction + masked_input + segments[0] + '#') is processed
    ONCE per frame in a single prefill pass.
  • Digit decoding (1–4 tokens per MASK) feeds one token at a time.
  • Chunk extension (segment text between MASKs) is batched into a single
    multi-token forward pass — typically 50–200× fewer calls than one-at-a-time.
  • If the runtime (e.g. unsloth) rejects multi-token decode (assert q_len==1),
    the first failure is caught and all subsequent chunks fall back to
    single-token mode automatically.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import torch


# Module-level flag: start with batched chunk extension, fall back if needed.
_use_batched_chunks: bool = True


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------

def _prefill(model, input_ids, attention_mask):
    """
    Initial prefill forward pass (past_key_values=None).
    Handles multi-token input. Returns (last-position logits, KV cache).
    """
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    return out.logits[:, -1, :], out.past_key_values


def _decode_one(model, token_id: int, past_key_values, cur_len: int, device):
    """
    Single-token decode forward pass.
    Returns (last-position logits, updated KV cache).
    """
    input_ids      = torch.tensor([[token_id]], device=device)
    attention_mask = torch.ones(1, cur_len + 1, dtype=torch.long, device=device)
    position_ids   = torch.tensor([[cur_len]], dtype=torch.long, device=device)

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


def _extend_kv_chunk(model, chunk_ids, kv, kv_len, device):
    """
    Feed a chunk of tokens into the KV cache.

    Tries a single batched multi-token forward first (standard transformers
    path — much faster). If the runtime asserts q_len==1 (unsloth fast path),
    falls back to single-token mode for this call AND flips the module flag
    so all future calls skip the attempt.

    Returns (logits, updated_kv, new_kv_len).
    """
    global _use_batched_chunks

    n = len(chunk_ids)
    if n == 0:
        return None, kv, kv_len

    # ── Try batched multi-token extension ───────────────────────────────
    if _use_batched_chunks and n > 1:
        try:
            chunk_t = torch.tensor([chunk_ids], device=device)
            attn    = torch.ones(1, kv_len + n, dtype=torch.long, device=device)
            pos     = torch.arange(kv_len, kv_len + n,
                                   dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                out = model(
                    input_ids=chunk_t,
                    attention_mask=attn,
                    past_key_values=kv,
                    position_ids=pos,
                    use_cache=True,
                    return_dict=True,
                )
            return out.logits[:, -1, :], out.past_key_values, kv_len + n
        except (AssertionError, RuntimeError):
            _use_batched_chunks = False
            # Fall through to single-token path

    # ── Single-token fallback ───────────────────────────────────────────
    logits = None
    for tok_id in chunk_ids:
        logits, kv = _decode_one(model, tok_id, kv, kv_len, device)
        kv_len += 1
    return logits, kv, kv_len


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prefix_str(tokenizer, instruction: str, masked_input: str) -> str:
    """Apply the chat template to produce the constant per-frame prefix."""
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
    """Tokenise with left-side truncation so the tail (most recent context) is kept."""
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

    Steps:
      1. Tokenise [prefix + segments[0] + '#'] and run one prefill pass.
      2. For each MASK i:
           a. Greedy-decode up to 4 digit tokens (one `_decode_one` call each)
              from the last logit vector; stop at the first non-digit.
           b. Record the predicted cluster number.
           c. If more MASKs remain, tokenise [segments[i+1] + '#'] and feed
              them through `_extend_kv_chunk` (batched or single-token).
      3. Reconstruct and return the output string.
    """
    segments = re.split(r"[#@]MASK", masked_input)
    n_masks = len(segments) - 1

    if n_masks == 0:
        return masked_input, []

    # ── Step 1: prefill [prefix + segments[0] + '#'] ────────────────────────
    prefix_str  = _build_prefix_str(tokenizer, instruction, masked_input)
    initial_str = prefix_str + segments[0] + "#"

    enc = _tokenize_left_truncated(
        tokenizer, initial_str, max_seq_length - 4, device
    )

    logits, kv = _prefill(model, enc["input_ids"], enc["attention_mask"])
    kv_len = enc["input_ids"].shape[1]

    predicted_locals: List[int] = []

    for i in range(n_masks):
        # ── Step 2a: greedy-decode up to 4 digit tokens ─────────────────────
        pred_ids: List[int] = []

        for _ in range(4):
            next_id  = int(logits.argmax(dim=-1).item())
            next_tok = tokenizer.decode([next_id]).strip()

            if not re.match(r"^\d", next_tok):
                break                          # non-digit → stop

            pred_ids.append(next_id)
            logits, kv = _decode_one(model, next_id, kv, kv_len, device)
            kv_len += 1

        # ── Step 2b: parse the decoded cluster number ───────────────────────
        raw      = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        m        = re.match(r"^\d+", raw)
        pred_num = min(int(m.group()), max_cluster_id) if m else 0
        predicted_locals.append(pred_num)

        # ── Step 2c: extend KV with [seg[i+1] + '#'] (batched when possible) ──
        if i + 1 < n_masks:
            chunk     = segments[i + 1] + "#"
            chunk_ids = tokenizer.encode(chunk, add_special_tokens=False)
            logits, kv, kv_len = _extend_kv_chunk(
                model, chunk_ids, kv, kv_len, device
            )

    # ── Reconstruct the output string ────────────────────────────────────────
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

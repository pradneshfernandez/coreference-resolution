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

import json
import re
from typing import Callable, List, Optional, Sequence, Tuple

try:                                    # torch is only needed for real decoding;
    import torch                        # the prompt/postprocessing helpers below
except ImportError:                     # are pure Python and run without it.
    torch = None                        # type: ignore[assignment]


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
            # The failed forward may already have appended keys/values to the
            # cache. Roll it back to kv_len so the single-token path below does
            # not decode against a corrupted (too long) cache.
            crop = getattr(kv, "crop", None)
            if callable(crop):
                try:
                    crop(kv_len)
                except Exception:
                    pass
            # Fall through to single-token path

    # ── Single-token fallback ───────────────────────────────────────────
    logits = None
    for tok_id in chunk_ids:
        logits, kv = _decode_one(model, tok_id, kv, kv_len, device)
        kv_len += 1
    return logits, kv, kv_len


# ---------------------------------------------------------------------------
# Mask handling (pure Python — no torch required)
# ---------------------------------------------------------------------------

def split_masked_input(masked_input: str) -> List[str]:
    """
    Split a masked input on its mask tokens.

    Overt mentions are marked '#MASK', zero mentions '@MASK'; both are split
    points. Returns n_masks + 1 text segments.
    """
    return re.split(r"[#@]MASK", masked_input)


def reconstruct_output(segments: Sequence[str], predicted: Sequence[int]) -> str:
    """Re-join segments, replacing each mask with '#<predicted cluster number>'."""
    parts = [segments[0]]
    for j, num in enumerate(predicted):
        parts.append(f"#{num}")
        parts.append(segments[j + 1])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prefix_str(tokenizer, instruction: str, masked_input: str) -> str:
    """Apply the chat template to produce the constant per-frame prefix."""
    from coref.data.chat_format import build_chat_text

    return build_chat_text(
        tokenizer, instruction, masked_input, add_generation_prompt=True
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
    segments = split_masked_input(masked_input)
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

    return reconstruct_output(segments, predicted_locals), predicted_locals


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
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    def _predict(instruction: str, masked_input: str, n_masks: int) -> List[int]:
        _, predicted = controlled_inference(
            model, tokenizer, instruction, masked_input, device,
            max_cluster_id, max_seq_length,
        )
        return predicted

    return run_inference_with_predictor(examples, _predict, verbose=verbose)


def _unpack_example(ex) -> dict:
    """Normalise a FrameExample or a JSONL record into a plain dict."""
    def _maybe_json(v):
        return json.loads(v) if isinstance(v, str) else v

    if hasattr(ex, "masked_input"):
        return {
            "doc_id":          ex.doc_id,
            "instruction":     ex.instruction,
            "masked_input":    ex.masked_input,
            "before_mentions": ex.before_mentions,
            "after_mentions":  ex.after_mentions,
            "before_si":       ex.before_sent_indices,
            "after_si":        ex.after_sent_indices,
        }
    return {
        "doc_id":          ex["doc_id"],
        "instruction":     ex["instruction"],
        "masked_input":    ex["input"],
        "before_mentions": _maybe_json(ex["before_mentions"]),
        "after_mentions":  _maybe_json(ex["after_mentions"]),
        "before_si":       _maybe_json(ex["before_sent_indices"]),
        "after_si":        _maybe_json(ex["after_sent_indices"]),
    }


def run_inference_with_predictor(
    examples: list,
    predict_fn: Callable[[str, str, int], List[int]],
    verbose: bool = False,
) -> List[dict]:
    """
    Drive the frame loop with an arbitrary predictor.

    *predict_fn* receives (instruction, masked_input, n_masks) and returns one
    predicted local cluster number per mask, in order. This is what lets the
    same postprocessing / evaluation path be exercised by a real model, by a
    heuristic, or by a stub in tests — with no torch required.
    """
    results: List[dict] = []

    for idx, raw in enumerate(examples):
        ex = _unpack_example(raw)

        if verbose and idx % 50 == 0:
            print(f"  Inference [{idx}/{len(examples)}] doc={ex['doc_id']} …")

        segments = split_masked_input(ex["masked_input"])
        n_masks  = len(segments) - 1

        predicted = list(predict_fn(ex["instruction"], ex["masked_input"], n_masks))[:n_masks]

        before_mentions = list(ex["before_mentions"])
        after_mentions  = list(ex["after_mentions"])
        all_mentions    = before_mentions + after_mentions

        # Any mask the model failed to answer for becomes its own singleton,
        # numbered above every cluster it did predict — falling back to the
        # mention's own index would silently merge it with a real cluster.
        next_free = (max(predicted) + 1) if predicted else 0
        while len(predicted) < max(n_masks, len(all_mentions)):
            predicted.append(next_free)
            next_free += 1

        annotated = []
        for k, mention in enumerate(all_mentions):
            m_dict = dict(mention)
            m_dict["predicted_local_no"] = predicted[k]
            annotated.append(m_dict)

        n_before = len(before_mentions)
        results.append({
            "doc_id":              ex["doc_id"],
            "before_sent_indices": ex["before_si"],
            "after_sent_indices":  ex["after_si"],
            "before_mentions":     annotated[:n_before],
            "after_mentions":      annotated[n_before:],
            "output_text":         reconstruct_output(segments, predicted[:n_masks]),
        })

    return results

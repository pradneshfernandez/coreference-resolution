"""
inference.py — Controlled inference engine for CorefInst.

The key idea (Section 3.3 of the paper):
  • The model's input contains multiple #MASK tokens — one per mention.
  • Instead of generating the entire output in one shot, we predict each
    #MASK token's cluster number sequentially, feeding every previous
    prediction back into the context before predicting the next one.
  • This forces the model to rely on its own earlier decisions (autoregressive
    consistency) and avoids random cluster-number fluctuations.

Implementation:
  1. Split the masked input on '#MASK' → list of text segments.
  2. For each MASK position i:
       a. Build the partial assistant output: seg_0 + '#' + n_0 + seg_1 + '#' + n_1 + … + seg_i + '#'
       b. Format as a complete prompt (system+user+partial_assistant).
       c. Run model.generate() with max_new_tokens=4; extract the leading integer.
       d. Append the predicted cluster number.
  3. Reconstruct the full output string and return predicted (mention, local_no) pairs.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_partial_prompt(
    tokenizer,
    instruction: str,
    masked_input: str,
    partial_output: str,
) -> str:
    """
    Construct the prompt for predicting the next cluster number.

    We use the model's chat template with add_generation_prompt=True to place
    the partial assistant output at the end (so the model continues from there).
    """
    # Build the conversation without the final assistant turn
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": masked_input},
    ]
    # Apply template up to the start of the assistant turn
    prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # appends the assistant header
    )
    # Append the partial output produced so far
    return prefix + partial_output


# ---------------------------------------------------------------------------
# Number prediction
# ---------------------------------------------------------------------------

def _predict_cluster_number(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_cluster_id: int = 200,
    max_seq_length: int = 4096,
) -> int:
    """
    Given a prompt that ends with '#', predict the next integer (cluster number).

    Strategy: generate up to 4 new tokens greedily, decode them, extract the
    leading digit sequence.
    """
    # Truncate from the LEFT so we preserve the most recent context (the partial
    # output + current MASK position). tokenizer.model_max_length is often 131072
    # for Llama 3.1 — we must use max_seq_length (4096) explicitly.
    max_input_len = max_seq_length - 4   # leave room for 4 new tokens
    orig_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
    ).to(device)

    tokenizer.truncation_side = orig_truncation_side  # restore

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(new_token_ids, skip_special_tokens=True)

    m = re.match(r"^\s*(\d+)", generated)
    if m:
        num = int(m.group(1))
        return min(num, max_cluster_id)
    return 0    # default if no digit found


# ---------------------------------------------------------------------------
# Controlled inference for one example
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
    Run the controlled inference procedure on one masked input.

    Args:
        model         — the fine-tuned causal LM
        tokenizer     — matching tokenizer
        instruction   — the instruction string (system message)
        masked_input  — frame pair text with '#MASK' placeholders
        device        — torch device
        max_cluster_id — upper bound on cluster numbers (for sanity clamping)

    Returns:
        output_text      — masked_input with #MASK replaced by predicted numbers
        predicted_locals — list of predicted local cluster numbers, in order of mention
    """
    # Split on #MASK or @MASK to get text segments between MASK positions
    segments = re.split(r"[#@]MASK", masked_input)
    n_masks = len(segments) - 1

    if n_masks == 0:
        # No mentions in this example
        return masked_input, []

    predicted_locals: List[int] = []
    partial_output = segments[0]    # starts with the text before the first MASK

    for i in range(n_masks):
        # The partial output ends with segment[i] and we need to predict
        # the cluster number that comes after '#'
        prompt = _build_partial_prompt(
            tokenizer,
            instruction,
            masked_input,
            partial_output + "#",
        )

        pred_num = _predict_cluster_number(
            model, tokenizer, prompt, device, max_cluster_id, max_seq_length
        )
        predicted_locals.append(pred_num)

        # Extend partial output with prediction and the next segment
        partial_output = partial_output + f"#{pred_num}" + segments[i + 1]

    return partial_output, predicted_locals


# ---------------------------------------------------------------------------
# Batch inference over a list of FrameExamples
# ---------------------------------------------------------------------------

def run_inference_on_examples(
    model,
    tokenizer,
    examples: list,                 # list of FrameExample (or dicts with same fields)
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

    Returns a list of result dicts:
    {
        'doc_id':            str,
        'before_sent_indices': List[int],
        'after_sent_indices':  List[int],
        'before_mentions':   List[dict],  # with 'predicted_local_no' added
        'after_mentions':    List[dict],  # with 'predicted_local_no' added
        'output_text':       str,
    }
    """
    import json

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    results = []

    for idx, ex in enumerate(examples):
        # Support both FrameExample objects and plain dicts
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
            max_cluster_id, max_seq_length
        )

        # Distribute predicted local numbers back to before / after mentions
        all_mentions = list(before_mentions) + list(after_mentions)
        for i, m in enumerate(all_mentions):
            if i < len(predicted_locals):
                m = dict(m)
                m["predicted_local_no"] = predicted_locals[i]
            else:
                m = dict(m)
                m["predicted_local_no"] = i    # fallback: unique singleton

        n_before = len(before_mentions)
        before_with_pred = all_mentions[:n_before]
        after_with_pred  = all_mentions[n_before:]

        results.append(
            {
                "doc_id":              doc_id,
                "before_sent_indices": before_si,
                "after_sent_indices":  after_si,
                "before_mentions":     before_with_pred,
                "after_mentions":      after_with_pred,
                "output_text":         output_text,
            }
        )

    return results

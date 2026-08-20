"""
collator.py — Completion-only loss masking that does not depend on TRL's version.

CorefInst trains the model to emit cluster numbers *given* the instruction and
the masked text. Computing loss over the prompt as well teaches it to
regenerate the instruction, which is not the task and wastes capacity, so every
token before the assistant header must be excluded from the loss.

TRL used to ship `DataCollatorForCompletionOnlyLM` for this, but the class moved
between modules several times and recent releases dropped it entirely. Rather
than let a version bump silently turn on full-sequence loss, this module
implements the masking directly.

The token-level logic lives in `mask_prompt_labels`, which is pure Python — no
torch, no tokenizer — so the unit tests can cover it on any machine.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Pure-Python core
# ---------------------------------------------------------------------------

def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> Optional[int]:
    """
    Index of the first element *after* the last occurrence of *needle*.

    The last occurrence is what matters: the response template appears once per
    conversation, but a stray copy inside the user text would otherwise cut the
    mask short. Returns None when the needle is absent.
    """
    if not needle or len(needle) > len(haystack):
        return None

    found = None
    for i in range(len(haystack) - len(needle) + 1):
        if list(haystack[i:i + len(needle)]) == list(needle):
            found = i + len(needle)
    return found


def mask_prompt_labels(
    input_ids: Sequence[int],
    template_ids: Sequence[int],
    pad_token_id: Optional[int] = None,
    ignore_index: int = IGNORE_INDEX,
) -> List[int]:
    """
    Build the label row for one example: prompt tokens masked, answer kept.

    Everything up to and including the response template becomes *ignore_index*,
    as does any padding. If the template is not found the whole row is masked —
    a truncated example whose assistant turn was cut off contributes no
    supervision, which is correct and better than training on the prompt.
    """
    labels = list(input_ids)
    start = find_subsequence(input_ids, template_ids)

    if start is None:
        return [ignore_index] * len(labels)

    for i in range(start):
        labels[i] = ignore_index

    if pad_token_id is not None:
        for i in range(start, len(labels)):
            if labels[i] == pad_token_id:
                labels[i] = ignore_index

    return labels


# ---------------------------------------------------------------------------
# Torch collator
# ---------------------------------------------------------------------------

class CompletionOnlyCollator:
    """
    Pad a batch of tokenised examples and mask prompt tokens out of the loss.

    Drop-in replacement for TRL's `DataCollatorForCompletionOnlyLM`, accepting
    the same two arguments so the call site does not have to branch.
    """

    def __init__(self, response_template: str, tokenizer, ignore_index: int = IGNORE_INDEX):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.response_template = response_template
        # The template must be encoded without a BOS, or it will never match
        # mid-sequence.
        self.template_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )
        if not self.template_ids:
            raise ValueError(
                f"Response template {response_template!r} encoded to no tokens."
            )

    def __call__(self, features: List[dict]) -> dict:
        import torch

        # SFTTrainer hands over already-tokenised rows; tolerate raw text too.
        if "input_ids" not in features[0]:
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )
        else:
            batch = self.tokenizer.pad(
                [{"input_ids": f["input_ids"],
                  "attention_mask": f.get(
                      "attention_mask", [1] * len(f["input_ids"]))}
                 for f in features],
                padding=True,
                return_tensors="pt",
            )

        pad_id = self.tokenizer.pad_token_id
        labels = [
            mask_prompt_labels(
                row.tolist(), self.template_ids, pad_id, self.ignore_index
            )
            for row in batch["input_ids"]
        ]
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def count_supervised_tokens(labels: Sequence[int], ignore_index: int = IGNORE_INDEX) -> int:
    """How many tokens in a label row actually contribute to the loss."""
    return sum(1 for x in labels if x != ignore_index)

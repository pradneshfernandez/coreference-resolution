"""
chat_format.py — Apply a tokenizer's chat template consistently everywhere.

Training (dataset_builder) and inference (eval.inference) must format prompts
identically or the fine-tuned model sees a different prefix at test time than it
saw during training. Both therefore go through `build_chat_text` here.

Not every chat template accepts a 'system' role (Gemma's, for one, raises).
When that happens the instruction is prepended to the user turn instead, which
is what those templates expect anyway.
"""

from __future__ import annotations

from typing import List, Optional


def build_chat_text(
    tokenizer,
    instruction: str,
    user_content: str,
    assistant_content: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Render one conversation to the string the model is trained/prompted on.

    Args:
        instruction           — system-level instruction
        user_content          — the masked input
        assistant_content     — the target output (training) or None (inference)
        add_generation_prompt — append the assistant header (inference only)
    """
    def _render(messages: List[dict]) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    messages: List[dict] = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": user_content},
    ]
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})

    try:
        return _render(messages)
    except Exception:
        # Template rejects the system role — fold it into the user turn.
        merged = [{"role": "user", "content": f"{instruction}\n\n{user_content}"}]
        if assistant_content is not None:
            merged.append({"role": "assistant", "content": assistant_content})
        return _render(merged)

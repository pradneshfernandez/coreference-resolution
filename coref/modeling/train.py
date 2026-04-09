"""
train.py — Instruction fine-tune a causal LM on CorefInst examples using SFT + LoRA.

Training loop:
  1. Load processed JSONL dataset (created by prepare_data.py).
  2. Apply the model's chat template to format each example as a single string.
  3. Fine-tune with HuggingFace TRL's SFTTrainer; loss is computed only on the
     assistant (output) tokens.
  4. Save the LoRA adapter checkpoint.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from datasets import Dataset
from transformers import TrainingArguments

from coref.data.dataset_builder import format_for_sft, load_jsonl
from coref.modeling.model import load_model_and_tokenizer


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    train_path: str,
    dev_path: Optional[str] = None,
    output_dir: str = "model_output",
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[list] = None,
    load_in_4bit: bool = True,
    num_epochs: int = 3,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    lr_scheduler: str = "cosine",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    logging_steps: int = 10,
    save_steps: int = 200,
    eval_steps: int = 200,
    seed: int = 42,
    bf16: bool = True,
    fp16: bool = False,
    dataloader_workers: int = 0,
    backend: str = "auto",
) -> None:
    """
    Fine-tune a causal LM on CorefInst examples.

    Args:
        train_path   — path to train.jsonl (from prepare_data.py)
        dev_path     — optional path to dev.jsonl for eval during training
        output_dir   — directory to save checkpoints and final model
        model_name   — HuggingFace model ID or local path
        …            — remaining kwargs map to TrainingArguments / LoRA config
    """
    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print(f"Loading training data from {train_path} …")
    train_dataset = load_jsonl(train_path)
    print(f"  {len(train_dataset)} training examples")

    eval_dataset: Optional[Dataset] = None
    if dev_path and os.path.exists(dev_path):
        eval_dataset = load_jsonl(dev_path)
        print(f"  {len(eval_dataset)} dev examples")

    # ------------------------------------------------------------------
    # 2. Load model + tokenizer
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        load_in_4bit=load_in_4bit,
        backend=backend,
    )

    # ------------------------------------------------------------------
    # 3. Format examples with the chat template
    # ------------------------------------------------------------------
    def _format(batch):
        texts = [
            format_for_sft(
                {"instruction": inst, "input": inp, "output": out},
                tokenizer,
            )
            for inst, inp, out in zip(
                batch["instruction"], batch["input"], batch["output"]
            )
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(_format, batched=True, remove_columns=train_dataset.column_names)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(_format, batched=True, remove_columns=eval_dataset.column_names)

    # ------------------------------------------------------------------
    # 4. TrainingArguments
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # Allow CUDA memory allocator to use expandable segments — reduces OOM fragmentation.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    # Unsloth sets up gradient checkpointing on the model itself via
    # use_gradient_checkpointing="unsloth" in get_peft_model — don't duplicate it
    # in TrainingArguments or the two implementations conflict.
    _use_unsloth = False
    try:
        import unsloth  # type: ignore  # noqa: F401
        _use_unsloth = True
    except ImportError:
        pass

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=not _use_unsloth,   # unsloth handles its own GC
        gradient_checkpointing_kwargs={"use_reentrant": False} if not _use_unsloth else {},
        learning_rate=learning_rate,
        warmup_steps=int(warmup_ratio * (len(train_dataset) // (per_device_batch_size * gradient_accumulation_steps)) * num_epochs),
        lr_scheduler_type=lr_scheduler,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        load_best_model_at_end=eval_dataset is not None,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=dataloader_workers,
        seed=seed,
        report_to="none",
        optim="adamw_8bit" if load_in_4bit else "adamw_torch",
    )

    # ------------------------------------------------------------------
    # 5. SFTTrainer
    # ------------------------------------------------------------------
    try:
        from trl import SFTTrainer  # type: ignore

        # DataCollatorForCompletionOnlyLM moved across TRL versions — try all locations.
        DataCollatorForCompletionOnlyLM = None
        for _mod, _cls in [
            ("trl", "DataCollatorForCompletionOnlyLM"),
            ("trl.trainer", "DataCollatorForCompletionOnlyLM"),
            ("trl.trainer.utils", "DataCollatorForCompletionOnlyLM"),
            ("trl.data_utils", "DataCollatorForCompletionOnlyLM"),
        ]:
            try:
                import importlib
                _m = importlib.import_module(_mod)
                DataCollatorForCompletionOnlyLM = getattr(_m, _cls, None)
                if DataCollatorForCompletionOnlyLM is not None:
                    break
            except ImportError:
                pass

        # Identify the response template so we only compute loss on output tokens.
        response_template = _find_response_template(tokenizer)

        if response_template and DataCollatorForCompletionOnlyLM is not None:
            collator = DataCollatorForCompletionOnlyLM(
                response_template=response_template,
                tokenizer=tokenizer,
            )
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                data_collator=collator,
                args=training_args,
            )
        else:
            if DataCollatorForCompletionOnlyLM is None:
                print("[warn] DataCollatorForCompletionOnlyLM not found — computing loss on all tokens.")
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=training_args,
            )

    except ImportError:
        raise ImportError(
            "trl is required for training. Install with: pip install trl"
        )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\nStarting training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_response_template(tokenizer) -> Optional[str]:
    """
    Return the token sequence that marks the start of the assistant response,
    so DataCollatorForCompletionOnlyLM can mask prompt tokens from the loss.
    """
    # Common patterns for popular models
    candidates = [
        "<|start_header_id|>assistant<|end_header_id|>\n\n",   # Llama 3 / 3.1
        "[/INST]",                                               # Mistral / Llama 2
        "<start_of_turn>model\n",                               # Gemma
        "### Response:\n",                                      # generic alpaca
        "<|im_start|>assistant\n",                              # ChatML / Qwen
    ]
    # Test which template produces tokens present in a dummy formatted string
    dummy = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": "hello"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    for cand in candidates:
        if cand in dummy:
            return cand
    return None

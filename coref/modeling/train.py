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

import inspect
import math
import os
from typing import Optional

from datasets import Dataset

from coref.data.dataset_builder import format_for_sft, load_jsonl
from coref.modeling.model import (cuda_available, load_model_and_tokenizer,
                                  select_backend)


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
    eval_max_examples: int = 300,
    seed: int = 42,
    bf16: bool = True,
    fp16: bool = False,
    dataloader_workers: int = 0,
    backend: str = "auto",
    resume: bool = True,
) -> None:
    """
    Fine-tune a causal LM on CorefInst examples.

    Args:
        train_path   — path to train.jsonl (from prepare_data.py)
        dev_path     — optional path to dev.jsonl for eval during training
        output_dir   — directory to save checkpoints and final model
        model_name   — HuggingFace model ID or local path
        resume       — continue from the newest checkpoint in output_dir if any
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
        # Mid-training eval runs every eval_steps and its only job is to show
        # whether the loss is still falling. Scoring the whole dev split for
        # that costs GPU hours the run cannot spare, so take a fixed random
        # subsample (seeded, hence identical at every eval).
        if eval_max_examples and len(eval_dataset) > eval_max_examples:
            eval_dataset = eval_dataset.shuffle(seed=seed).select(
                range(eval_max_examples)
            )
            print(f"  using a {len(eval_dataset)}-example dev subsample for "
                  f"mid-training eval (set training.eval_max_examples: 0 for all)")

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
    # Ask the same helper load_model_and_tokenizer used, so this cannot
    # disagree with which backend actually loaded the model.
    _use_unsloth = select_backend(backend)

    on_cuda = cuda_available()
    if not on_cuda and (bf16 or fp16):
        print("[info] No CUDA device — training in fp32 (bf16/fp16 disabled).")
        bf16 = fp16 = False

    # adamw_8bit needs bitsandbytes + CUDA.
    optim = "adamw_8bit" if (load_in_4bit and on_cuda) else "adamw_torch"

    # load_best_model_at_end requires the save and eval schedules to line up,
    # and save_steps to be a whole multiple of eval_steps — Trainer raises
    # otherwise. Only enable it when that actually holds.
    do_eval = eval_dataset is not None
    can_load_best = do_eval and eval_steps > 0 and save_steps % eval_steps == 0
    if do_eval and not can_load_best:
        print(f"[warn] save_steps={save_steps} is not a multiple of "
              f"eval_steps={eval_steps} — not loading the best checkpoint at end.")

    # transformers 5 dropped warmup_ratio and keeps only warmup_steps, so the
    # ratio is converted here and both are offered; _build_training_args keeps
    # whichever the installed class accepts. Without this the configured warmup
    # is silently discarded and the run starts at full learning rate.
    effective_batch = max(per_device_batch_size * gradient_accumulation_steps, 1)
    total_steps = math.ceil(len(train_dataset) / effective_batch) * max(num_epochs, 1)
    warmup_steps = max(1, round(warmup_ratio * total_steps)) if warmup_ratio else 0
    print(f"[info] {total_steps} optimizer steps planned; warmup {warmup_steps} "
          f"step(s) ({warmup_ratio:.1%})")

    training_args = _build_training_args(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=not _use_unsloth,   # unsloth handles its own GC
        gradient_checkpointing_kwargs={"use_reentrant": False} if not _use_unsloth else {},
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps" if do_eval else "no",
        eval_steps=eval_steps if do_eval else None,
        save_total_limit=3,
        load_best_model_at_end=can_load_best,
        bf16=bf16,
        fp16=fp16,
        dataloader_num_workers=dataloader_workers,
        seed=seed,
        report_to="none",
        optim=optim,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    # ------------------------------------------------------------------
    # 5. SFTTrainer
    # ------------------------------------------------------------------
    try:
        from trl import SFTTrainer  # type: ignore
    except ImportError:
        raise ImportError(
            "trl is required for training. Install with: pip install trl"
        )

    # Identify the response template so we only compute loss on output tokens.
    response_template = _find_response_template(tokenizer)

    trainer_kwargs = dict(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # TRL renamed `tokenizer` → `processing_class` in 0.12.
    _sft_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in _sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    # Old TRL took these on the trainer; new TRL takes them on SFTConfig,
    # where _build_training_args has already set them.
    for key, value in (("dataset_text_field", "text"),
                       ("max_seq_length", max_seq_length)):
        if key in _sft_params:
            trainer_kwargs[key] = value

    # Loss must be computed on the assistant answer only. TRL's own collator for
    # this moved modules repeatedly and newer releases removed it, so the local
    # implementation in coref.modeling.collator is used instead — same masking,
    # no version coupling.
    if not response_template:
        raise RuntimeError(
            "Could not identify the assistant response template for "
            f"{model_name!r}. Without it, loss would be computed over the "
            "prompt as well, which does not train the CorefInst task. Add the "
            "model's template to _find_response_template() in "
            "coref/modeling/train.py before training."
        )

    from coref.modeling.collator import CompletionOnlyCollator

    trainer_kwargs["data_collator"] = CompletionOnlyCollator(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    print(f"[info] Completion-only loss enabled (template={response_template!r}).")

    trainer = SFTTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    # Colab sessions time out well before a full run finishes, so pick up from
    # the newest checkpoint in output_dir if one is there. Without this, every
    # reconnect silently restarts from step 0 and the run never completes.
    resume_from = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint
        try:
            resume_from = get_last_checkpoint(output_dir)
        except (FileNotFoundError, OSError):
            resume_from = None

    if resume_from:
        print(f"\nResuming training from {resume_from} …")
    else:
        print("\nStarting training from scratch …")

    trainer.train(resume_from_checkpoint=resume_from)

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    final_path = os.path.join(output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_training_args(**kwargs):
    """
    Build the trainer config in a way that survives TRL/transformers churn.

    • TRL >= 0.12 wants an SFTConfig that also carries `max_seq_length` and
      `dataset_text_field`; older TRL wants a plain TrainingArguments and takes
      those two on the trainer instead.
    • transformers renamed `evaluation_strategy` → `eval_strategy` in 4.41.
    • transformers 5 removed `warmup_ratio`, leaving only `warmup_steps`.

    Keywords the installed classes do not accept are dropped rather than
    raising, so a version bump degrades instead of crashing at startup. Because
    a dropped keyword is a training hyperparameter that silently stops applying,
    anything discarded is reported as a warning, not a note.
    """
    from transformers import TrainingArguments

    cls = TrainingArguments
    try:
        from trl import SFTConfig  # type: ignore
        cls = SFTConfig
    except ImportError:
        pass

    params = inspect.signature(cls.__init__).parameters

    # Renames across versions: keep the value, change the key.
    for old, new in (("eval_strategy", "evaluation_strategy"),   # transformers < 4.41
                     ("max_seq_length", "max_length")):          # trl >= 0.20
        if old in kwargs and old not in params and new in params:
            kwargs[new] = kwargs.pop(old)

    # Equivalent pairs: the caller supplies both spellings of the same setting
    # and exactly one survives, so dropping the other is not a loss.
    equivalents = {"warmup_ratio": "warmup_steps", "warmup_steps": "warmup_ratio"}
    benign = {
        key for key, twin in equivalents.items()
        if key not in params and twin in params and twin in kwargs
    }
    for key in benign:
        kwargs.pop(key, None)

    dropped = [k for k in kwargs if k not in params]
    for k in dropped:
        kwargs.pop(k)
    if dropped:
        print(f"[warn] {cls.__name__} does not accept {dropped} — these settings "
              "will NOT be applied to this run.")

    return cls(**kwargs)


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
    try:
        dummy = tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"},
             {"role": "assistant", "content": "hello"}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        print(f"[warn] tokenizer has no usable chat template ({exc}).")
        return None

    for cand in candidates:
        if cand in dummy:
            return cand
    return None

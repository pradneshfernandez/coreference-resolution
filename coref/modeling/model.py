"""
model.py — Load an LLM with optional 4-bit quantization and configure LoRA adapters.

Supports two backends:
  • unsloth  — faster, memory-efficient, quantization-aware LoRA (preferred)
  • standard — HuggingFace transformers + PEFT + bitsandbytes (fallback)

The backend is selected automatically: if `unsloth` is importable it is used;
otherwise the standard stack is used.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Default LoRA target modules for common model families
# ---------------------------------------------------------------------------

_LORA_TARGETS_LLAMA = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
_LORA_TARGETS_GEMMA = _LORA_TARGETS_LLAMA
_LORA_TARGETS_MISTRAL = _LORA_TARGETS_LLAMA


def _infer_target_modules(model_name: str) -> List[str]:
    name = model_name.lower()
    if "gemma" in name:
        return _LORA_TARGETS_GEMMA
    if "mistral" in name:
        return _LORA_TARGETS_MISTRAL
    return _LORA_TARGETS_LLAMA   # Llama default


# ---------------------------------------------------------------------------
# Unsloth backend
# ---------------------------------------------------------------------------

def _load_unsloth(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    load_in_4bit: bool,
):
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,            # auto-detect
        load_in_4bit=load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Standard HF + PEFT backend
# ---------------------------------------------------------------------------

def _load_standard(
    model_name: str,
    max_seq_length: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    load_in_4bit: bool,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"    # causal LM padding

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if _flash_attn_available() else "eager",
    )

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def _flash_attn_available() -> bool:
    try:
        import flash_attn  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    load_in_4bit: bool = True,
    backend: str = "auto",          # 'auto' | 'unsloth' | 'standard'
) -> Tuple:
    """
    Load a causal LM with LoRA adapters.

    Args:
        model_name     — HuggingFace model ID (local path also supported)
        max_seq_length — maximum sequence length for the model
        lora_rank      — LoRA rank (R in the paper = 16)
        lora_alpha     — LoRA alpha (usually equal to rank)
        lora_dropout   — LoRA dropout
        target_modules — list of module names to apply LoRA to;
                         if None, auto-detected from model name
        load_in_4bit   — use 4-bit quantization (QLoRA)
        backend        — 'auto', 'unsloth', or 'standard'

    Returns:
        (model, tokenizer)
    """
    if target_modules is None:
        target_modules = _infer_target_modules(model_name)

    use_unsloth = False
    if backend == "auto":
        try:
            import unsloth  # type: ignore  # noqa: F401
            use_unsloth = True
        except ImportError:
            use_unsloth = False
    elif backend == "unsloth":
        use_unsloth = True

    print(f"Backend: {'unsloth' if use_unsloth else 'standard HF+PEFT'}")
    print(f"Model  : {model_name}")
    print(f"LoRA   : rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}")

    if use_unsloth:
        return _load_unsloth(
            model_name, max_seq_length, lora_rank, lora_alpha,
            lora_dropout, target_modules, load_in_4bit,
        )
    else:
        return _load_standard(
            model_name, max_seq_length, lora_rank, lora_alpha,
            lora_dropout, target_modules, load_in_4bit,
        )


def load_for_inference(
    checkpoint_path: str,
    base_model_name: Optional[str] = None,
    max_seq_length: int = 4096,
    load_in_4bit: bool = True,
    backend: str = "auto",
) -> Tuple:
    """
    Load a fine-tuned model (LoRA adapter merged or from checkpoint directory).

    If *base_model_name* is given, loads base + adapter separately via PEFT.
    Otherwise loads a fully-merged checkpoint.
    """
    use_unsloth = False
    if backend == "auto":
        try:
            import unsloth  # type: ignore  # noqa: F401
            use_unsloth = True
        except ImportError:
            pass
    elif backend == "unsloth":
        use_unsloth = True

    if use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore
        import json as _json

        # Detect whether checkpoint_path is a LoRA adapter directory or a full model.
        # A LoRA adapter directory has adapter_config.json; a full model has config.json
        # with a non-adapter architecture.
        adapter_cfg = os.path.join(checkpoint_path, "adapter_config.json")
        is_adapter = os.path.isfile(adapter_cfg)

        if is_adapter:
            # Use unsloth's native adapter loading.
            # Passing the adapter directory directly lets unsloth read
            # adapter_config.json, load the base model internally, and apply
            # the LoRA weights — without going through PeftModel, which would
            # wrap the model and break unsloth's fast-inference kernel patch.
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )
            # Override tokenizer with the one saved alongside the adapter
            # (handles any special tokens added during fine-tuning)
            from transformers import AutoTokenizer  # type: ignore
            _saved_tok = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            if _saved_tok.pad_token is None:
                _saved_tok.pad_token = _saved_tok.eos_token
            tokenizer = _saved_tok
        else:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=max_seq_length,
                dtype=None,
                load_in_4bit=load_in_4bit,
            )

        FastLanguageModel.for_inference(model)
        # Remove max_length from the stored generation_config so that
        # calling generate(max_new_tokens=4) does not trigger a conflict
        # warning between max_new_tokens and the stored max_length=131072.
        if hasattr(model, "generation_config"):
            model.generation_config.max_length = None
        return model, tokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        if base_model_name:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(base, checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

        model.eval()
        return model, tokenizer

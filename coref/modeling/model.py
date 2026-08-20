"""
model.py — Load an LLM with optional 4-bit quantization and configure LoRA adapters.

Supports two backends:
  • unsloth  — faster, memory-efficient, quantization-aware LoRA (preferred)
  • standard — HuggingFace transformers + PEFT + bitsandbytes (fallback)

The backend is selected automatically: unsloth is used when it is importable
AND a CUDA device is present; otherwise the standard stack is used. On CPU,
4-bit quantization and flash-attention are also switched off, since both are
CUDA-only — so the same config file works on a laptop and on a GPU.
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


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _resolve_device_settings(load_in_4bit: bool) -> Tuple[bool, dict]:
    """
    Reconcile the requested precision with the hardware actually present.

    bitsandbytes 4-bit quantisation and device_map='auto' offloading both
    require CUDA; on a CPU-only box they either raise or silently produce a
    model that cannot run. Fall back to plain fp32 on CPU.
    """
    if cuda_available():
        return load_in_4bit, {"device_map": "auto", "torch_dtype": torch.bfloat16}

    if load_in_4bit:
        print("[info] No CUDA device found — disabling 4-bit quantization "
              "and loading in fp32 on CPU.")
    return False, {"device_map": None, "torch_dtype": torch.float32}


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

    load_in_4bit, device_kwargs = _resolve_device_settings(load_in_4bit)

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
        trust_remote_code=True,
        # flash-attn is CUDA-only; 'eager' is the portable fallback.
        attn_implementation=(
            "flash_attention_2"
            if (cuda_available() and _flash_attn_available())
            else "eager"
        ),
        **device_kwargs,
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


def select_backend(backend: str) -> bool:
    """Decide whether to use unsloth. unsloth requires CUDA, so 'auto' never
    picks it on a CPU-only machine."""
    if backend == "unsloth":
        return True
    if backend == "standard":
        return False
    if not cuda_available():
        return False
    try:
        import unsloth  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


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

    use_unsloth = select_backend(backend)

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
    use_unsloth = select_backend(backend)

    if use_unsloth:
        from unsloth import FastLanguageModel  # type: ignore

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

        load_in_4bit, device_kwargs = _resolve_device_settings(load_in_4bit)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # A LoRA checkpoint has no weights of its own — the base model it was
        # trained on must be loaded first. Read it from adapter_config.json when
        # the caller did not pass one explicitly.
        if base_model_name is None:
            base_model_name = _base_model_from_adapter(checkpoint_path)

        if base_model_name:
            base = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                **device_kwargs,
            )
            model = PeftModel.from_pretrained(base, checkpoint_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                quantization_config=bnb_config,
                trust_remote_code=True,
                **device_kwargs,
            )

        model.eval()
        return model, tokenizer


def _base_model_from_adapter(checkpoint_path: str) -> Optional[str]:
    """Return the base model recorded in a PEFT adapter_config.json, if any."""
    import json

    cfg_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("base_model_name_or_path")
    except (OSError, ValueError):
        return None

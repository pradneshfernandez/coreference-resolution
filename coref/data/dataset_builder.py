"""
dataset_builder.py — Load TransMuCoRes data for Hindi, Tamil, Bengali and
build HuggingFace Dataset objects ready for instruction fine-tuning.

TransMuCoRes directory layout
------------------------------
transmucores_data/
├── mujadia_conll/          (Hindi only)
│   ├── train/
│   ├── development/
│   └── test/
├── onto_notes_archive/     (multilingual, nested)
│   ├── train/data/english/annotations/wb/{a2e,c2e,eng}/00/*.conll
│   ├── development/...
│   └── test/...
├── litbank_train/          (flat, multilingual)
├── litbank_val/
└── litbank_test/

Language filter codes
---------------------
  Hindi   → 'hin_Deva'  (mujadia: no filter needed — all files are Hindi)
  Tamil   → 'tam_Taml'
  Bengali → 'ben_Beng'
"""

import json
import os
from typing import Dict, List, Optional

from datasets import Dataset

from coref.data.conll_parser import Document, load_conll_dir
from coref.data.preprocessor import FrameExample, create_frame_examples


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, dict] = {
    "hi": {"name": "Hindi",   "codes": ["hin_Deva"]},
    "ta": {"name": "Tamil",   "codes": ["tam_Taml"]},
    "bn": {"name": "Bengali", "codes": ["ben_Beng"]},
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_mujadia(root: str, split: str) -> List[Document]:
    """Load mujadia_conll (all Hindi, no language filter needed)."""
    split_map = {"train": "train", "dev": "development", "test": "test"}
    dname = split_map.get(split, split)
    data_dir = os.path.join(root, "mujadia_conll", dname)
    return load_conll_dir(data_dir, language_filter=None, language="hi")


def _load_onto_notes(root: str, split: str, lang_code: str, lang: str) -> List[Document]:
    """Load onto_notes_archive for one language (recursive search)."""
    split_map = {"train": "train", "dev": "development", "test": "test"}
    dname = split_map.get(split, split)
    data_dir = os.path.join(root, "onto_notes_archive", dname)
    return load_conll_dir(
        data_dir, language_filter=[lang_code], language=lang, recursive=True
    )


def _load_litbank(root: str, split: str, lang_code: str, lang: str) -> List[Document]:
    """Load litbank_{train,val,test} for one language (flat directory)."""
    dir_map = {"train": "litbank_train", "dev": "litbank_val", "test": "litbank_test"}
    dname = dir_map.get(split, f"litbank_{split}")
    data_dir = os.path.join(root, dname)
    return load_conll_dir(data_dir, language_filter=[lang_code], language=lang)


def load_documents(
    data_root: str,
    split: str,
    languages: Optional[List[str]] = None,
) -> List[Document]:
    """
    Load all Documents for the given split and languages.

    Args:
        data_root — path to the transmucores_data/ directory
        split     — 'train', 'dev', or 'test'
        languages — list of language codes from LANGUAGE_CONFIGS
                    (default: all three — hi, ta, bn)
    """
    if languages is None:
        languages = list(LANGUAGE_CONFIGS.keys())

    docs: List[Document] = []

    for lang in languages:
        cfg = LANGUAGE_CONFIGS.get(lang)
        if cfg is None:
            print(f"[warn] Unknown language '{lang}', skipping.")
            continue
        lang_codes = cfg["codes"]
        lang_name = cfg["name"]

        # Hindi → also load mujadia_conll
        if lang == "hi":
            mujadia_docs = _load_mujadia(data_root, split)
            print(f"  mujadia ({split}): {len(mujadia_docs)} Hindi docs")
            docs.extend(mujadia_docs)

        for code in lang_codes:
            on_docs = _load_onto_notes(data_root, split, code, lang)
            lb_docs = _load_litbank(data_root, split, code, lang)
            print(
                f"  onto_notes ({split}, {code}): {len(on_docs)} docs | "
                f"litbank ({split}, {code}): {len(lb_docs)} docs"
            )
            docs.extend(on_docs)
            docs.extend(lb_docs)

    return docs


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------

def build_examples(
    documents: List[Document],
    instruction_id: int = 5,
    max_tokens_per_frame: int = 256,
) -> List[FrameExample]:
    """Convert a list of Documents into FrameExamples."""
    examples: List[FrameExample] = []
    for doc in documents:
        ex = create_frame_examples(
            doc,
            instruction_id=instruction_id,
            max_tokens_per_frame=max_tokens_per_frame,
        )
        examples.extend(ex)
    return examples


# ---------------------------------------------------------------------------
# HuggingFace Dataset construction
# ---------------------------------------------------------------------------

def _example_to_dict(ex: FrameExample) -> dict:
    """Serialise a FrameExample to a plain dict for the HF Dataset."""
    return {
        "doc_id": ex.doc_id,
        "language": ex.language,
        "instruction": ex.instruction,
        "input": ex.masked_input,
        "output": ex.output,
        "before_mentions": json.dumps(ex.before_mentions),
        "after_mentions": json.dumps(ex.after_mentions),
        "before_sent_indices": json.dumps(ex.before_sent_indices),
        "after_sent_indices": json.dumps(ex.after_sent_indices),
    }


def examples_to_hf_dataset(examples: List[FrameExample]) -> Dataset:
    """Create a HuggingFace Dataset from a list of FrameExamples."""
    records = [_example_to_dict(ex) for ex in examples]
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Save / load processed data as JSONL
# ---------------------------------------------------------------------------

def save_jsonl(examples: List[FrameExample], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(_example_to_dict(ex), ensure_ascii=False) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def load_jsonl(path: str) -> Dataset:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Formatting for SFT
# ---------------------------------------------------------------------------

def format_for_sft(
    example: dict,
    tokenizer,
    add_eos: bool = True,
) -> str:
    """
    Apply the model's chat template to produce the full training string.

    System  = instruction
    User    = masked_input
    Assistant = output
    """
    messages = [
        {"role": "system",    "content": example["instruction"]},
        {"role": "user",      "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if add_eos and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text

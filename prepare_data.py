"""
prepare_data.py — Pre-process TransMuCoRes data into JSONL files for training.

Usage:
  python prepare_data.py [--config config.yaml]

Reads transmucores_data/, generates FrameExamples, saves:
  processed_data/train.jsonl
  processed_data/dev.jsonl
  processed_data/test.jsonl

Each line in the JSONL is one training example (one frame pair).
"""

import argparse
import os
import sys

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)

    data_root  = cfg["data"]["root"]
    output_dir = cfg["data"]["output_dir"]
    languages  = list(cfg["data"]["languages"].keys())   # ['hi', 'ta', 'bn']

    instr_id    = cfg["preprocessing"]["instruction_id"]
    max_tokens  = cfg["preprocessing"]["max_tokens_per_frame"]

    os.makedirs(output_dir, exist_ok=True)

    # Lazy import (avoids loading torch just to preprocess)
    from src.dataset_builder import load_documents, build_examples, save_jsonl

    for split, out_name in [("train", "train"), ("dev", "dev"), ("test", "test")]:
        print(f"\n{'='*55}")
        print(f"  Loading split: {split}")
        print(f"{'='*55}")

        docs = load_documents(data_root, split=split, languages=languages)
        print(f"  Total documents loaded: {len(docs)}")

        examples = build_examples(docs, instruction_id=instr_id, max_tokens_per_frame=max_tokens)
        print(f"  Total frame examples: {len(examples)}")

        out_path = os.path.join(output_dir, f"{out_name}.jsonl")
        save_jsonl(examples, out_path)

    print("\nData preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TransMuCoRes data for CorefInst training.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()
    main(args.config)

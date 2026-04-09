"""
scripts/train_model.py — Fine-tune a causal LM on CorefInst examples.

Usage:
  python scripts/train_model.py [--config config.yaml]
  python scripts/train_model.py --config configs/t4.yaml --few_shot 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(config_path: str = "config.yaml", few_shot_n: int = None) -> None:
    cfg = load_config(config_path)

    data_dir   = cfg["data"]["output_dir"]
    train_path = os.path.join(data_dir, "train.jsonl")
    dev_path   = os.path.join(data_dir, "dev.jsonl")

    if few_shot_n is not None:
        import json, random
        print(f"\n[Few-shot] Sampling {few_shot_n} examples per language …")
        random.seed(cfg.get("training", {}).get("seed", 42))
        lang_buckets: dict = {}
        with open(train_path) as fh:
            for line in fh:
                ex = json.loads(line)
                lang = ex.get("language", "unk")
                lang_buckets.setdefault(lang, []).append(ex)
        selected = []
        for lang, bucket in lang_buckets.items():
            sample = random.sample(bucket, min(few_shot_n, len(bucket)))
            selected.extend(sample)
            print(f"  {lang}: {len(sample)} examples")
        fs_path = os.path.join(data_dir, f"train_fs{few_shot_n}.jsonl")
        with open(fs_path, "w") as fh:
            for ex in selected:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
        train_path = fs_path
        print(f"Few-shot train file: {fs_path} ({len(selected)} total examples)")

    model_cfg = cfg["model"]
    lora_cfg  = cfg["lora"]
    train_cfg = cfg["training"]

    from coref.modeling.train import train

    train(
        train_path=train_path,
        dev_path=dev_path if os.path.exists(dev_path) else None,
        output_dir=train_cfg["output_dir"],
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
        lora_rank=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg.get("target_modules"),
        num_epochs=train_cfg["num_epochs"],
        per_device_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler=train_cfg["lr_scheduler"],
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["max_grad_norm"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        seed=train_cfg["seed"],
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        dataloader_workers=train_cfg.get("dataloader_num_workers", 0),
        backend="auto",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune CorefInst model.")
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--few_shot", type=int, default=None,
                        help="Train on N examples per language (few-shot mode)")
    args = parser.parse_args()
    main(args.config, few_shot_n=args.few_shot)

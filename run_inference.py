"""
run_inference.py — Run controlled inference and evaluate on the test set.

Usage:
  python run_inference.py [--config config.yaml] [--checkpoint PATH] [--split test]
                          [--language hi|ta|bn|all] [--output_dir inference_output/]

For each document:
  1. Group its FrameExamples by doc_id (from the test JSONL).
  2. Run controlled inference → per-example predicted local cluster numbers.
  3. Apply Algorithm 1 (merge_clusters_over_frames) → global clusters.
  4. Compare against gold clusters → CoNLL metrics.

Results are printed per language and overall.
"""

import argparse
import collections
import json
import os
from typing import Dict, List, Optional

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main(
    config_path: str = "config.yaml",
    checkpoint: Optional[str] = None,
    split: str = "test",
    languages: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)

    data_dir    = cfg["data"]["output_dir"]
    data_root   = cfg["data"]["root"]
    model_cfg   = cfg["model"]
    infer_cfg   = cfg.get("inference", {})
    max_clust   = infer_cfg.get("max_cluster_id", 200)
    instr_id    = cfg["preprocessing"]["instruction_id"]
    max_tokens  = cfg["preprocessing"]["max_tokens_per_frame"]

    if checkpoint is None:
        checkpoint = os.path.join(cfg["training"]["output_dir"], "final")
    if output_dir is None:
        output_dir = infer_cfg.get("output_dir", "inference_output")

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    import torch
    from src.model import load_for_inference

    print(f"Loading model from {checkpoint} …")
    model, tokenizer = load_for_inference(
        checkpoint_path=checkpoint,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    # ------------------------------------------------------------------
    # 2. Load test data (gold documents + frame examples)
    # ------------------------------------------------------------------
    from src.dataset_builder import load_documents, build_examples
    from src.inference import run_inference_on_examples
    from src.postprocessor import merge_clusters_over_frames, extract_gold_clusters
    from src.evaluate import evaluate_documents, print_scores

    if languages is None:
        languages = list(cfg["data"]["languages"].keys())

    test_jsonl = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(test_jsonl):
        print(f"[warn] {test_jsonl} not found — regenerating from raw data …")
        docs = load_documents(data_root, split=split, languages=languages)
        examples_all = build_examples(docs, instruction_id=instr_id, max_tokens_per_frame=max_tokens)
    else:
        from src.dataset_builder import load_jsonl
        hf_ds = load_jsonl(test_jsonl)
        examples_all = list(hf_ds)

    # Load gold documents (needed for gold cluster extraction)
    gold_docs = load_documents(data_root, split=split, languages=languages)
    gold_doc_map = {doc.doc_id: doc for doc in gold_docs}

    # ------------------------------------------------------------------
    # 3. Group examples by doc_id (maintain order within each doc)
    # ------------------------------------------------------------------
    doc_examples: Dict[str, List[dict]] = collections.OrderedDict()
    for ex in examples_all:
        did = ex["doc_id"] if isinstance(ex, dict) else ex.doc_id
        doc_examples.setdefault(did, []).append(ex)

    print(f"\n{len(doc_examples)} documents | {len(examples_all)} frame examples")

    # ------------------------------------------------------------------
    # 4. Inference + postprocessing + evaluation per language
    # ------------------------------------------------------------------
    lang_gold: Dict[str, List] = collections.defaultdict(list)
    lang_pred: Dict[str, List] = collections.defaultdict(list)

    for di, (doc_id, frame_exs) in enumerate(doc_examples.items()):
        if di % 20 == 0:
            print(f"  [{di}/{len(doc_examples)}] {doc_id}")

        # Run controlled inference for all frame pairs in this document
        results = run_inference_on_examples(
            model, tokenizer, frame_exs, device=device,
            max_cluster_id=max_clust, verbose=False,
        )

        # Merge local → global clusters (Algorithm 1)
        _, pred_clusters = merge_clusters_over_frames(results)

        # Gold clusters
        if doc_id in gold_doc_map:
            _, gold_clusters = extract_gold_clusters(gold_doc_map[doc_id])
            lang = gold_doc_map[doc_id].language or "all"
        else:
            gold_clusters = {}
            lang = "all"

        lang_gold[lang].append(gold_clusters)
        lang_pred[lang].append(pred_clusters)

        # Save predicted CoNLL (optional, for external scorer)
        if doc_id in gold_doc_map:
            from src.postprocessor import write_conll_predictions
            pred_glob = {mpos: gid for gid, mset in pred_clusters.items() for mpos in mset}
            out_file = os.path.join(output_dir, f"{doc_id}.conll")
            write_conll_predictions(gold_doc_map[doc_id], pred_glob, out_file)

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    from src.evaluate import evaluate_documents, print_scores

    all_gold, all_pred = [], []
    for lang in sorted(lang_gold.keys()):
        glist = lang_gold[lang]
        plist = lang_pred[lang]
        scores = evaluate_documents(glist, plist)
        print_scores(scores, label=f"Language: {lang.upper()} ({len(glist)} docs)")
        all_gold.extend(glist)
        all_pred.extend(plist)

    if len(lang_gold) > 1:
        overall = evaluate_documents(all_gold, all_pred)
        print_scores(overall, label=f"OVERALL ({len(all_gold)} docs)")

    # Save JSON results
    results_path = os.path.join(output_dir, "results.json")
    summary = {}
    for lang in sorted(lang_gold.keys()):
        summary[lang] = evaluate_documents(lang_gold[lang], lang_pred[lang])
    if len(lang_gold) > 1:
        summary["overall"] = evaluate_documents(all_gold, all_pred)
    with open(results_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CorefInst inference and evaluate.")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--split",      default="test", choices=["train", "dev", "test"])
    parser.add_argument("--language",   default="all",  help="'all', 'hi', 'ta', or 'bn'")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    langs = None if args.language == "all" else [args.language]
    main(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        languages=langs,
        output_dir=args.output_dir,
    )

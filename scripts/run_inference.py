"""
scripts/run_inference.py — Run controlled inference and evaluate on the test set.

Usage:
  python scripts/run_inference.py [--config config.yaml] [--split test]
                                  [--checkpoint PATH] [--language hi|ta|bn|all]
                                  [--max-docs N] [--no-resume]

The full test split is ~2,000 frames and ~87,000 masks, each decoded one token
at a time — several hours on an A100. A dropped session must not throw that
away, so every document's predictions are appended to a JSONL shard as soon as
it is scored, and a rerun skips documents already present in that shard.
Scores are recomputed from the shard at the end, so they are identical whether
the run happened in one sitting or five.
"""

import argparse
import collections
import json
import os
import sys
import time
from typing import Dict, List, Optional

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")   # suppress telemetry timeout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _safe_name(doc_id: str) -> str:
    """Make a doc_id safe to use as a filename (ids can contain '#' and '/')."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in doc_id)


def _load_shard(path: str) -> Dict[str, dict]:
    """Read the per-document shard written by earlier runs: doc_id → record."""
    done: Dict[str, dict] = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                # A run killed mid-write leaves one truncated line. Everything
                # before it is still good, so stop here rather than discard it.
                print("[warn] shard ends in a partial record — ignoring the tail")
                break
            done[rec["doc_id"]] = rec
    return done


def main(
    config_path: str = "config.yaml",
    checkpoint: Optional[str] = None,
    split: str = "test",
    languages: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
    resume: bool = True,
) -> None:
    cfg = load_config(config_path)

    data_dir   = cfg["data"]["output_dir"]
    data_root  = cfg["data"]["root"]
    model_cfg  = cfg["model"]
    infer_cfg  = cfg.get("inference", {})
    max_clust  = infer_cfg.get("max_cluster_id", 200)
    instr_id   = cfg["preprocessing"]["instruction_id"]
    max_tokens = cfg["preprocessing"]["max_tokens_per_frame"]
    min_ments  = cfg["preprocessing"].get("min_mentions_per_example", 1)

    if checkpoint is None:
        checkpoint = os.path.join(cfg["training"]["output_dir"], "final")
    if output_dir is None:
        output_dir = infer_cfg.get("output_dir", "inference_output")

    os.makedirs(output_dir, exist_ok=True)

    import torch
    from coref.modeling.model import load_for_inference

    print(f"Loading model from {checkpoint} …")
    model, tokenizer = load_for_inference(
        checkpoint_path=checkpoint,
        max_seq_length=model_cfg["max_seq_length"],
        load_in_4bit=model_cfg["load_in_4bit"],
    )
    device = next(model.parameters()).device
    print(f"Model on device: {device}")

    from coref.data.dataset_builder import load_documents, build_examples, load_jsonl
    from coref.eval.inference import run_inference_on_examples
    from coref.eval.postprocessor import (clusters_from_json, clusters_to_json,
                                      extract_gold_clusters,
                                      merge_clusters_over_frames, write_conll_predictions)
    from coref.eval.evaluate import evaluate_documents, print_scores

    if languages is None:
        languages = list(cfg["data"]["languages"].keys())

    test_jsonl = os.path.join(data_dir, f"{split}.jsonl")
    if not os.path.exists(test_jsonl):
        print(f"[warn] {test_jsonl} not found — regenerating from raw data …")
        docs = load_documents(data_root, split=split, languages=languages)
        examples_all = build_examples(docs, instruction_id=instr_id,
                                      max_tokens_per_frame=max_tokens,
                                      min_mentions=min_ments)
    else:
        hf_ds = load_jsonl(test_jsonl)
        examples_all = list(hf_ds)

    gold_docs = load_documents(data_root, split=split, languages=languages)
    gold_doc_map = {doc.doc_id: doc for doc in gold_docs}

    doc_examples: Dict[str, List[dict]] = collections.OrderedDict()
    for ex in examples_all:
        did = ex["doc_id"] if isinstance(ex, dict) else ex.doc_id
        doc_examples.setdefault(did, []).append(ex)

    print(f"\n{len(doc_examples)} documents | {len(examples_all)} frame examples")

    missing = [d for d in doc_examples if d not in gold_doc_map]
    if missing:
        print(f"[warn] {len(missing)} document(s) have no gold counterpart and are "
              f"excluded from scoring, e.g. {missing[:3]}")

    # Documents are scored in a fixed order, so --max-docs N always covers the
    # same N and a resumed run continues where the previous one stopped.
    scorable = [d for d in doc_examples if d in gold_doc_map]
    if max_docs:
        scorable = scorable[:max_docs]
        print(f"[info] --max-docs {max_docs}: scoring the first "
              f"{len(scorable)} document(s)")

    shard_path = os.path.join(output_dir, f"predictions_{split}.jsonl")
    done = _load_shard(shard_path) if resume else {}
    if done:
        print(f"[resume] {len(done)} document(s) already in {shard_path} — skipping them")
    elif not resume and os.path.exists(shard_path):
        os.remove(shard_path)
        print(f"[info] --no-resume: cleared {shard_path}")

    todo = [d for d in scorable if d not in done]
    print(f"{len(todo)} document(s) left to run\n")

    shard_fh = open(shard_path, "a", encoding="utf-8")

    for di, doc_id in enumerate(todo):
        frame_exs = doc_examples[doc_id]

        t0 = time.time()
        print(f"  [{di+1}/{len(todo)}] {doc_id} ({len(frame_exs)} frames) …", end=" ", flush=True)

        results = run_inference_on_examples(
            model, tokenizer, frame_exs, device=device,
            max_cluster_id=max_clust,
            max_seq_length=model_cfg["max_seq_length"],
            verbose=False,
        )
        _, pred_clusters = merge_clusters_over_frames(results)

        elapsed = time.time() - t0
        print(f"done in {elapsed:.1f}s ({elapsed / max(len(frame_exs), 1):.2f}s/frame)")

        gold_doc = gold_doc_map[doc_id]
        _, gold_clusters = extract_gold_clusters(gold_doc)
        lang = gold_doc.language or "all"

        pred_glob = {mpos: gid for gid, mset in pred_clusters.items() for mpos in mset}
        write_conll_predictions(
            gold_doc, pred_glob,
            os.path.join(output_dir, f"{_safe_name(doc_id)}.conll"),
        )

        record = {
            "doc_id":   doc_id,
            "language": lang,
            "gold":     clusters_to_json(gold_clusters),
            "pred":     clusters_to_json(pred_clusters),
        }
        # Flush per document: a session that dies on document 900 keeps 899.
        shard_fh.write(json.dumps(record) + "\n")
        shard_fh.flush()
        os.fsync(shard_fh.fileno())
        done[doc_id] = record

    shard_fh.close()

    # Score from the shard, so the numbers do not depend on how many sittings
    # the run took.
    lang_gold: Dict[str, List] = collections.defaultdict(list)
    lang_pred: Dict[str, List] = collections.defaultdict(list)
    per_doc: List[dict] = []

    for doc_id in scorable:
        rec = done.get(doc_id)
        if rec is None:
            continue
        lang = rec["language"]
        lang_gold[lang].append(clusters_from_json(rec["gold"]))
        lang_pred[lang].append(clusters_from_json(rec["pred"]))
        per_doc.append(rec)

    if not per_doc:
        print("\n[error] No documents could be scored — nothing written.")
        return

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

    results_path = os.path.join(output_dir, "results.json")
    summary = {}
    for lang in sorted(lang_gold.keys()):
        summary[lang] = evaluate_documents(lang_gold[lang], lang_pred[lang])
    if len(lang_gold) > 1:
        summary["overall"] = evaluate_documents(all_gold, all_pred)
    with open(results_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults saved to {results_path}")

    # Per-document clusters, so analyse_results.py can do cluster-size and
    # error analysis without re-running the model.
    preds_path = os.path.join(output_dir, "predictions.json")
    with open(preds_path, "w") as fh:
        json.dump(per_doc, fh)
    print(f"Per-document clusters saved to {preds_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CorefInst inference and evaluate.")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split",      default="test", choices=["train", "dev", "test"])
    parser.add_argument("--language",   default="all", help="'all', 'hi', 'ta', or 'bn'")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max-docs", dest="max_docs", type=int, default=None,
                        help="score only the first N documents (smoke runs)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="ignore the existing shard and re-run every document")
    args = parser.parse_args()

    langs = None if args.language == "all" else [args.language]
    main(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        languages=langs,
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        resume=args.resume,
    )

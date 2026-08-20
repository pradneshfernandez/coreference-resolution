"""
analysis/baseline.py — Two simple coreference baselines for comparison.

Baselines implemented:
  1. All-singletons  — every mention is its own cluster (lower bound on linking)
  2. All-one-cluster — all mentions in a document form a single cluster
  3. Most-frequent-entity (MFE) — greedily assigns each mention to the cluster
     whose head word it has previously seen, otherwise creates a new cluster.
     A lightweight surface-form heuristic; no model required.

Usage:
  python analysis/baseline.py --config config.yaml [--split test]

Prints CoNLL scores for each baseline per language.
"""

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from coref.data.conll_parser import Document, load_conll_dir, parse_conll_file
from coref.eval.evaluate import evaluate_documents, print_scores
from coref.eval.postprocessor import extract_gold_clusters

# Language config inline (avoids importing dataset_builder which requires `datasets`)
LANGUAGE_CONFIGS: Dict[str, dict] = {
    "hi": {"name": "Hindi",   "codes": ["hin_Deva"]},
    "ta": {"name": "Tamil",   "codes": ["tam_Taml"]},
    "bn": {"name": "Bengali", "codes": ["ben_Beng"]},
}

MPos = Tuple[int, int, int]
Clusters = Dict[int, Set[MPos]]


# ---------------------------------------------------------------------------
# Baseline 1 — All singletons
# ---------------------------------------------------------------------------

def all_singletons(doc: Document) -> Clusters:
    """Every mention is its own cluster."""
    clusters: Clusters = {}
    for i, m in enumerate(doc.mentions):
        clusters[i] = {m.position_key}
    return clusters


# ---------------------------------------------------------------------------
# Baseline 2 — All one cluster
# ---------------------------------------------------------------------------

def all_one_cluster(doc: Document) -> Clusters:
    """All mentions in the document share a single cluster."""
    if not doc.mentions:
        return {}
    return {0: {m.position_key for m in doc.mentions}}


# ---------------------------------------------------------------------------
# Baseline 3 — Most Frequent Entity (surface-form heuristic)
# ---------------------------------------------------------------------------

def _mention_head(doc: Document, m, sent_map: Dict[int, "object"] = None) -> str:
    """Return the last non-punctuation token of a mention as its 'head'."""
    # Index sentences by sent_idx, not by list position — they are not
    # guaranteed to be the same thing.
    sent = (sent_map or {s.sent_idx: s for s in doc.sentences}).get(m.sent_idx)
    if sent is None or not sent.tokens:
        return ""

    span = [tok for tok in sent.tokens if m.start_tok <= tok.idx <= m.end_tok]
    span_tokens = [tok.text for tok in span if tok.text not in ".,;:!?।॥"]
    if span_tokens:
        return span_tokens[-1].lower()
    if span:
        return span[-1].text.lower()     # all-punctuation span
    return ""


def most_frequent_entity(doc: Document) -> Clusters:
    """
    Greedy surface-form baseline.

    Process mentions left-to-right; assign a mention to the existing cluster
    whose head word matches the current mention's head word (exact match).
    Otherwise create a new cluster.
    """
    head_to_cluster: Dict[str, int] = {}
    clusters: Clusters = {}
    next_id = 0
    sent_map = {s.sent_idx: s for s in doc.sentences}

    for m in sorted(doc.mentions, key=lambda x: (x.sent_idx, x.start_tok)):
        head = _mention_head(doc, m, sent_map)
        if head in head_to_cluster:
            cid = head_to_cluster[head]
        else:
            cid = next_id
            next_id += 1
            head_to_cluster[head] = cid
        clusters.setdefault(cid, set()).add(m.position_key)

    return clusters


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

BASELINES = {
    "All-singletons": all_singletons,
    "All-one-cluster": all_one_cluster,
    "Most-frequent-entity (MFE)": most_frequent_entity,
}


def evaluate_baselines(docs: List[Document]) -> Dict[str, dict]:
    """Score every baseline on *docs*; returns {baseline_name: scores}."""
    if not docs:
        print("  (no documents)")
        return {}

    gold_list = []
    for doc in docs:
        _, gc = extract_gold_clusters(doc)
        gold_list.append(gc)

    out: Dict[str, dict] = {}
    for name, fn in BASELINES.items():
        pred_list = [fn(doc) for doc in docs]
        scores = evaluate_documents(gold_list, pred_list)
        print_scores(scores, label=name)
        out[name] = scores
    return out


def main(config_path: str = "config.yaml", split: str = "test",
         save_json: Optional[str] = None) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    data_root = cfg["data"]["root"]
    languages = list(cfg["data"]["languages"].keys())

    print(f"\nBaseline evaluation on split='{split}'")

    lang_docs: Dict[str, List[Document]] = {}
    all_docs: List[Document] = []
    summary: Dict[str, dict] = {}

    for lang in languages:
        codes = LANGUAGE_CONFIGS[lang]["codes"]
        lang_name = LANGUAGE_CONFIGS[lang]["name"]

        split_map = {"train": "train", "dev": "development", "test": "test"}
        dir_map   = {"train": "litbank_train", "dev": "litbank_val", "test": "litbank_test"}

        docs: List[Document] = []
        if lang == "hi":
            d = os.path.join(data_root, "mujadia_conll", split_map[split])
            docs += load_conll_dir(d, language_filter=None, language=lang)
        d = os.path.join(data_root, "onto_notes_archive", split_map[split])
        docs += load_conll_dir(d, language_filter=codes, language=lang, recursive=True)
        d = os.path.join(data_root, dir_map[split])
        docs += load_conll_dir(d, language_filter=codes, language=lang)

        # Keep only docs that have at least 2 mentions (so baseline is meaningful)
        docs = [d for d in docs if len(d.mentions) >= 2]

        print(f"\n{'='*55}")
        print(f"  Language: {lang_name} ({lang}) — {len(docs)} documents")
        print(f"{'='*55}")
        summary[lang] = evaluate_baselines(docs)
        summary[lang]["num_docs"] = len(docs)

        lang_docs[lang] = docs
        all_docs.extend(docs)

    if len(languages) > 1:
        print(f"\n{'='*55}")
        print(f"  OVERALL — {len(all_docs)} documents")
        print(f"{'='*55}")
        summary["overall"] = evaluate_baselines(all_docs)
        summary["overall"]["num_docs"] = len(all_docs)

    if save_json:
        os.makedirs(os.path.dirname(save_json) or ".", exist_ok=True)
        with open(save_json, "w") as fh:
            json.dump({"split": split, "scorer": "coref.eval.evaluate (in-repo)",
                       "baselines": summary}, fh, indent=2)
        print(f"\nBaseline scores saved to {save_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split",  default="test", choices=["train", "dev", "test"])
    parser.add_argument("--save_json", default=None,
                        help="also write the scores to this JSON path")
    args = parser.parse_args()
    main(args.config, args.split, args.save_json)

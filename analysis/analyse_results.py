"""
analysis/analyse_results.py — Analyse inference results in depth.

Reads the results.json produced by run_inference.py and produces:
  1. Per-language CoNLL table (MUC / B³ / CEAFe / CoNLL-F)
  2. Per-language cluster-size distribution (gold vs predicted)
  3. Error analysis: over-merging / under-merging / correct singleton rate
  4. Instruction-set ablation table (if multiple result files are provided)

Usage:
  # Single run
  python analysis/analyse_results.py --results_json inference_output/results.json

  # Instruction ablation (supply one results.json per instruction set)
  python analysis/analyse_results.py \
      --ablation_dir ablation_results/ \
      --ablation_names inst1 inst2 inst3 inst4 inst5
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Pretty table helpers
# ---------------------------------------------------------------------------

def _hdr(cols: List[str], widths: List[int]) -> str:
    row = " | ".join(f"{c:{w}s}" for c, w in zip(cols, widths))
    sep = "-+-".join("-" * w for w in widths)
    return f"{row}\n{sep}"


def _row(vals: List, widths: List[int]) -> str:
    return " | ".join(f"{str(v):{w}s}" for v, w in zip(vals, widths))


# ---------------------------------------------------------------------------
# 1. Per-language score table
# ---------------------------------------------------------------------------

def print_language_table(results: dict) -> None:
    lang_keys = [k for k in results if k != "overall"]
    if not lang_keys:
        print("No per-language results found.")
        return

    print("\n" + "=" * 70)
    print("  Per-language CoNLL results")
    print("=" * 70)
    print(_hdr(["Language", "MUC-F", "B³-F", "CEAFe-F", "CoNLL-F", "Docs"],
               [12, 8, 8, 9, 9, 6]))

    for lang in sorted(lang_keys):
        r = results[lang]
        print(_row(
            [lang.upper(),
             f"{r['muc']['f']:.2f}",
             f"{r['b3']['f']:.2f}",
             f"{r['ceafe']['f']:.2f}",
             f"{r['conll']['f']:.2f}",
             str(r.get("num_docs", "?"))],
            [12, 8, 8, 9, 9, 6],
        ))

    if "overall" in results:
        r = results["overall"]
        print("-" * 55)
        print(_row(
            ["OVERALL",
             f"{r['muc']['f']:.2f}",
             f"{r['b3']['f']:.2f}",
             f"{r['ceafe']['f']:.2f}",
             f"{r['conll']['f']:.2f}",
             str(r.get("num_docs", "?"))],
            [12, 8, 8, 9, 9, 6],
        ))

    print()


# ---------------------------------------------------------------------------
# 2. Cluster-size distribution (gold vs predicted)
# ---------------------------------------------------------------------------

def _cluster_size_dist(clusters_list: List[Dict]) -> Counter:
    """Count how many clusters of each size appear across all documents."""
    dist: Counter = Counter()
    for clusters in clusters_list:
        for mset in clusters.values():
            dist[len(mset)] += 1
    return dist


def print_cluster_distribution(gold_list: List[Dict], pred_list: List[Dict],
                                label: str = "") -> None:
    gold_dist = _cluster_size_dist(gold_list)
    pred_dist = _cluster_size_dist(pred_list)

    all_sizes = sorted(set(gold_dist) | set(pred_dist))
    max_size   = min(max(all_sizes, default=1), 10)

    print(f"\n  Cluster-size distribution — {label}")
    print(_hdr(["Size", "Gold #", "Pred #", "Δ"], [6, 8, 8, 8]))
    for sz in range(1, max_size + 1):
        g = gold_dist.get(sz, 0)
        p = pred_dist.get(sz, 0)
        print(_row([f"={sz}" if sz < 10 else f"≥10",
                    str(g), str(p), f"{p-g:+d}"], [6, 8, 8, 8]))
    if max(all_sizes, default=0) > 10:
        g = sum(v for k, v in gold_dist.items() if k > 10)
        p = sum(v for k, v in pred_dist.items() if k > 10)
        print(_row(["≥11", str(g), str(p), f"{p-g:+d}"], [6, 8, 8, 8]))
    print()


# ---------------------------------------------------------------------------
# 3. Error analysis
# ---------------------------------------------------------------------------

def error_analysis(gold_list: List[Dict], pred_list: List[Dict],
                   label: str = "") -> None:
    """
    Classify predicted clusters relative to gold:
      - Perfect match: pred cluster = gold cluster exactly
      - Over-merge:    pred cluster spans 2+ gold clusters
      - Under-merge:   a gold cluster is split across 2+ pred clusters
      - Spurious:      pred cluster has no overlap with any gold cluster
    """
    total_gold   = sum(len(gc) for gc in gold_list)
    total_pred   = sum(len(pc) for pc in pred_list)
    perfect      = over_merge = under_merge = spurious = 0

    for gold, pred in zip(gold_list, pred_list):
        # Build reverse map: mpos → gold_cluster_id
        mpos_to_gold = {mpos: cid for cid, mset in gold.items() for mpos in mset}
        mpos_to_pred = {mpos: cid for cid, mset in pred.items() for mpos in mset}

        # Gold cluster analysis
        for cid, g_mset in gold.items():
            if len(g_mset) < 2:
                continue
            pred_ids = {mpos_to_pred[m] for m in g_mset if m in mpos_to_pred}
            if len(pred_ids) > 1:
                under_merge += 1

        # Pred cluster analysis
        for cid, p_mset in pred.items():
            if not p_mset:
                continue
            gold_ids = {mpos_to_gold[m] for m in p_mset if m in mpos_to_gold}
            if not gold_ids:
                spurious += 1
            elif len(gold_ids) > 1:
                over_merge += 1
            else:
                # Exactly one gold cluster covered — check if it's a perfect match
                g_mset = gold.get(next(iter(gold_ids)), set())
                if p_mset == g_mset:
                    perfect += 1

    print(f"\n  Error analysis — {label}")
    print(f"  Gold clusters (size≥2): {total_gold}")
    print(f"  Pred clusters total:    {total_pred}")
    print(f"  Perfect matches:        {perfect}")
    print(f"  Over-merged pred:       {over_merge}  (one pred spans multiple gold clusters)")
    print(f"  Under-merged gold:      {under_merge}  (one gold cluster split across pred)")
    print(f"  Spurious pred:          {spurious}  (pred cluster has no gold overlap)")
    print()


# ---------------------------------------------------------------------------
# 4. Instruction ablation table
# ---------------------------------------------------------------------------

def print_ablation_table(ablation_dir: str, names: List[str]) -> None:
    print("\n" + "=" * 70)
    print("  Instruction-set ablation (CoNLL-F per language)")
    print("=" * 70)

    # Collect all language keys
    lang_keys: set = set()
    results_by_name: Dict[str, dict] = {}
    for name in names:
        path = os.path.join(ablation_dir, f"{name}.json")
        if not os.path.exists(path):
            print(f"  [warn] {path} not found — skipping {name}")
            continue
        with open(path) as fh:
            r = json.load(fh)
        results_by_name[name] = r
        lang_keys |= {k for k in r if k != "overall"}

    if not results_by_name:
        print("  No ablation results found.")
        return

    lang_keys_sorted = sorted(lang_keys) + ["overall"]
    col_width = 9
    headers = ["Instruction"] + lang_keys_sorted
    widths  = [15] + [col_width] * len(lang_keys_sorted)
    print(_hdr(headers, widths))

    for name, r in results_by_name.items():
        row = [name]
        for lang in lang_keys_sorted:
            if lang in r:
                row.append(f"{r[lang]['conll']['f']:.2f}")
            else:
                row.append("—")
        print(_row(row, widths))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    results_json: Optional[str] = None,
    ablation_dir: Optional[str] = None,
    ablation_names: Optional[List[str]] = None,
) -> None:
    if results_json and os.path.exists(results_json):
        with open(results_json) as fh:
            results = json.load(fh)

        print_language_table(results)
    else:
        if results_json:
            print(f"[warn] {results_json} not found — skipping score table")

    if ablation_dir and ablation_names:
        print_ablation_table(ablation_dir, ablation_names)
    elif ablation_dir:
        # Auto-discover JSON files in the directory
        names = [os.path.splitext(f)[0]
                 for f in sorted(os.listdir(ablation_dir))
                 if f.endswith(".json")]
        if names:
            print_ablation_table(ablation_dir, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_json",   default=None,
                        help="Path to inference_output/results.json")
    parser.add_argument("--ablation_dir",   default=None,
                        help="Directory containing one results JSON per instruction run")
    parser.add_argument("--ablation_names", nargs="*", default=None,
                        help="Names (without .json) of ablation files to compare")
    parser.add_argument("--config",         default="config.yaml")
    args = parser.parse_args()

    main(
        results_json=args.results_json,
        ablation_dir=args.ablation_dir,
        ablation_names=args.ablation_names,
    )

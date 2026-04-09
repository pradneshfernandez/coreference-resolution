"""
evaluate.py — Standard CoNLL coreference evaluation metrics.

Implements from scratch:
  • MUC   (Vilain et al., 1995) — link-based metric
  • B³    (Bagga & Baldwin, 1998) — mention-based metric
  • CEAFe (Luo, 2005) — entity-based metric with optimal assignment
  • CoNLL = average of MUC, B³, CEAFe

All three metrics are combined into a single CoNLL score that is the
primary metric reported in the CorefInst paper (and in CorefUD shared tasks).

Inputs are dicts of {cluster_id: frozenset_of_mention_position_keys}.
Mention position keys are (sent_idx, start_tok, end_tok) tuples.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


MentionKey = Tuple[int, int, int]   # (sent_idx, start_tok, end_tok)
Clusters   = Dict[int, Set[MentionKey]]


# ---------------------------------------------------------------------------
# Precision / Recall / F1 helpers
# ---------------------------------------------------------------------------

def _prf(num: float, denom_p: float, denom_r: float) -> Tuple[float, float, float]:
    p = num / denom_p if denom_p > 0 else 0.0
    r = num / denom_r if denom_r > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# MUC metric
# ---------------------------------------------------------------------------

def _muc_partition_count(cluster: Set[MentionKey], other_clusters: Clusters) -> int:
    """
    Number of partitions of *cluster* induced by *other_clusters*.

    partition_count = number of distinct "other" clusters that contain at
    least one mention from *cluster*.  Mentions not found in any other cluster
    each form their own singleton partition (counts as +1 per orphan).
    """
    seen: Set[int] = set()
    orphans = 0
    for mkey in cluster:
        found = False
        for cid, cset in other_clusters.items():
            if mkey in cset:
                seen.add(cid)
                found = True
                break
        if not found:
            orphans += 1
    return len(seen) + orphans


def muc_score(
    gold: Clusters, pred: Clusters
) -> Tuple[float, float, float]:
    """
    MUC coreference metric.

    Returns (precision, recall, F1).
    """
    # Recall: for each gold cluster, count links partitioned by predicted
    recall_num   = 0.0
    recall_denom = 0.0
    for cid, g_cluster in gold.items():
        size = len(g_cluster)
        if size < 2:
            continue   # skip singletons (they have 0 links)
        k = _muc_partition_count(g_cluster, pred)
        recall_num   += size - k
        recall_denom += size - 1

    # Precision: symmetric (swap gold / pred)
    prec_num   = 0.0
    prec_denom = 0.0
    for cid, p_cluster in pred.items():
        size = len(p_cluster)
        if size < 2:
            continue
        k = _muc_partition_count(p_cluster, gold)
        prec_num   += size - k
        prec_denom += size - 1

    return _prf(recall_num, prec_denom, recall_denom)


# ---------------------------------------------------------------------------
# B³ metric
# ---------------------------------------------------------------------------

def _get_cluster_of(mkey: MentionKey, clusters: Clusters) -> Optional[Set[MentionKey]]:
    """Return the cluster containing mkey, or None if not found."""
    for cset in clusters.values():
        if mkey in cset:
            return cset
    return None


def b3_score(
    gold: Clusters, pred: Clusters
) -> Tuple[float, float, float]:
    """
    B³ coreference metric.

    Returns (precision, recall, F1).
    """
    all_mentions: Set[MentionKey] = set()
    for cset in gold.values():
        all_mentions |= cset

    prec_sum = 0.0
    rec_sum  = 0.0
    count    = 0

    for mkey in all_mentions:
        g_cluster = _get_cluster_of(mkey, gold) or {mkey}
        p_cluster = _get_cluster_of(mkey, pred) or {mkey}
        overlap = len(g_cluster & p_cluster)

        prec_sum += overlap / len(p_cluster)
        rec_sum  += overlap / len(g_cluster)
        count    += 1

    if count == 0:
        return 0.0, 0.0, 0.0

    p = prec_sum / count
    r = rec_sum  / count
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# CEAFe metric (entity-level CEAF, phi_4 similarity)
# ---------------------------------------------------------------------------

def _ceafe_similarity(g_cluster: Set[MentionKey], p_cluster: Set[MentionKey]) -> float:
    """
    phi_4(Gi, Pj) = 2 * |Gi ∩ Pj| / (|Gi| + |Pj|)

    This is the entity-level CEAF (CEAFe) similarity (Luo 2005).
    Self-similarity phi_4(G, G) = 1.0 regardless of cluster size,
    so both precision and recall denominators equal the number of clusters.
    """
    denom = len(g_cluster) + len(p_cluster)
    if denom == 0:
        return 0.0
    return 2.0 * len(g_cluster & p_cluster) / denom


def _hungarian_max_assignment(
    gold_list: List[Set[MentionKey]],
    pred_list: List[Set[MentionKey]],
) -> float:
    """
    Find maximum-weight bipartite matching between gold and pred clusters.

    Uses the Kuhn-Munkres (Hungarian) algorithm via scipy if available;
    falls back to a greedy approximation.
    """
    n = len(gold_list)
    m = len(pred_list)
    if n == 0 or m == 0:
        return 0.0

    # Build similarity matrix
    sim = [[_ceafe_similarity(gold_list[i], pred_list[j]) for j in range(m)] for i in range(n)]

    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        import numpy as np

        mat = np.array(sim)
        row_ind, col_ind = linear_sum_assignment(-mat)   # negate for maximisation
        return float(mat[row_ind, col_ind].sum())
    except ImportError:
        # Greedy fallback: repeatedly pick the best remaining pair
        matched = 0.0
        used_g, used_p = set(), set()
        pairs = sorted(
            [(sim[i][j], i, j) for i in range(n) for j in range(m)],
            reverse=True,
        )
        for val, i, j in pairs:
            if i not in used_g and j not in used_p:
                matched += val
                used_g.add(i)
                used_p.add(j)
        return matched


def ceafe_score(
    gold: Clusters, pred: Clusters
) -> Tuple[float, float, float]:
    """
    CEAFe coreference metric.

    Returns (precision, recall, F1).
    """
    gold_list = list(gold.values())
    pred_list = list(pred.values())

    best = _hungarian_max_assignment(gold_list, pred_list)

    p = best / len(pred_list) if pred_list else 0.0
    r = best / len(gold_list) if gold_list else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


# ---------------------------------------------------------------------------
# CoNLL combined score
# ---------------------------------------------------------------------------

def conll_score(
    gold: Clusters, pred: Clusters
) -> dict:
    """
    Compute all three CoNLL coreference metrics and their average.

    Returns a dict:
    {
        'muc':   {'p': …, 'r': …, 'f': …},
        'b3':    {'p': …, 'r': …, 'f': …},
        'ceafe': {'p': …, 'r': …, 'f': …},
        'conll': {'p': …, 'r': …, 'f': …},   # macro avg of p/r/f above
    }
    """
    muc_p,   muc_r,   muc_f   = muc_score(gold, pred)
    b3_p,    b3_r,    b3_f    = b3_score(gold, pred)
    ceafe_p, ceafe_r, ceafe_f = ceafe_score(gold, pred)

    avg_p = (muc_p   + b3_p   + ceafe_p)  / 3.0
    avg_r = (muc_r   + b3_r   + ceafe_r)  / 3.0
    avg_f = (muc_f   + b3_f   + ceafe_f)  / 3.0

    return {
        "muc":   {"p": muc_p,   "r": muc_r,   "f": muc_f},
        "b3":    {"p": b3_p,    "r": b3_r,    "f": b3_f},
        "ceafe": {"p": ceafe_p, "r": ceafe_r, "f": ceafe_f},
        "conll": {"p": avg_p,   "r": avg_r,   "f": avg_f},
    }


# ---------------------------------------------------------------------------
# Aggregate over a list of documents
# ---------------------------------------------------------------------------

def evaluate_documents(
    gold_clusters_list: List[Clusters],
    pred_clusters_list: List[Clusters],
) -> dict:
    """
    Micro-average CoNLL metrics over a list of documents.

    Each element of gold/pred_clusters_list is a Clusters dict for one document.
    """
    # Accumulate numerators and denominators for micro-averaging
    muc_num_p = muc_den_p = 0.0
    muc_num_r = muc_den_r = 0.0
    b3_sum_p  = b3_sum_r  = 0
    b3_count  = 0
    ce_match  = ce_den_p = ce_den_r = 0.0

    for gold, pred in zip(gold_clusters_list, pred_clusters_list):
        # MUC numerators/denominators
        for cid, g_cluster in gold.items():
            s = len(g_cluster)
            if s < 2:
                continue
            k = _muc_partition_count(g_cluster, pred)
            muc_num_r += s - k
            muc_den_r += s - 1
        for cid, p_cluster in pred.items():
            s = len(p_cluster)
            if s < 2:
                continue
            k = _muc_partition_count(p_cluster, gold)
            muc_num_p += s - k
            muc_den_p += s - 1

        # B³
        all_m: Set[MentionKey] = set()
        for cset in gold.values():
            all_m |= cset
        for mkey in all_m:
            gc = _get_cluster_of(mkey, gold) or {mkey}
            pc = _get_cluster_of(mkey, pred) or {mkey}
            ov = len(gc & pc)
            b3_sum_p += ov / len(pc)
            b3_sum_r += ov / len(gc)
            b3_count += 1

        # CEAFe — uses phi_4 similarity (via _hungarian_max_assignment which
        # calls _ceafe_similarity internally), denominator = cluster count
        gold_list = list(gold.values())
        pred_list = list(pred.values())
        ce_match  += _hungarian_max_assignment(gold_list, pred_list)
        # phi_4(G, G) = 1.0 for every cluster, so denominator = count
        ce_den_p  += len(pred_list)
        ce_den_r  += len(gold_list)

    muc_p = muc_num_p / muc_den_p if muc_den_p > 0 else 0.0
    muc_r = muc_num_r / muc_den_r if muc_den_r > 0 else 0.0
    muc_f = 2 * muc_p * muc_r / (muc_p + muc_r) if (muc_p + muc_r) > 0 else 0.0

    b3_p = b3_sum_p / b3_count if b3_count > 0 else 0.0
    b3_r = b3_sum_r / b3_count if b3_count > 0 else 0.0
    b3_f = 2 * b3_p * b3_r / (b3_p + b3_r) if (b3_p + b3_r) > 0 else 0.0

    ce_p = ce_match / ce_den_p if ce_den_p > 0 else 0.0
    ce_r = ce_match / ce_den_r if ce_den_r > 0 else 0.0
    ce_f = 2 * ce_p * ce_r / (ce_p + ce_r) if (ce_p + ce_r) > 0 else 0.0

    avg_p = (muc_p + b3_p + ce_p) / 3.0
    avg_r = (muc_r + b3_r + ce_r) / 3.0
    avg_f = (muc_f + b3_f + ce_f) / 3.0

    return {
        "muc":   {"p": round(muc_p * 100, 2), "r": round(muc_r * 100, 2), "f": round(muc_f * 100, 2)},
        "b3":    {"p": round(b3_p  * 100, 2), "r": round(b3_r  * 100, 2), "f": round(b3_f  * 100, 2)},
        "ceafe": {"p": round(ce_p  * 100, 2), "r": round(ce_r  * 100, 2), "f": round(ce_f  * 100, 2)},
        "conll": {"p": round(avg_p * 100, 2), "r": round(avg_r * 100, 2), "f": round(avg_f * 100, 2)},
        "num_docs": len(gold_clusters_list),
    }


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_scores(scores: dict, label: str = "") -> None:
    if label:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
    for metric in ("muc", "b3", "ceafe", "conll"):
        d = scores[metric]
        tag = metric.upper() if metric != "conll" else "CoNLL"
        print(f"  {tag:6s}  P={d['p']:6.2f}  R={d['r']:6.2f}  F={d['f']:6.2f}")
    if "num_docs" in scores:
        print(f"  Documents evaluated: {scores['num_docs']}")

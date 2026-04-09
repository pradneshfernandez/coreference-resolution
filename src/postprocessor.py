"""
postprocessor.py — Convert per-frame predicted local cluster numbers into
                   a document-level global clustering.

Implements Algorithm 1 ("Merge Clusters over Frames") from the paper verbatim.

The core insight:
  • Frame pair k covers (frame_k, frame_{k+1}).
  • Frame pair k+1 covers (frame_{k+1}, frame_{k+2}).
  • frame_{k+1} appears as the "after" in pair k and as the "before" in pair k+1.
  • This overlap lets us map local cluster numbers across consecutive pairs via
    the shared mention positions.

Algorithm 1 pseudocode (from paper, reproduced):
─────────────────────────────────────────────────
1: globCls is initialized.
2: for all before, after ∈ frameTuples do
3:   map = {}
4:   for mPos, localNo ← before do
5:     map[localNo] ← globCls[mPos]
6:   end for
7:   for all mPos, localNo ∈ after do
8:     if localNo ∉ map then
9:       i ← max(globCls) + 1
10:      map[localNo] ← i
11:    end if
12:    globCls[mPos] = map[localNo]
13:  end for
14: end for
─────────────────────────────────────────────────

mPos = (sent_idx, start_tok, end_tok) — a unique positional key per mention.

Note: for the very first frame pair, all before-mentions will be absent from
globCls (it is empty).  We handle this by assigning new global IDs in that
case (see _ensure_before_initialised).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MPos = Tuple[int, int, int]      # (sent_idx, start_tok, end_tok)
GlobalCluster = Dict[MPos, int]  # mPos → global cluster id


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

def merge_clusters_over_frames(
    inference_results: List[dict],
) -> Tuple[GlobalCluster, Dict[int, Set[MPos]]]:
    """
    Merge per-frame local cluster numbers into a document-level clustering.

    Args:
        inference_results — list of per-frame-pair dicts produced by
                            inference.run_inference_on_examples(), each with:
                              'before_mentions': [{sent_idx, start_tok, end_tok,
                                                   cluster_id, predicted_local_no}, …]
                              'after_mentions':  same

    Returns:
        glob_cls   — dict mPos → global_cluster_id for every mention in the doc
        clusters   — dict global_cluster_id → {mPos, …}  (inverse index)
    """
    glob_cls: GlobalCluster = {}
    all_global_ids: Set[int] = set()

    def _new_global_id() -> int:
        gid = (max(all_global_ids) + 1) if all_global_ids else 0
        all_global_ids.add(gid)
        return gid

    for frame_pair in inference_results:
        before_mentions: List[dict] = frame_pair.get("before_mentions", [])
        after_mentions:  List[dict] = frame_pair.get("after_mentions",  [])

        # ------------------------------------------------------------------
        # Steps 3–6: build map from (local_no → global_id) using before frame.
        # For mentions already in glob_cls, use their existing global id.
        # For mentions NOT yet in glob_cls (first pair), assign fresh global ids.
        # ------------------------------------------------------------------
        local_to_global: Dict[int, int] = {}

        for m in before_mentions:
            local_no = int(m.get("predicted_local_no", m.get("local_no", 0)))
            mpos: MPos = (int(m["sent_idx"]), int(m["start_tok"]), int(m["end_tok"]))

            if mpos in glob_cls:
                # Existing mention — use its already-assigned global id
                existing_gid = glob_cls[mpos]
                # If this local_no was already mapped to a different global id,
                # prefer the one already in glob_cls (the earlier assignment wins)
                if local_no not in local_to_global:
                    local_to_global[local_no] = existing_gid
                # If there's a conflict, keep the mapping consistent with glob_cls
            else:
                # Mention not yet seen — assign a new global id (first frame pair)
                if local_no not in local_to_global:
                    local_to_global[local_no] = _new_global_id()
                glob_cls[mpos] = local_to_global[local_no]

        # ------------------------------------------------------------------
        # Steps 7–13: process after frame using the established map.
        # ------------------------------------------------------------------
        for m in after_mentions:
            local_no = int(m.get("predicted_local_no", m.get("local_no", 0)))
            mpos: MPos = (int(m["sent_idx"]), int(m["start_tok"]), int(m["end_tok"]))

            if local_no not in local_to_global:
                # New cluster not seen in before frame → assign fresh global id
                local_to_global[local_no] = _new_global_id()

            glob_cls[mpos] = local_to_global[local_no]

    # Build inverse index: global_cluster_id → {mPos, …}
    clusters: Dict[int, Set[MPos]] = defaultdict(set)
    for mpos, gid in glob_cls.items():
        clusters[gid].add(mpos)

    return glob_cls, dict(clusters)


# ---------------------------------------------------------------------------
# Gold cluster extraction (for evaluation)
# ---------------------------------------------------------------------------

def extract_gold_clusters(doc) -> Tuple[GlobalCluster, Dict[int, Set[MPos]]]:
    """
    Build the gold clustering from a parsed Document object.

    Returns:
        gold_glob_cls  — dict mPos → gold_cluster_id
        gold_clusters  — dict gold_cluster_id → {mPos, …}
    """
    gold_glob_cls: GlobalCluster = {}
    gold_clusters: Dict[int, Set[MPos]] = defaultdict(set)

    for m in doc.mentions:
        mpos: MPos = (m.sent_idx, m.start_tok, m.end_tok)
        gold_glob_cls[mpos] = m.cluster_id
        gold_clusters[m.cluster_id].add(mpos)

    return gold_glob_cls, dict(gold_clusters)


# ---------------------------------------------------------------------------
# CoNLL output (for use with external scorers)
# ---------------------------------------------------------------------------

def write_conll_predictions(
    doc,
    pred_glob_cls: GlobalCluster,
    out_path: str,
    part: int = 0,
) -> None:
    """
    Write predicted coreference annotations in CoNLL format.

    Produces a file readable by the CorefUD scorer or the standard CoNLL scorer.

    Args:
        doc           — parsed Document (for token/sentence structure)
        pred_glob_cls — dict (sent_idx, start_tok, end_tok) → pred_cluster_id
        out_path      — output file path
        part          — part number (usually 0)
    """
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Build token-level coreference annotations from predicted spans
    # coref_col[sent_idx][tok_idx] = list of annotation strings (e.g. '(0)', '(0', '0)')
    coref_col: Dict[int, Dict[int, List[str]]] = {}
    for sent in doc.sentences:
        coref_col[sent.sent_idx] = {tok.idx: [] for tok in sent.tokens}

    for (s_idx, start, end), gid in pred_glob_cls.items():
        if s_idx not in coref_col:
            continue
        if start == end:
            coref_col[s_idx][start].append(f"({gid})")
        else:
            coref_col[s_idx][start].append(f"({gid}")
            coref_col[s_idx][end].append(f"{gid})")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"#begin document ({doc.doc_id}); part {part}\n")
        for sent in doc.sentences:
            for tok in sent.tokens:
                ann_list = coref_col.get(sent.sent_idx, {}).get(tok.idx, [])
                ann = "|".join(ann_list) if ann_list else "-"
                fh.write(
                    f"{doc.doc_id}\t{part}\t{tok.idx}\t{tok.text}\t_\t_\t{ann}\n"
                )
            fh.write("\n")
        fh.write("#end document\n")

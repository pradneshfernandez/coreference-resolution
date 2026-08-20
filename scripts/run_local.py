"""
scripts/run_local.py — Run the CorefInst pipeline on a laptop, without a GPU.

The Colab notebook covers the GPU path (fine-tune Llama 3.1 8B in 4-bit, then
run controlled inference). This script covers everything that can be validated
*before* that: it exercises the same parsing → framing → inference → Algorithm 1
→ CoNLL-scoring path locally, so a bug in any of those is found on the laptop
rather than halfway through a paid GPU session.

Stages
------
  check     Environment + dataset sanity check. Pure Python; no torch needed.
  prepare   Build processed_data/{train,dev,test}.jsonl (optionally a subset).
  dryrun    Full pipeline with a *stub* predictor instead of an LLM. Proves the
            postprocessor and scorer work end to end. No torch needed.
  baseline  Score the all-singletons / all-one-cluster / MFE baselines. These
            are the numbers a fine-tuned model has to beat. No torch needed.
  smoke     Real controlled inference on a handful of frames using a small
            CPU-sized model. Needs torch + transformers; slow but real.
  all       check → prepare → dryrun → baseline (skips smoke).

Usage
-----
  python scripts/run_local.py check
  python scripts/run_local.py all --config configs/cpu.yaml --max-docs 5
  python scripts/run_local.py smoke --model Qwen/Qwen2.5-0.5B-Instruct --max-frames 2

Every stage accepts --max-docs / --max-frames so a run finishes in seconds.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

DEFAULT_CONFIG = "configs/cpu.yaml"
# Small enough to load and run on a CPU in a reasonable time.
DEFAULT_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def _have(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


def _load_docs(cfg: dict, split: str, max_docs: Optional[int]) -> List:
    """Load gold documents for a split, optionally capped for a quick run."""
    from coref.data.dataset_builder import load_documents

    languages = list(cfg["data"]["languages"].keys())
    docs = load_documents(cfg["data"]["root"], split=split, languages=languages)

    if max_docs:
        # Round-robin across languages so a capped run still covers all of them.
        by_lang: Dict[str, List] = collections.OrderedDict()
        for d in docs:
            by_lang.setdefault(d.language or "all", []).append(d)
        capped: List = []
        per_lang = max(1, max_docs // max(len(by_lang), 1))
        for lang, dlist in by_lang.items():
            capped.extend(dlist[:per_lang])
        docs = capped[:max_docs]

    return docs


# ---------------------------------------------------------------------------
# Stage: check
# ---------------------------------------------------------------------------

def stage_check(cfg: dict, config_path: str, max_docs: Optional[int]) -> bool:
    """Verify the environment and that the dataset parses into usable examples."""
    _banner("Stage 1/4 — environment & data check")

    print(f"  Python           : {sys.version.split()[0]}")
    print(f"  Config           : {config_path}")

    torch_ok = _have("torch")
    if torch_ok:
        import torch
        print(f"  torch            : {torch.__version__} "
              f"(CUDA {'yes' if torch.cuda.is_available() else 'no'})")
    else:
        print("  torch            : not installed "
              "(check/prepare/dryrun/baseline still run; smoke does not)")

    for mod in ("transformers", "datasets", "peft", "trl", "scipy", "numpy"):
        print(f"  {mod:17s}: {'installed' if _have(mod) else 'MISSING'}")

    if not _have("scipy"):
        print("  [warn] scipy missing — CEAFe falls back to a greedy assignment, "
              "which understates the score slightly.")

    data_root = cfg["data"]["root"]
    if not os.path.isdir(data_root):
        print(f"\n  [error] data root '{data_root}' does not exist.")
        return False

    print(f"\n  Scope            : {'every document' if not max_docs else f'a {max_docs}-document sample'}")

    ok = True
    for split in ("train", "dev", "test"):
        docs = _load_docs(cfg, split, max_docs)
        if not docs:
            print(f"\n  [error] split '{split}' produced 0 documents.")
            ok = False
            continue

        ids = collections.Counter(d.doc_id for d in docs)
        dups = sum(v - 1 for v in ids.values() if v > 1)
        n_mentions = sum(len(d.mentions) for d in docs)
        n_sents = sum(len(d.sentences) for d in docs)
        print(f"\n  split={split:5s} docs={len(docs):5d} sentences={n_sents:6d} "
              f"mentions={n_mentions:6d} duplicate-ids={dups}")
        if dups:
            print("  [error] duplicate doc_ids would merge unrelated documents.")
            ok = False

        # Mention spans must stay inside their own sentence, or the position
        # keys used by the scorer point at the wrong tokens.
        bad = 0
        for d in docs:
            smap = {s.sent_idx: s for s in d.sentences}
            for m in d.mentions:
                sent = smap.get(m.sent_idx)
                if sent is None or m.end_tok >= len(sent.tokens) or m.end_tok < m.start_tok:
                    bad += 1
        if bad:
            print(f"  [error] {bad} mention(s) with out-of-range spans.")
            ok = False
        else:
            print("  all mention spans in range ✓")

    return ok


# ---------------------------------------------------------------------------
# Stage: prepare
# ---------------------------------------------------------------------------

def stage_prepare(cfg: dict, max_docs: Optional[int]) -> bool:
    """Build the JSONL files the trainer reads."""
    _banner("Stage 2/4 — build frame examples")

    from coref.data.dataset_builder import build_examples, save_jsonl

    out_dir = cfg["data"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    instr_id = cfg["preprocessing"]["instruction_id"]
    max_tokens = cfg["preprocessing"]["max_tokens_per_frame"]
    min_ments = cfg["preprocessing"].get("min_mentions_per_example", 1)

    max_seq_length = cfg["model"]["max_seq_length"]

    ok = True
    for split in ("train", "dev", "test"):
        docs = _load_docs(cfg, split, max_docs)
        examples = build_examples(docs, instruction_id=instr_id,
                                  max_tokens_per_frame=max_tokens,
                                  min_mentions=min_ments)
        if not examples:
            print(f"  [error] split '{split}' produced 0 frame examples.")
            ok = False
            continue

        path = os.path.join(out_dir, f"{split}.jsonl")
        save_jsonl(examples, path)

        # Masks and mention records must correspond 1:1, in the same order —
        # that correspondence is what maps a prediction back to a span.
        for ex in examples:
            n_masks = ex.masked_input.count("#MASK") + ex.masked_input.count("@MASK")
            n_mentions = len(ex.before_mentions) + len(ex.after_mentions)
            if n_masks != n_mentions:
                print(f"  [error] {split}: an example has {n_masks} masks but "
                      f"{n_mentions} mention records.")
                ok = False
                break

        # Length sanity. What must fit in max_seq_length during TRAINING is the
        # whole SFT string — instruction + masked_input + output — and the
        # output roughly doubles the masked text. Measuring only the prompt
        # understates this by about half. Overflow is right-truncated, which
        # cuts off the assistant target and trains on inputs with no labels.
        # Llama/Qwen spend roughly 2-4 subword tokens per whitespace word on
        # Devanagari/Tamil/Bengali, so 3 is used as the central estimate.
        sft_words = sorted(
            len(ex.instruction.split()) + len(ex.masked_input.split())
            + len(ex.output.split())
            for ex in examples
        )
        p95 = sft_words[int(0.95 * len(sft_words))]
        over = sum(1 for w in sft_words if w * 3 > max_seq_length)
        pct = 100.0 * over / len(sft_words)
        print(f"    {split}: {len(examples)} examples | SFT words "
              f"mean={sum(sft_words) // len(sft_words)} p95={p95} "
              f"(p95 ≈ {p95 * 3} tokens, max_seq_length={max_seq_length})")
        if pct > 5.0:
            print(f"    [error] ~{pct:.0f}% of examples exceed max_seq_length and "
                  f"would be truncated, cutting off the training target. Lower "
                  f"preprocessing.max_tokens_per_frame (roughly "
                  f"max_seq_length/16 = {max_seq_length // 16}) or raise "
                  f"model.max_seq_length.")
            ok = False

    if ok:
        print("\n  mask count matches mention count in every example ✓")
    return ok


# ---------------------------------------------------------------------------
# Stage: dryrun
# ---------------------------------------------------------------------------

def stub_predictor(seed: int = 0):
    """
    A predictor with the same interface as the model, but no model.

    It answers every mask with a cluster number drawn from a small range, which
    is enough to drive Algorithm 1 through its real code paths: repeated numbers
    create cross-frame merges, fresh numbers create new global clusters.
    Scores from this are meaningless — the point is that it runs and produces a
    well-formed clustering.
    """
    rng = random.Random(seed)

    def _predict(instruction: str, masked_input: str, n_masks: int) -> List[int]:
        # Bias towards low numbers so clusters actually get shared across frames.
        return [rng.randint(0, max(1, n_masks // 3)) for _ in range(n_masks)]

    return _predict


def gold_predictor(frame):
    """
    Predictor that replays one frame's *gold* local numbers — an upper bound.

    If Algorithm 1 and the scorer are correct, feeding gold local numbers back
    through them must reproduce the gold clustering almost exactly. A CoNLL-F
    well below 100 here means the bug is in the pipeline, not the model.
    """
    mentions = list(frame.before_mentions) + list(frame.after_mentions)
    gold_locals = [int(m["local_no"]) for m in mentions]

    def _predict(instruction: str, masked_input: str, n_masks: int) -> List[int]:
        return gold_locals

    return _predict


def stage_dryrun(cfg: dict, max_docs: Optional[int], max_frames: Optional[int],
                 output_dir: str) -> bool:
    """Run the whole inference→postprocess→score path with no LLM involved."""
    _banner("Stage 3/4 — pipeline dry run (stub + gold predictors)")

    from coref.data.dataset_builder import build_examples
    from coref.eval.evaluate import evaluate_documents, print_scores
    from coref.eval.inference import run_inference_with_predictor
    from coref.eval.postprocessor import (clusters_to_json, extract_gold_clusters,
                                          merge_clusters_over_frames,
                                          write_conll_predictions)

    docs = _load_docs(cfg, "test", max_docs)
    if not docs:
        print("  [error] no test documents.")
        return False

    instr_id = cfg["preprocessing"]["instruction_id"]
    max_tokens = cfg["preprocessing"]["max_tokens_per_frame"]
    min_ments = cfg["preprocessing"].get("min_mentions_per_example", 1)
    stub_fn = stub_predictor()

    UPPER, LOWER = "gold-local (upper bound)", "random stub (lower bound)"
    scored: Dict[str, Dict[str, list]] = {
        UPPER: {"gold": [], "pred": []},
        LOWER: {"gold": [], "pred": []},
    }

    os.makedirs(output_dir, exist_ok=True)
    n_frames_total = 0
    sample: Optional[tuple] = None
    n_gold_mentions = n_covered = n_mislinked = 0
    n_split_clusters = n_multi_clusters = 0
    per_doc: List[dict] = []

    for doc in docs:
        frames = build_examples([doc], instruction_id=instr_id,
                                max_tokens_per_frame=max_tokens,
                                min_mentions=min_ments)
        if max_frames:
            frames = frames[:max_frames]
        if not frames:
            continue
        n_frames_total += len(frames)

        gold_glob, gold_clusters = extract_gold_clusters(doc)

        # Upper bound: replay each frame's gold local numbers through Algorithm 1.
        gold_results = []
        for frame in frames:
            gold_results.extend(
                run_inference_with_predictor([frame], gold_predictor(frame))
            )
        pred_glob, pred_gold = merge_clusters_over_frames(gold_results)

        # Coverage must be measured against the sentences the processed frames
        # actually span. --max-frames truncates a document, and smaller frames
        # mean more of them, so counting against the whole document would blame
        # the frame budget for mentions this run simply never looked at.
        seen_sents = {
            s for frame in frames
            for s in list(frame.before_sent_indices) + list(frame.after_sent_indices)
        }
        in_scope = {k for k in gold_glob if k[0] in seen_sents}

        covered = in_scope & set(pred_glob)
        n_gold_mentions += len(in_scope)
        n_covered += len(covered)

        # Correctness invariant. Algorithm 1 can only relate two mentions that
        # co-occur in some frame pair — a cluster that skips a frame entirely is
        # split by design. So the check is restricted to mention pairs that DO
        # share a frame: for those, replaying gold local numbers must reproduce
        # the gold partition exactly.
        for res in gold_results:
            keys = [(m["sent_idx"], m["start_tok"], m["end_tok"])
                    for m in list(res["before_mentions"]) + list(res["after_mentions"])]
            keys = sorted({k for k in keys if k in covered})
            for i, a in enumerate(keys):
                for b in keys[i + 1:]:
                    if (gold_glob[a] == gold_glob[b]) != (pred_glob[a] == pred_glob[b]):
                        n_mislinked += 1

        # How much recall Algorithm 1's frame chaining costs on its own.
        by_gold: Dict[int, set] = {}
        for key in covered:
            by_gold.setdefault(gold_glob[key], set()).add(pred_glob[key])
        for gid, pids in by_gold.items():
            if len(gold_clusters.get(gid, ())) > 1:
                n_multi_clusters += 1
                if len(pids) > 1:
                    n_split_clusters += 1

        # Lower bound: random cluster numbers.
        _, pred_stub = merge_clusters_over_frames(
            run_inference_with_predictor(frames, stub_fn)
        )

        for label, pred in ((UPPER, pred_gold), (LOWER, pred_stub)):
            scored[label]["gold"].append(gold_clusters)
            scored[label]["pred"].append(pred)

        if sample is None:
            sample = (doc, pred_stub)

        per_doc.append({
            "doc_id":   doc.doc_id,
            "language": doc.language or "all",
            "gold":     clusters_to_json(gold_clusters),
            "pred":     clusters_to_json(pred_stub),
        })

    print(f"  {len(docs)} documents | {n_frames_total} frame pairs")

    # Writing CoNLL output is part of the path too — make sure it doesn't throw.
    if sample:
        sample_doc, sample_pred = sample
        out_path = os.path.join(output_dir, "dryrun_sample.conll")
        write_conll_predictions(
            sample_doc,
            {mpos: gid for gid, mset in sample_pred.items() for mpos in mset},
            out_path,
        )
        print(f"  wrote {out_path} ✓")

    ok = True
    for label, data in scored.items():
        print_scores(evaluate_documents(data["gold"], data["pred"]), label=label)

    # Per-language summary of the stub run, in run_inference.py's schema.
    from coref.eval.postprocessor import clusters_from_json
    by_lang: Dict[str, List[tuple]] = {}
    for rec in per_doc:
        pair = (clusters_from_json(rec["gold"]), clusters_from_json(rec["pred"]))
        by_lang.setdefault(rec["language"], []).append(pair)
        by_lang.setdefault("overall", []).append(pair)
    summary = {
        lang: evaluate_documents([g for g, _ in pairs], [p for _, p in pairs])
        for lang, pairs in by_lang.items()
    }

    # Emit the same artefacts run_inference.py produces, so `make analysis`
    # (analysis/analyse_results.py) can be exercised locally too.
    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(output_dir, "predictions.json"), "w") as fh:
        json.dump(per_doc, fh)
    print(f"  wrote {output_dir}/results.json and predictions.json "
          f"(stub predictions — for exercising analyse_results.py) ✓")

    # The upper bound does not reach 100 CoNLL-F, and that is expected: the
    # preprocessor keeps only the outermost mention of any nested group, so the
    # mentions it drops can never be recovered. Separate that known ceiling from
    # an actual linking bug.
    coverage = 100.0 * n_covered / max(n_gold_mentions, 1)
    split_rate = 100.0 * n_split_clusters / max(n_multi_clusters, 1)
    print(f"\n  Ceilings measured with perfect (gold) predictions:")
    print(f"    Mention coverage : {n_covered}/{n_gold_mentions} ({coverage:.1f}%) — "
          "the rest are nested mentions dropped at framing time.")
    print(f"    Clusters split by frame chaining: {n_split_clusters}/{n_multi_clusters} "
          f"({split_rate:.1f}%) — Algorithm 1 cannot relink a cluster that skips a frame.")
    print(f"    Mislinked pairs within a shared frame: {n_mislinked} (must be 0)")

    if n_mislinked:
        print("  [error] within a single frame, replaying gold local numbers must "
              "reproduce the gold partition exactly — the frame overlap in "
              "Algorithm 1 is wrong.")
        ok = False
    if coverage < 80.0:
        print(f"  [error] only {coverage:.1f}% of gold mentions survive framing.")
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Stage: baseline
# ---------------------------------------------------------------------------

def stage_baseline(cfg: dict, max_docs: Optional[int]) -> bool:
    """Score the model-free baselines the fine-tuned model must beat."""
    _banner("Stage 4/4 — model-free baselines")

    from analysis.baseline import BASELINES
    from coref.eval.evaluate import evaluate_documents, print_scores
    from coref.eval.postprocessor import extract_gold_clusters

    docs = [d for d in _load_docs(cfg, "test", max_docs) if len(d.mentions) >= 2]
    if not docs:
        print("  [error] no test documents with >= 2 mentions.")
        return False

    gold_list = [extract_gold_clusters(d)[1] for d in docs]
    print(f"  {len(docs)} documents")

    for name, fn in BASELINES.items():
        scores = evaluate_documents(gold_list, [fn(d) for d in docs])
        print_scores(scores, label=name)

    return True


# ---------------------------------------------------------------------------
# Stage: smoke (needs torch)
# ---------------------------------------------------------------------------

def stage_smoke(cfg: dict, model_name: Optional[str], max_frames: int,
                checkpoint: Optional[str]) -> bool:
    """Run real controlled inference on a few frames with a CPU-sized model."""
    _banner("Optional stage — CPU inference smoke test")

    if not (_have("torch") and _have("transformers")):
        print("  [skip] torch/transformers not installed.\n"
              "         pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
              "         pip install transformers")
        return True

    import torch

    from coref.data.dataset_builder import build_examples
    from coref.eval.inference import run_inference_on_examples
    from coref.eval.postprocessor import merge_clusters_over_frames

    docs = _load_docs(cfg, "test", max_docs=1)
    frames = build_examples(
        docs[:1],
        instruction_id=cfg["preprocessing"]["instruction_id"],
        max_tokens_per_frame=cfg["preprocessing"]["max_tokens_per_frame"],
        min_mentions=cfg["preprocessing"].get("min_mentions_per_example", 1),
    )[:max_frames]
    if not frames:
        print("  [error] no frames to run on.")
        return False

    if checkpoint and os.path.isdir(checkpoint):
        from coref.modeling.model import load_for_inference
        print(f"  Loading fine-tuned checkpoint {checkpoint} …")
        model, tokenizer = load_for_inference(
            checkpoint_path=checkpoint,
            max_seq_length=cfg["model"]["max_seq_length"],
            load_in_4bit=False,          # forced off on CPU anyway
            backend="standard",
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = model_name or DEFAULT_SMOKE_MODEL
        print(f"  Loading base model {name} on CPU (this downloads on first run) …")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float32, attn_implementation="eager"
        )

    model.eval()
    device = torch.device("cpu")

    # Exact prompt length — the estimate printed by the prepare stage is only a
    # rule of thumb, and an overflowing prompt is left-truncated in silence.
    from coref.data.chat_format import build_chat_text
    lengths = [
        len(tokenizer(
            build_chat_text(tokenizer, f.instruction, f.masked_input,
                            add_generation_prompt=True)
        )["input_ids"])
        for f in frames
    ]
    max_seq_length = cfg["model"]["max_seq_length"]
    print(f"  Prompt tokens: {lengths} (max_seq_length={max_seq_length})")
    if max(lengths) > max_seq_length:
        print(f"  [warn] {sum(l > max_seq_length for l in lengths)}/{len(lengths)} "
              "prompt(s) exceed max_seq_length and will be left-truncated, "
              "dropping the instruction. Reduce preprocessing.max_tokens_per_frame.")

    t0 = time.time()
    results = run_inference_on_examples(
        model, tokenizer, frames, device=device,
        max_cluster_id=cfg.get("inference", {}).get("max_cluster_id", 200),
        max_seq_length=cfg["model"]["max_seq_length"],
        verbose=True,
    )
    elapsed = time.time() - t0

    n_masks = sum(len(r["before_mentions"]) + len(r["after_mentions"]) for r in results)
    print(f"\n  {len(frames)} frame(s), {n_masks} masks in {elapsed:.1f}s "
          f"({elapsed / max(len(frames), 1):.1f}s/frame)")
    print("  NOTE: an un-fine-tuned base model predicts nonsense cluster ids. "
          "This stage checks that decoding runs, not that it is accurate.")

    preview = results[0]["output_text"]
    print(f"\n  Sample output (first 300 chars):\n  {preview[:300]!r}")

    _, clusters = merge_clusters_over_frames(results)
    print(f"  Merged into {len(clusters)} global cluster(s) ✓")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the CorefInst pipeline locally, without a GPU.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["check", "prepare", "dryrun", "baseline",
                                 "smoke", "all"])
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"config YAML (default: {DEFAULT_CONFIG})")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="cap documents per split (0 = no cap). Default: no "
                             "cap for 'check', 6 for the stages that build "
                             "frames or score.")
    parser.add_argument("--max-frames", type=int, default=4,
                        help="cap frame pairs per document (0 = no cap)")
    parser.add_argument("--model", default=None,
                        help=f"model for the smoke stage (default: {DEFAULT_SMOKE_MODEL})")
    parser.add_argument("--checkpoint", default=None,
                        help="fine-tuned checkpoint to smoke-test instead of a base model")
    parser.add_argument("--output-dir", default="local_output",
                        help="where dry-run artefacts are written")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[error] config '{args.config}' not found.")
        return 2

    cfg = load_config(args.config)
    # 'check' is the stage that certifies the corpus — duplicate doc ids and
    # out-of-range spans are exactly the faults that hide in the documents a
    # sample leaves out, so it reads everything unless told otherwise. The
    # other stages build frames or score, which is slow, so they stay sampled.
    explicit_cap = args.max_docs is not None
    check_max_docs = (args.max_docs or None) if explicit_cap else None
    max_docs = (args.max_docs or None) if explicit_cap else 6
    max_frames = args.max_frames or None

    stages = (["check", "prepare", "dryrun", "baseline"]
              if args.stage == "all" else [args.stage])

    t_start = time.time()
    failed: List[str] = []

    for stage in stages:
        if stage == "check":
            ok = stage_check(cfg, args.config, check_max_docs)
        elif stage == "prepare":
            ok = stage_prepare(cfg, max_docs)
        elif stage == "dryrun":
            ok = stage_dryrun(cfg, max_docs, max_frames, args.output_dir)
        elif stage == "baseline":
            ok = stage_baseline(cfg, max_docs)
        elif stage == "smoke":
            ok = stage_smoke(cfg, args.model, max_frames or 2, args.checkpoint)
        else:
            ok = False
        if not ok:
            failed.append(stage)

    _banner("Summary")
    for stage in stages:
        print(f"  {stage:9s} {'FAILED' if stage in failed else 'ok'}")
    print(f"\n  Total time: {time.time() - t_start:.1f}s")

    if failed:
        print("\n  Fix the failures above before spending GPU time.")
        return 1

    print("\n  Local checks passed. Next, on a GPU:\n"
          "    make prepare CONFIG=configs/a100.yaml\n"
          "    make train   CONFIG=configs/a100.yaml\n"
          "    make infer   CONFIG=configs/a100.yaml\n"
          "  (configs/t4.yaml fits a free-tier T4 but its 64-token frames cost\n"
          "   accuracy and make a full 3-epoch run too long — see that file.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

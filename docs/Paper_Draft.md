# Instruction-Tuned Coreference Resolution for Hindi, Tamil and Bengali

**A replication of CorefInst on the TransMuCoRes corpus**

Pradnesh Fernandez A

---

> **Status of this draft (2026-08-20).** The pipeline is complete and validated
> end to end, and every number in Sections 3, 5.3 and 6.1–6.3 is measured. The
> fine-tuned model has **not been trained yet** — no GPU run has taken place.
> Section 6.4, which reports system scores, is therefore an empty table with a
> fixed protocol rather than results. Nothing in this draft may be presented as
> a system result until that table is filled from a real run. Earlier versions
> of the project documentation contained illustrative placeholder scores; those
> have been removed (see Section 8.3).

---

## Abstract

Coreference resolution for Indian languages remains under-served relative to
English and the European languages that dominate CorefUD. We replicate
**CorefInst** (Arslan et al., 2026) — which recasts coreference as constrained
autoregressive generation over an instruction-tuned decoder-only LLM — on the
Hindi, Tamil and Bengali portions of **TransMuCoRes** (Sharma et al., 2024).
Documents are cut into overlapping sentence frames; every mention is marked
`<m>…</m>#MASK`; a QLoRA-tuned Llama 3.1 8B predicts one cluster number per
mask under controlled, token-by-token decoding; and Algorithm 1 of the paper
stitches per-frame numbering into a document-level clustering.

This draft reports the parts of that system that can be established without a
GPU: corpus statistics, three model-free baselines on the full test split, and
— the contribution we would emphasise — a **measured ceiling on the pipeline
itself**. Replaying gold cluster numbers through framing, merging and scoring
yields CoNLL-F 93.04, not 100: the framing step drops 8.0% of gold mentions,
and frame chaining splits 6.0% of multi-mention clusters. Any system score must
be read against that ceiling rather than against 100, and against an MFE
surface-matching baseline that already reaches CoNLL-F 53.34.

---

## 1. Introduction

Coreference resolution — deciding which mentions in a document refer to the
same entity — has been dominated by encoder-based span-ranking models. CorefInst
showed that a decoder-only LLM, instruction-tuned with QLoRA and decoded under
constraint, is competitive with those systems across CorefUD v1.2.

CorefUD is overwhelmingly Indo-European and European. Hindi, Tamil and Bengali
differ from that population in ways that plausibly matter for the approach:
Tamil is agglutinative, all three are pro-drop, and all three are written in
scripts that modern BPE tokenizers fragment far more aggressively than Latin
script. Whether the CorefInst recipe transfers is an empirical question, and it
is the question this project sets out to answer.

Our contributions in this draft are:

1. A complete, tested re-implementation of the CorefInst pipeline for three
   Indic languages, validated end to end on CPU (Section 5.3).
2. Measured corpus statistics for the Hindi/Tamil/Bengali portions of
   TransMuCoRes as actually distributed (Section 3).
3. Three model-free baselines on the full 1,118-document test split
   (Section 6.2).
4. **A ceiling analysis of the CorefInst formulation itself** (Section 6.3) —
   what the pipeline can achieve with perfect predictions — which we argue
   should accompany any reported score in this framework.
5. Two corpus-handling faults found by that analysis, with fixes
   (Sections 4.2 and 8.2).

## 2. Related Work

**CorefInst** (Arslan et al., 2026) is the method under replication: instruction
tuning of decoder-only LLMs for coreference, with five instruction variants, of
which Instruction #5 performed best; QLoRA at rank 16; and controlled inference
that decodes one cluster number per mention rather than free-generating the
whole answer.

**CorPipe** (Straka, 2024) represents the encoder-based line of work and won
the CRAC 2024 shared task, predicting zero mentions and coreference jointly.

**TransMuCoRes** (Sharma et al., 2024) is the corpus: coreference annotations
for Indian languages produced by translating and projecting English sources
(OntoNotes, LitBank) plus a natively-annotated Hindi news corpus (Mujadia).
Its provenance matters for interpretation and we return to it in Section 7.2.

## 3. Data

### 3.1 Composition

The corpus as distributed contains three sources, all filtered to the language
codes `hin_Deva`, `tam_Taml` and `ben_Beng`:

| Source | Languages | Origin |
|---|---|---|
| `mujadia_conll` | Hindi only | Natively annotated Hindi news |
| `onto_notes_archive` | all three | Translation-projected OntoNotes |
| `litbank_*` | all three | Translation-projected LitBank |

### 3.2 Measured statistics

All figures below are counted from the distributed files by
`scripts/run_local.py check`, after the span-deduplication fix of Section 4.2.

| Split | Documents | hi / ta / bn | Sentences | Mentions | Gold clusters |
|---|---:|---|---:|---:|---:|
| train | 8,748 | 3,063 / 2,842 / 2,843 | 193,805 | 458,042 | 120,885 |
| dev | 1,083 | 379 / 352 / 352 | 23,762 | 57,133 | 15,805 |
| test | 1,125 | 394 / 365 / 366 | 26,890 | 67,143 | 17,761 |

Document identifiers are unique across all three languages (verified over the
full test split), so the language-specific translations of one source document
never collide.

### 3.3 Zero mentions are absent from this corpus

CorefInst devotes part of its instruction set to zero mentions — dropped
pronouns, marked `</z>@MASK` — and our pipeline implements and unit-tests that
path. **The distributed corpus contains none.** Counting over all three splits
returns 0 zero mentions out of 582,318, and the source files contain no trace
tokens (`*PRO*`, `*T*-1` or equivalent) from which they could be recovered.

This is a consequence of provenance: the projected sources annotate overt
mentions only. The consequence for this work is that **no claim about
zero-mention or pro-drop resolution can be supported by these experiments**,
notwithstanding that pro-drop is a genuine feature of all three languages. The
capability exists in the code and is dormant in the data. Testing it would
require a corpus that annotates zero anaphora.

## 4. Method

### 4.1 Framing, masking and merging

We follow CorefInst. A document is split into sentence frames under a token
budget (`max_tokens_per_frame`); each training example is a *pair* of adjacent
frames joined by a `[MID]` separator, so consecutive examples overlap in one
frame. Within an example, every mention is wrapped `<m>…</m>#MASK` and the
target replaces each `#MASK` with a cluster number local to that example.

At inference, decoding is *controlled*: the prompt is prefilled once, and for
each mask in turn the model emits digits which are fed back before the next
segment of text is appended to the KV cache. The model therefore conditions on
its own earlier cluster assignments, and cannot produce a malformed answer.

Algorithm 1 then maps local numbers to global ones: because example *k* and
example *k+1* share a frame, the mentions in that shared frame anchor the two
local numberings to each other.

### 4.2 Two corpus-handling faults, and their fixes

Both were found by the gold-replay diagnostic of Section 6.3 at full corpus
scale; both are invisible on a small sample.

**Spans annotated for two clusters.** CoNLL permits `(3|(4` … `3)|4)`, marking
one span as a member of two chains at once. This occurs on 438 of 67,984 test
mentions (0.64%), in 31% of test documents. A span is identified downstream by
its position key `(sent_idx, start_tok, end_tok)`, and the task asks for exactly
one number per mask, so a two-cluster span is representable neither in gold nor
in any possible prediction. Left implicit, gold and predicted clusterings each
retained whichever copy their own dict construction saw last, and could
disagree — the diagnostic reported 9,823 mention pairs as mislinked that had in
fact been linked correctly. We resolve it once at parse time: lowest cluster id
wins.

**Frame-budget/sequence-length coupling.** One training example is
instruction + masked input + output, and the output roughly doubles the masked
text. For Indic scripts, a whitespace word costs 2–4 subword tokens, so the SFT
string runs to ≈16× the frame budget. A frame budget chosen without reference
to `max_seq_length` causes right-truncation that removes the assistant target,
training the model on inputs with no labels. We keep
`max_tokens_per_frame ≈ max_seq_length / 16` and verify empirically
(Section 5.2).

## 5. Experimental Setup

### 5.1 Model and tuning

| | |
|---|---|
| Base model | Llama 3.1 8B Instruct, NF4 4-bit (double quant) |
| Adapter | QLoRA, rank 16, alpha 16, dropout 0.0 |
| Target modules | q, k, v, o, gate, up, down projections |
| Instruction | #5 (best in CorefInst Table 1) |
| Optimizer | adamw_8bit, lr 2e-4, cosine, warmup 3% |
| Weight decay / clip | 0.01 / 1.0 |
| Effective batch | 16, 3 epochs |
| Loss | assistant answer tokens only |
| Seed | 42 |

At 13,761 training examples and effective batch 16, a full run is **≈2,580
optimizer steps**.

Loss is masked to the assistant answer. This is worth stating explicitly
because the library function that used to provide it was removed upstream, and
the failure mode — computing loss over the prompt as well — degrades the model
while leaving the loss curve looking entirely normal. We mask in-repo and
unit-test the masking.

### 5.2 Sequence budget

Measured over all 13,761 training examples, at 3 subword tokens per whitespace
word:

| `max_seq_length` | Frame budget | Examples truncated |
|---:|---:|---:|
| 4096 | 256 | **0.9%** |
| 2048 | 128 | (scaled budget, comparable) |
| 1024 | 64 | (scaled budget, comparable) |

SFT string length: mean 738 words, p95 1,085, max 1,921. The A100/H100 preset
(4096/256) is therefore adequately sized; the T4 preset requires a 64-token
frame budget, which fragments clusters far more aggressively and is not
recommended for a headline run.

### 5.3 Pre-GPU validation

Everything except fine-tuning runs without a GPU, and we run it before booking
GPU time. `make test` executes 36 unit tests (parsing, Algorithm 1, scorer,
loss masking, resume logic). `make local` runs four stages against the real
corpus: environment and data check over every document, frame construction with
a 1:1 mask/mention-record assertion, an end-to-end dry run driven by a stub
predictor, and the baselines. This is what surfaced both faults in Section 4.2.

### 5.4 Evaluation

MUC, B³ and CEAFe, with CoNLL-F their unweighted mean.

**Comparability caveat.** Our scorer is an independent implementation, not the
official CorefUD scorer (`coval`/`corefud-scorer`), and it retains singletons,
which the shared-task setup treats differently. Its numbers are internally
consistent — valid for comparing our systems, baselines and configurations
against each other — but they are **not** directly comparable with published
CorefUD or CorefInst figures. Cross-corpus comparison requires re-scoring the
emitted CoNLL files with the official tool, which we have not yet done.

## 6. Results

### 6.1 What is and is not measured

| Result | Status |
|---|---|
| Corpus statistics (§3.2) | Measured |
| Model-free baselines (§6.2) | Measured, full test split |
| Pipeline ceiling (§6.3) | Measured, full test split |
| Fine-tuned system scores (§6.4) | **Not yet run** |

### 6.2 Model-free baselines

Full test split, restricted to the 1,118 documents with ≥2 mentions. CoNLL-F:

| Baseline | Hindi | Tamil | Bengali | Overall |
|---|---:|---:|---:|---:|
| All-singletons | 22.78 | 23.75 | 22.68 | **23.05** |
| All-one-cluster | 39.17 | 38.63 | 39.17 | **39.00** |
| Most-frequent-entity (MFE) | 55.58 | 49.91 | 54.13 | **53.34** |

Component scores for MFE overall: MUC 59.25, B³ 54.73, CEAFe 46.04.

MFE is greedy surface-form head-word matching with no model of any kind, and it
reaches 53.34. **This is the number the fine-tuned system has to beat**, and it
is a demanding one: high precision (MUC-P 75.8) against modest recall is exactly
what a repeated-string heuristic achieves on news and literary prose. A random
stub predictor scores 31.60 for reference.

Tamil is the hardest language for MFE by ~5 points, consistent with
agglutination defeating surface-form matching — though whether the neural
system shows the same ordering is precisely what remains to be measured.

### 6.3 The pipeline's own ceiling

We replay *gold* cluster numbers through the full path — framing, controlled
inference interface, Algorithm 1, scorer — and score the result. A correct
pipeline given perfect predictions should approach 100. It does not:

| Metric | P | R | F |
|---|---:|---:|---:|
| MUC | 100.00 | 88.46 | 93.87 |
| B³ | 100.00 | 86.07 | 92.51 |
| CEAFe | 91.84 | 93.67 | 92.74 |
| **CoNLL** | 97.28 | 89.40 | **93.04** |

Two structural losses account for the gap, both inherent to the formulation
rather than to our implementation of it:

1. **Mention coverage 92.0%** (61,783 / 67,130). The framing step keeps only
   the outermost mention of a nested group, so 8.0% of gold mentions are never
   presented to the model and can never be predicted. Precision is unaffected
   (100.00 for MUC and B³); the entire loss is recall.
2. **Frame chaining splits 6.0% of multi-mention clusters** (746 / 12,422).
   Algorithm 1 can only relate mentions that co-occur in some frame pair, so an
   entity that disappears for longer than a frame and returns is split into two
   clusters.

A third quantity, mislinked pairs within a shared frame, must be exactly 0 —
that is the correctness invariant for Algorithm 1, and it holds (it did not
before the fix in Section 4.2).

**We would stress this as the most useful result available before training.**
A system reported at, say, CoNLL-F 70 in this framework has captured 75% of
what the framing permits, not 70% of what is achievable. Reporting against 100
understates such a system, and — more importantly — misattributes framing loss
to the model.

### 6.4 Fine-tuned system — protocol, not results

Not yet run. The protocol is fixed in advance:

| Language | Docs | MUC-F | B³-F | CEAFe-F | CoNLL-F | vs. MFE | vs. ceiling |
|---|---:|---|---|---|---|---|---|
| Hindi | 391 | — | — | — | — | — | — |
| Tamil | 363 | — | — | — | — | — | — |
| Bengali | 364 | — | — | — | — | — | — |
| **Overall** | **1,118** | — | — | — | — | — | — |

Reporting commitments, recorded here so they cannot be chosen after seeing the
numbers:

- Scores come from `inference_output/results.json`, written by
  `scripts/run_inference.py`; per-document clusterings are retained.
- The comparison points are **MFE at 53.34** and **the ceiling at 93.04**, both
  in the same table.
- The scorer caveat of Section 5.4 is restated wherever the numbers appear.
- Any comparison with published CorefInst/CorefUD figures requires the official
  scorer first, or is not made.

Inference cost, for planning: 2,006 test frames carrying 86,663 masks, each
decoded token-by-token at batch size 1 — on the order of 2.6×10⁵ sequential
forward passes.

## 7. Discussion

### 7.1 What the ceiling implies for the method

The 8.0% mention-coverage loss is a design property of frame-based masking, not
a bug: nested mentions cannot both be wrapped as the outermost `<m>` span. A
system that needs those mentions — and 8% is a large fraction of the headroom
between MFE and a strong result — would need a different marking scheme. This
seems to us the most substantive limitation of the CorefInst formulation as
applied to this corpus, and it is not visible unless the ceiling is measured.

### 7.2 Corpus provenance

Two thirds of the corpus is translation-projected from English sources. Scores
on it measure resolution over translated text, which need not behave like
natively-authored Hindi, Tamil or Bengali — projected annotations inherit the
referential structure of the English original, including its mention density
and its lack of pro-drop. The Mujadia Hindi subset (220/27/28 documents) is
natively annotated and is the only part free of this concern; a per-source
breakdown of the final results would be worth reporting for that reason.

### 7.3 Limitations

- No system results yet (§6.4).
- No zero-mention evaluation is possible on this corpus (§3.3).
- Scores are not comparable to published CorefUD numbers without re-scoring
  (§5.4).
- Single seed, single base model. CorefInst compares Llama, Gemma and Mistral
  backbones; we have budgeted for one.
- Wall-clock training and inference times are unmeasured, as no GPU run has
  occurred.

## 8. Reproducibility

### 8.1 Running it

```bash
make test                              # 36 unit tests, no GPU
make local                             # full CPU validation against the corpus
make prepare CONFIG=configs/a100.yaml  # 13,761 / 1,758 / 2,006 frame examples
make train   CONFIG=configs/a100.yaml  # ≈2,580 steps; resumes from checkpoints
make infer   CONFIG=configs/a100.yaml  # resumable; --max-docs N for a smoke run
make baseline && make analysis
```

### 8.2 Faults found and fixed during validation

Recorded because each was silent, and each would have corrupted a GPU run:

| Fault | Effect if unfixed |
|---|---|
| Spans annotated for two clusters | 9,823 spurious mislinks; split rate overstated 8.2% vs 6.0% |
| Removed upstream loss-masking collator | Loss computed over the prompt; normal-looking loss curve |
| Inference results written only at the end | A dropped session loses the entire multi-hour run |
| Unquoted pip version pins in the notebooks | Shell redirect; version floors silently ignored |
| A100 preset at batch 8 × seq 4096 | Does not fit 40 GB |
| Corpus check sampled 6 documents by default | The two faults above were invisible at that scale |

### 8.3 Provenance of numbers in earlier drafts

Earlier project documentation contained a table of system scores
(CoNLL-F 72.85 average; Hindi 74.21, Bengali 72.84, Tamil 71.50) attributed to
this project, and a results file reporting CoNLL-F 61.34. No model had been
trained at any point. The results file was traced to the **MFE baseline output**
written to the model's results path — its overall CoNLL-F 53.22 reproduces to
within 0.01 as this project's MFE baseline. The system table appears to have
been a projection. All of it has been removed. This section exists so that the
removal is on the record rather than silent.

## References

- Arslan, P., Erol, E., and Eryiğit, G. (2026). *CorefInst: Leveraging LLMs for
  Multilingual Coreference Resolution.* TACL.
- Straka, M. (2024). *CorPipe at CRAC 2024: Predicting Zero Mentions and
  Coreference at Once.* CRAC 2024 Shared Task.
- Sharma, R., et al. (2024). *TransMuCoRes: A Multilingual Coreference
  Resolution Dataset for Indian Languages.*
- Vilain, M., et al. (1995). *A model-theoretic coreference scoring scheme.* MUC-6.
- Bagga, A., and Baldwin, B. (1998). *Algorithms for scoring coreference chains.* LREC.
- Luo, X. (2005). *On coreference resolution performance metrics.* HLT-EMNLP.

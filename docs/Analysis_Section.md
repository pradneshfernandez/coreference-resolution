# 4. Analysis and Findings

> **Status.** The fine-tuned model has not been trained yet. This section
> reports what has been measured without a GPU: the baselines, and the ceiling
> the pipeline imposes on any model running inside it. Section 4.3 states the
> reporting protocol for the system run and holds an empty results table.

## 4.1 Evaluation Methodology

Performance is reported as precision, recall and F1 under the three standard
coreference metrics, with CoNLL-F1 as the headline figure:

1. **MUC** — link-based, counts partitions of gold clusters induced by predictions
2. **B³** — mention-based cluster overlap
3. **CEAFe** — entity-based optimal alignment (φ₄)
4. **CoNLL-F1** — unweighted mean of the three F1 scores

**Comparability caveat.** The scorer in `coref/eval/evaluate.py` is an
independent implementation, not the official CorefUD scorer
(`coval`/`corefud-scorer`), and it keeps singletons, which the shared task
handles differently. Its numbers are internally consistent and valid for
comparing our own systems, baselines and configurations against each other.
They are **not** interchangeable with published CorefUD or CorefInst figures.
Making that comparison requires re-scoring the CoNLL files emitted by
`write_conll_predictions` with the official tool.

## 4.2 Comparative Baselines — measured

Three model-free baselines, run by `analysis/baseline.py` over the **full test
split** (1,118 documents with ≥2 mentions). These are measured numbers, not
estimates.

| Baseline | Strategy | Hindi | Tamil | Bengali | **Overall CoNLL-F** |
|---|---|---:|---:|---:|---:|
| All-singletons | every mention its own cluster | 22.78 | 23.75 | 22.68 | **23.05** |
| All-one-cluster | one cluster per document | 39.17 | 38.63 | 39.17 | **39.00** |
| **MFE** | greedy head-word surface matching | 55.58 | 49.91 | 54.13 | **53.34** |

MFE component scores (overall): MUC 59.25, B³ 54.73, CEAFe 46.04. A random stub
predictor scores 31.60 for reference.

**MFE at 53.34 is the bar.** A surface-form heuristic with no model reaches it,
because repeated proper nouns in news and literary prose are genuinely
informative. Tamil is ~5 points harder for MFE than Hindi, consistent with
agglutination breaking surface matching; whether the neural system reproduces
that ordering is one of the questions the run will answer.

## 4.3 Results: fine-tuned system — protocol

Not yet run. The table below is fixed in advance so that reporting choices are
not made after seeing the numbers.

| Language | Docs | MUC-F | B³-F | CEAFe-F | CoNLL-F | vs. MFE | vs. ceiling |
|---|---:|---|---|---|---|---|---|
| Hindi | 391 | — | — | — | — | — | — |
| Tamil | 363 | — | — | — | — | — | — |
| Bengali | 364 | — | — | — | — | — | — |
| **Overall** | **1,118** | — | — | — | — | — | — |

Commitments:

- Numbers come from `inference_output/results.json` (written by
  `scripts/run_inference.py`), with per-document clusterings retained.
- **MFE (53.34)** and **the ceiling (93.04)** appear in the same table as the
  system score, always.
- The scorer caveat of Section 4.1 is restated wherever the numbers appear.
- No comparison against published CorefInst/CorefUD figures without re-scoring
  through the official scorer.

## 4.4 The pipeline's own ceiling — measured

Replaying **gold** cluster numbers through framing → Algorithm 1 → scorer tests
what the pipeline could achieve with a perfect model. It does not reach 100:

| Metric | P | R | F |
|---|---:|---:|---:|
| MUC | 100.00 | 88.46 | 93.87 |
| B³ | 100.00 | 86.07 | 92.51 |
| CEAFe | 91.84 | 93.67 | 92.74 |
| **CoNLL** | 97.28 | 89.40 | **93.04** |

Measured over 1,115 test documents. Two structural losses explain the gap:

1. **Mention coverage 92.0%** (61,783 / 67,130). Framing keeps only the
   outermost mention of a nested group, so 8.0% of gold mentions are never
   shown to the model. Precision is untouched — the whole loss is recall.
2. **Frame chaining splits 6.0% of multi-mention clusters** (746 / 12,422).
   Algorithm 1 links only through shared frames, so an entity absent for longer
   than one frame and then reappearing becomes two clusters.

A third quantity — mislinked pairs among mentions sharing a frame — is the
correctness invariant for Algorithm 1 and must be exactly 0. It is (see 4.6).

**Why this matters for interpretation.** A system scoring CoNLL-F 70 here has
captured roughly 75% of what the framing permits, not 70% of what is
achievable. Scoring against 100 charges framing loss to the model.

## 4.5 Zero mentions: implemented, but absent from this corpus

The CorefInst instruction set covers zero mentions (dropped pronouns, marked
`</z>@MASK`), and our preprocessor and tests implement that path.

**The distributed corpus contains none.** Counting across all three splits
gives 0 zero mentions out of 582,318, and the source files contain no trace
tokens from which they could be recovered — the projected OntoNotes/LitBank
sources annotate overt mentions only.

All three languages are genuinely pro-drop, so this is a property of the corpus
rather than of the languages. The consequence is firm: **no claim about
zero-mention or pro-drop resolution can be supported by these experiments.**
The capability is present in the code and dormant in the data. Evaluating it
needs a corpus that annotates zero anaphora.

## 4.6 Spans annotated for more than one cluster

CoNLL permits a span to be marked as a member of two chains at once
(`(3|(4` … `3)|4)`). This occurs on 438 of 67,984 test mentions (0.64%), spread
across 31% of test documents.

A span is identified throughout the pipeline by its position key
`(sent_idx, start_tok, end_tok)`, and the task asks the model for exactly one
number per `#MASK`, so a two-cluster span is representable neither in the gold
structures nor in any prediction the model can make. The parser therefore keeps
one annotation per span, chosen deterministically: the lowest cluster id wins.

Resolving it in one place matters because gold and predicted clusterings are
built separately. If each were left to keep whichever copy it happened to see
last, the two could disagree about a span, and mentions that had been linked
correctly would score as errors.

The correctness invariant for Algorithm 1 — that mentions sharing a frame are
linked exactly as gold links them when gold numbers are replayed — is checked
on every `make local` run and currently holds at 0 violations.

## 4.7 References

- Arslan, P., Erol, E., and Eryiğit, G. (2026). *CorefInst: Leveraging LLMs for
  Multilingual Coreference Resolution.* TACL. *(Method under replication.)*
- Straka, M. (2024). *CorPipe at CRAC 2024: Predicting Zero Mentions and
  Coreference at Once.* CRAC 2024 Shared Task. *(Encoder-based comparison point.)*
- Sharma, R., et al. (2024). *TransMuCoRes: A Multilingual Coreference
  Resolution Dataset for Indian Languages.* *(Source corpus.)*
- Vilain, M., et al. (1995). *A model-theoretic coreference scoring scheme.* MUC-6.
- Bagga, A., and Baldwin, B. (1998). *Algorithms for scoring coreference chains.* LREC.
- Luo, X. (2005). *On coreference resolution performance metrics.* HLT-EMNLP.

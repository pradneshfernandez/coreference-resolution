# CorefInst — Instruction-Tuned Coreference Resolution for Indian Languages

Implementation of **CorefInst** (TACL 2026) for Hindi, Tamil, and Bengali using the [TransMuCoRes](https://github.com/transmucores) dataset. The model fine-tunes a decoder-only LLM (Llama 3.1 8B) with QLoRA to predict coreference cluster IDs via controlled, token-by-token inference.

---

## Project status

**The model has not been trained yet.** The pipeline is complete and validated
end to end on CPU; no GPU run has taken place, so there are no system results.

| | |
|---|---|
| Pipeline | Complete — 40 unit tests, full-corpus CPU validation passing |
| Data | Prepared: 13,761 / 1,758 / 2,006 train/dev/test frame examples |
| Baselines | Measured on the full test split — MFE **CoNLL-F 53.34** |
| Pipeline ceiling | Measured — **CoNLL-F 93.04** with perfect predictions |
| Fine-tuned model | **Not run** |

Read `docs/Paper_Draft.md` for the measured numbers, the protocol fixed for the
run, and the limitations. Two points carry over into any result:

- Scores come from an in-repo scorer, not the official CorefUD scorer, and are
  not directly comparable with published CorefInst/CorefUD figures.
- The corpus contains **no zero mentions** (0 of 582,318), so nothing about
  pro-drop resolution can be claimed from it, though the code supports it.

---

## Project Structure

```
.
├── coref/
│   ├── data/
│   │   ├── conll_parser.py     # Parse TransMuCoRes CoNLL files → Document objects
│   │   ├── preprocessor.py     # Convert Documents → FrameExample training instances
│   │   ├── dataset_builder.py  # Load all 3 languages, build HuggingFace Datasets
│   │   └── chat_format.py      # One chat-template path shared by train + inference
│   ├── modeling/
│   │   ├── model.py            # Load LLM + QLoRA (auto-detects unsloth vs PEFT)
│   │   ├── collator.py         # Completion-only loss masking (assistant tokens only)
│   │   └── train.py            # SFT fine-tuning with TRL's SFTTrainer
│   └── eval/
│       ├── inference.py        # Controlled inference engine (Section 3.3 of paper)
│       ├── postprocessor.py    # Algorithm 1 — merge clusters across frames
│       └── evaluate.py         # CoNLL metrics: MUC, B³, CEAFe
│
├── scripts/
│   ├── prepare_data.py       # Step 1 — build JSONL datasets from raw CoNLL
│   ├── train_model.py        # Step 2 — fine-tune the model
│   ├── run_inference.py      # Step 3 — run inference and evaluate (resumable)
│   ├── benchmark_inference.py# Measure decode throughput before a long run
│   └── run_local.py          # CPU-only pipeline validation (no GPU needed)
│
├── analysis/
│   ├── baseline.py           # All-singletons, all-one-cluster, MFE baselines
│   └── analyse_results.py    # Score tables, error analysis, ablation comparison
│
├── tests/                    # Unit tests — pure Python, no torch, no GPU
│
├── configs/
│   ├── t4.yaml               # Colab free tier (T4, 16 GB VRAM)
│   ├── a100.yaml             # Colab Pro / A100 (40/80 GB VRAM)
│   ├── h100.yaml / l4.yaml   # Other GPU presets
│   └── cpu.yaml              # CPU-only (Qwen 1.5B, no quantization)
│
├── docs/                     # Paper draft, analysis, implementation notes
├── notebooks/                # Colab notebooks (all-in-one + step-by-step docs)
├── Makefile                  # Shortcuts for every step below
├── config.yaml               # Default configuration (A100 / local GPU)
├── pyproject.toml            # pip install -e .
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### Option A — Google Colab (recommended)

1. Open `CorefInst_Colab.ipynb` directly in Colab from the GitHub repository.
2. Clone the repository in the notebook's first cell to set up the environment:
   ```bash
   !git clone https://github.com/pradneshfernandez/InCoRes.git
   %cd InCoRes
   !pip install -r requirements.txt
   !pip install "unsloth[colab-new]"
   ```
3. Upload `transmucores_data.tar.gz` to your Colab workspace or mount Google Drive if stored there.
4. Execute the notebook cells. The system auto-detects the GPU (T4 or A100) and applies the appropriate hardware preset.

---

### Option B — Local, no GPU

Everything except fine-tuning runs on a laptop. Do this **before** booking GPU
time: it validates parsing, framing, Algorithm 1, and the CoNLL scorer against
the real dataset, so a bug shows up here instead of halfway through a GPU session.

Only `PyYAML`, `numpy`, and `scipy` are needed — not torch.

```bash
make test          # unit tests
make local         # env + data check → framing → pipeline dry run → baselines
```

`make local` runs four stages (see `python scripts/run_local.py --help`):

| Stage | What it proves | Needs torch |
|---|---|---|
| `check` | The dataset parses; no duplicate doc ids; every mention span is in range | no |
| `prepare` | Frames build; masks and mention records correspond 1:1; prompts fit `max_seq_length` | no |
| `dryrun` | Inference → Algorithm 1 → CoNLL scoring runs end to end, driven by a stub predictor instead of an LLM | no |
| `baseline` | Scores all-singletons / all-one-cluster / MFE — the numbers the model has to beat | no |
| `smoke` | Real controlled inference on a few frames with a small CPU model | **yes** |

The `dryrun` stage also reports two ceilings that no model can exceed, measured
by replaying the *gold* cluster numbers through the pipeline:

- **Mention coverage** — the preprocessor keeps only the outermost mention of a
  nested group, so the rest can never be predicted.
- **Clusters split by frame chaining** — Algorithm 1 links clusters only through
  overlapping frames, so a cluster that skips a frame is split in two.

For an actual CPU inference run (slow, and meaningless before fine-tuning, but
it exercises the decoding path for real):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers
make local-smoke
```

---

### Option C — Local / Cluster with a GPU

**Requirements:** Python 3.10+, CUDA GPU with ≥16 GB VRAM (for 8B model, 4-bit).

#### 1. Install dependencies

```bash
pip install -r requirements.txt
# Recommended: also install unsloth for faster training
pip install "unsloth[colab-new]"
```

#### 2. Prepare data

```bash
make prepare CONFIG=config.yaml          # python scripts/prepare_data.py --config …
```

Reads raw CoNLL files from `transmucores_data/`, writes `processed_data/train.jsonl`, `dev.jsonl`, `test.jsonl`.

#### 3. Train

```bash
# Full training (default config — expects A100 / high-VRAM GPU)
make train CONFIG=config.yaml

# T4 Colab preset
make train CONFIG=configs/t4.yaml

# Few-shot (50 examples per language — useful for quick experiments)
make train-few CONFIG=configs/t4.yaml N=50
```

The fine-tuned LoRA adapter is saved to `model_output/final/`.

#### 4. Run inference and evaluate

```bash
make infer CONFIG=config.yaml
```

Prints per-language CoNLL scores (MUC, B³, CEAFe, CoNLL-F) and saves
`inference_output/results.json` plus `predictions.json` (per-document clusters).

#### 5. Analyse results

```bash
# Score table, cluster-size distribution, and error analysis
make analysis

# Run baselines for comparison
make baseline CONFIG=config.yaml
```

---

## Hardware Presets

| Config | GPU | VRAM | Seq len | Frame | Batch | Precision |
|---|---|---|---|---|---|---|
| `config.yaml` | A100 | 40/80 GB | 4096 | 256 | 4 × 4 | bf16 |
| `configs/h100.yaml` | H100 | 80 GB | 4096 | 256 | 8 × 2 | bf16 |
| `configs/a100.yaml` | A100 | 40/80 GB | 4096 | 256 | 4 × 4 | bf16 |
| `configs/l4.yaml` | L4 | 24 GB | 2048 | 128 | 4 × 4 | bf16 |
| `configs/t4.yaml` | T4 | 16 GB | 1024 | 64 | 2 × 8 | fp16 |
| `configs/cpu.yaml` | CPU | — | 1024 | 64 | 1 × 16 | fp32 |

Batch is `per_device × gradient_accumulation` (effective 16 everywhere). All GPU
presets load Llama 3.1 8B in 4-bit; `cpu.yaml` uses Qwen2.5-1.5B unquantized,
since bitsandbytes needs CUDA.

**Frame** is `preprocessing.max_tokens_per_frame`, and it is tied to `Seq len` —
one training example is `instruction + masked_input + output`, and the output
roughly doubles the masked text, so the whole string is ≈ 16 × the frame budget
in subword tokens for Indic scripts. Overshoot is right-truncated, which cuts
off the training target. Keep `max_tokens_per_frame ≈ max_seq_length / 16` and
verify with `python scripts/run_local.py prepare --config <cfg>`.

Smaller frames also fragment more clusters: on a 6-doc sample, a 256-token frame
splits 0% of multi-mention clusters, a 64-token frame splits ~31%. **Prefer L4 or
A100 for the real run** — the T4 preset fits in 16 GB but pays for it in accuracy,
and its ~58k examples make 3 epochs too long for a free-tier session.

---

## Model and Approach

CorefInst frames coreference resolution as a **structured generation** task:

- **Input**: a pair of text frames (windows of sentences), each mention marked with `<m>…</m>#MASK` (the model must predict the cluster number at each `#MASK`).
- **Output**: one integer per mention, predicting which coreference cluster it belongs to.
- **Inference**: controlled generation — the model predicts each cluster number one at a time, conditioning on all previous predictions.
- **Cross-frame merging**: Algorithm 1 from the paper maps per-frame local cluster numbers to a global document-level cluster assignment.

Default instruction set: **Instruction #5** (best-performing in Table 1 of the paper).

---

## Baselines

Three simple baselines are provided for comparison:

| Baseline | Description |
|---|---|
| All-singletons | Every mention is its own cluster (lower bound) |
| All-one-cluster | All mentions in a document share one cluster (upper bound on recall) |
| MFE | Greedy surface-form matching — same head word → same cluster |

Run with:
```bash
make baseline CONFIG=config.yaml      # or: python analysis/baseline.py --config … --split test
```

---

## CoNLL Evaluation Metrics

Scores are reported as F1 (harmonic mean of precision and recall):

- **MUC** — mention-based link overlap
- **B³** — mention-based cluster overlap
- **CEAFe** — entity-based alignment (phi_4)
- **CoNLL-F** — average of MUC-F, B³-F, CEAFe-F

---

## Dependencies

| Purpose | Packages |
|---|---|
| Data, scoring, `make test`, `make local` | `PyYAML`, `numpy`, `scipy` |
| Inference | + `torch`, `transformers` (CPU wheels are fine) |
| Training | + `datasets`, `peft`, `trl`, `accelerate` |
| 4-bit QLoRA (GPU only) | + `bitsandbytes` |

Optional (strongly recommended for Colab): `unsloth` — enables faster QLoRA
training and Flash Attention 2. It requires CUDA, so the code falls back to the
standard HF + PEFT backend automatically on CPU.

---

## Citation

If you use this code, please cite the CorefInst paper (TACL 2026) and the TransMuCoRes dataset.

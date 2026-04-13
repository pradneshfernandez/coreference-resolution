# CorefInst — Instruction-Tuned Coreference Resolution for Indian Languages

Implementation of **CorefInst** (TACL 2026) for Hindi, Tamil, and Bengali using the [TransMuCoRes](https://github.com/transmucores) dataset. The model fine-tunes a decoder-only LLM (Llama 3.1 8B) with QLoRA to predict coreference cluster IDs via controlled, token-by-token inference.

---

## Project Structure

```
.
├── src/
│   ├── conll_parser.py       # Parse TransMuCoRes CoNLL files → Document objects
│   ├── preprocessor.py       # Convert Documents → FrameExample training instances
│   ├── dataset_builder.py    # Load all 3 languages, build HuggingFace Datasets
│   ├── model.py              # Load LLM + QLoRA (auto-detects unsloth vs PEFT)
│   ├── train.py              # SFT fine-tuning with TRL's SFTTrainer
│   ├── inference.py          # Controlled inference engine (Section 3.3 of paper)
│   ├── postprocessor.py      # Algorithm 1 — merge clusters across frames
│   └── evaluate.py           # CoNLL metrics: MUC, B³, CEAFe
│
├── analysis/
│   ├── baseline.py           # All-singletons, all-one-cluster, MFE baselines
│   └── analyse_results.py    # Score tables, error analysis, ablation comparison
│
├── configs/
│   ├── t4.yaml               # Colab free tier (T4, 16 GB VRAM)
│   ├── a100.yaml             # Colab Pro / A100 (40/80 GB VRAM)
│   └── cpu.yaml              # CPU-only (Qwen 1.5B, no quantization)
│
├── prepare_data.py           # Step 1 — build JSONL datasets from raw CoNLL
├── train_model.py            # Step 2 — fine-tune the model
├── run_inference.py          # Step 3 — run inference and evaluate
├── CorefInst_Colab.ipynb     # All-in-one Colab notebook (recommended)
├── DOCS.ipynb                # Step-by-step documentation notebook
├── package_for_colab.py      # Zip code for Google Drive upload
├── config.yaml               # Default configuration (A100 / local GPU)
├── setup.py                  # pip install -e . (for Colab)
└── requirements.txt          # Python dependencies
```

---

## Quick Start

### Option A — Google Colab (recommended)

1. Open `CorefInst_Colab.ipynb` directly in Colab from the GitHub repository.
2. Clone the repository in the notebook's first cell to set up the environment:
   ```bash
   !git clone https://github.com/pradneshfernandez/coreference-resolution.git
   %cd coreference-resolution
   !pip install -r requirements.txt
   !pip install "unsloth[colab-new]"
   ```
3. Upload `transmucores_data.tar.gz` to your Colab workspace or mount Google Drive if stored there.
4. Execute the notebook cells. The system auto-detects the GPU (T4 or A100) and applies the appropriate hardware preset.

---

### Option B — Local / Cluster

**Requirements:** Python 3.10+, CUDA GPU with ≥16 GB VRAM (for 8B model, 4-bit).

#### 1. Install dependencies

```bash
pip install -r requirements.txt
# Recommended: also install unsloth for faster training
pip install "unsloth[colab-new]"
```

#### 2. Prepare data

```bash
python prepare_data.py --config config.yaml
```

Reads raw CoNLL files from `transmucores_data/`, writes `processed_data/train.jsonl`, `dev.jsonl`, `test.jsonl`.

#### 3. Train

```bash
# Full training (default config — expects A100 / high-VRAM GPU)
python train_model.py --config config.yaml

# T4 Colab preset
python train_model.py --config configs/t4.yaml

# Few-shot (50 examples per language — useful for quick experiments)
python train_model.py --config configs/t4.yaml --few_shot 50
```

The fine-tuned LoRA adapter is saved to `model_output/final/`.

#### 4. Run inference and evaluate

```bash
python run_inference.py --config config.yaml
```

Prints per-language CoNLL scores (MUC, B³, CEAFe, CoNLL-F) and saves `inference_output/results.json`.

#### 5. Analyse results

```bash
# Detailed analysis of a single run
python analysis/analyse_results.py --results_json inference_output/results.json

# Run baselines for comparison
python analysis/baseline.py --config config.yaml --split test
```

---

## Hardware Presets

| Config | GPU | VRAM | Seq len | Batch | Precision |
|---|---|---|---|---|---|
| `config.yaml` | A100 | 40/80 GB | 4096 | 4 | bf16 |
| `configs/t4.yaml` | T4 | 16 GB | 2048 | 2 | fp16 |
| `configs/a100.yaml` | A100 | 40/80 GB | 4096 | 4 | bf16 |
| `configs/cpu.yaml` | CPU | — | 512 | 1 | fp32 |

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
python analysis/baseline.py --config config.yaml --split test
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

Core: `torch`, `transformers`, `datasets`, `peft`, `trl`, `bitsandbytes`, `accelerate`, `scipy`, `PyYAML`

Optional (strongly recommended for Colab): `unsloth` — enables faster QLoRA training and Flash Attention 2.

---

## Citation

If you use this code, please cite the CorefInst paper (TACL 2026) and the TransMuCoRes dataset.

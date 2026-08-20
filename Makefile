.PHONY: prepare train train-few infer baseline analysis test local local-check local-smoke clean

# Interpreter. Many distros ship only `python3`; override with `make PYTHON=python`.
PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)

CONFIG ?= config.yaml
# Few-shot sample size; override with `make train-few N=100`
N ?= 50
# Config used by the CPU-only targets
LOCAL_CONFIG ?= configs/cpu.yaml

## Prepare train/dev/test JSONL from raw CoNLL files
prepare:
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

## Fine-tune the model (set CONFIG=configs/t4.yaml for T4)
train:
	$(PYTHON) scripts/train_model.py --config $(CONFIG)

## Few-shot training (N=50 examples per language by default)
train-few:
	$(PYTHON) scripts/train_model.py --config $(CONFIG) --few_shot $(N)

## Run inference and print CoNLL scores
infer:
	$(PYTHON) scripts/run_inference.py --config $(CONFIG)

## Run baselines (all-singletons, all-one-cluster, MFE)
baseline:
	$(PYTHON) analysis/baseline.py --config $(CONFIG) --split test

## Deep analysis of inference results
analysis:
	$(PYTHON) analysis/analyse_results.py --results_json inference_output/results.json

## Full pipeline: prepare → train → infer → baseline → analysis
all: prepare train infer baseline analysis

# ------------------------------------------------------------------
# CPU-only targets — run these on a laptop before booking GPU time.
# ------------------------------------------------------------------

## Unit tests (no torch, no GPU)
test:
	$(PYTHON) -m unittest discover -s tests -v

## Full local validation: env + data check, framing, dry run, baselines
local:
	$(PYTHON) scripts/run_local.py all --config $(LOCAL_CONFIG)

## Just the environment and dataset sanity check
local-check:
	$(PYTHON) scripts/run_local.py check --config $(LOCAL_CONFIG)

## Real CPU inference on a couple of frames with a small model (needs torch)
local-smoke:
	$(PYTHON) scripts/run_local.py smoke --config $(LOCAL_CONFIG)

## Remove generated outputs (keeps model checkpoints)
clean:
	rm -rf processed_data/ inference_output/ local_output/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

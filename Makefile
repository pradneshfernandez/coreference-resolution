.PHONY: prepare train infer baseline analysis clean

CONFIG ?= config.yaml

## Prepare train/dev/test JSONL from raw CoNLL files
prepare:
	python scripts/prepare_data.py --config $(CONFIG)

## Fine-tune the model (set CONFIG=configs/t4.yaml for T4)
train:
	python scripts/train_model.py --config $(CONFIG)

## Few-shot training (N=50 examples per language by default)
train-few:
	python scripts/train_model.py --config $(CONFIG) --few_shot $(N)

## Run inference and print CoNLL scores
infer:
	python scripts/run_inference.py --config $(CONFIG)

## Run baselines (all-singletons, all-one-cluster, MFE)
baseline:
	python analysis/baseline.py --config $(CONFIG) --split test

## Deep analysis of inference results
analysis:
	python analysis/analyse_results.py --results_json inference_output/results.json

## Full pipeline: prepare → train → infer → baseline → analysis
all: prepare train infer baseline analysis

## Remove generated outputs (keeps model checkpoints)
clean:
	rm -rf processed_data/ inference_output/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

"""
scripts/benchmark_inference.py — Measure raw generation throughput.

Not a unit test (it used to live at the repo root as test_inference.py, where
unittest/pytest would try to collect it and fail on the missing GPU). It is a
manual benchmark: how long one short generation takes on the current hardware,
which sets the floor for how long a full controlled-inference run will take.

Usage:
  python scripts/benchmark_inference.py [--model NAME] [--iters 5] [--words 200]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from coref.modeling.model import cuda_available, load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark generation speed.")
    parser.add_argument("--model", default=None,
                        help="model id (default: the project default on GPU, "
                             "Qwen2.5-0.5B-Instruct on CPU)")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--words", type=int, default=200,
                        help="prompt length, in repetitions of a short phrase")
    args = parser.parse_args()

    on_gpu = cuda_available()
    model_name = args.model or (
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" if on_gpu
        else "Qwen/Qwen2.5-0.5B-Instruct"
    )
    print(f"Device: {'cuda' if on_gpu else 'cpu'} | model: {model_name}")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        load_in_4bit=on_gpu,           # bitsandbytes is CUDA-only
    )
    model.eval()

    if on_gpu:
        try:
            from unsloth import FastLanguageModel  # type: ignore
            FastLanguageModel.for_inference(model)
        except ImportError:
            pass

    device = next(model.parameters()).device
    prompt = "Hello, testing inference speed! " * args.words
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(f"Prompt: {inputs['input_ids'].shape[1]} tokens")

    print("Warming up …")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)

    print(f"Running {args.iters} iterations …")
    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.iters):
            model.generate(**inputs, max_new_tokens=4, do_sample=False)
    elapsed = time.time() - t0

    print(f"Time per generation: {elapsed / args.iters:.3f} s")


if __name__ == "__main__":
    main()

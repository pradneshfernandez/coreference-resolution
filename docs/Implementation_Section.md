# 3. Implementation Details

## 3.1 Systems Architecture Overview
The project is built as a complete, end-to-end Python pipeline engineered for maximum efficiency on GPU clusters. The system uses a sequence-to-sequence structure, where Coreference Resolution is framed strictly as an autoregressive generation task rather than a traditional classification loop. The architecture comprises data parsing, generation-based model training, controlled inference execution, and post-processing.

## 3.2 Data Preprocessing and Input Masking (`conll_parser.py` & `preprocessor.py`)
Raw CoNLL formatted files from the TransMuCoRes dataset are first parsed into `Document` and `FrameExample` objects.
During this phase, critical token tagging occurs:
- **Overt Mentions**: wrapped as `<m>…</m>#MASK`; the target replaces each `#MASK` with the cluster number.
- **Zero Mentions**: marked `</z>@MASK`, with the target writing `</z>#<number>`. This path is implemented and unit-tested, but the distributed corpus contains **no zero mentions at all** (0 of 582,318) — see `Analysis_Section.md` §4.5. No zero-mention result can be claimed from these experiments.

A span that the corpus annotates for two clusters at once (`(3|(4` … `3)|4)`, 0.64% of test mentions) is reduced to one annotation at parse time, lowest cluster id winning: a mask admits exactly one predicted number, so the second annotation is not representable.
The documents are then chunked into overlapping *frames* to bypass context-length limitations of LLMs while preserving the narrative flow of large text artifacts.

## 3.3 The Model Stack (`model.py` & `train.py`)
The foundational model operates explicitly via **Llama 3.1 8B Instruct**, which is optimized utilizing `unsloth` for accelerated training logic and Flash Attention 2. 
- The model is loaded in 4-bit precision. 
- A LoRA adapter (rank 16, alpha 16, dropout 0.0) is injected into the attention and MLP projections (q, k, v, o, gate, up, down) to learn the clustering semantics without disturbing the base weights.
- Training is orchestrated via Hugging Face’s `SFTTrainer` (Supervised Fine-Tuning) and `trl` libraries.
- Loss is computed on the assistant answer only, using the in-repo collator in `coref/modeling/collator.py`. TRL's own implementation was removed upstream, and falling back to full-sequence loss trains the model to regenerate the instruction while leaving the loss curve looking normal.
- One SFT example is instruction + masked input + output, and the output roughly doubles the masked text; on Indic scripts this comes to ≈16× the frame budget in subword tokens. `max_tokens_per_frame` is therefore kept near `max_seq_length / 16`, verified with `scripts/run_local.py prepare`.

## 3.4 Controlled Inference Engine (`inference.py`)
Because causal Language Models exhibit a tendency for hallucination if unconstrained, the project employs a **Controlled Inference** loop (Section 3.3 of the CorefInst methodology). 
Instead of requesting the LLM to rewrite the entire clustered text, the pipeline identifies the precise location of the `#MASK` placeholders. The generation constraint guarantees that the model iteratively predicts a specific cluster ID at the current active token. The decoded cluster ID is then fed securely into the context block for the next masked generation step, preventing syntax-drifting over long passages.

## 3.5 Cross-Frame Postprocessing (`postprocessor.py`)
Because inference occurs sequentially across bounded frames, the LLM naturally predicts *local* cluster integers relative to current context. To resolve document-level identities, we implement **Algorithm 1**, which iterates through consecutive frames containing textual overlap. A mapping dictionary coalesces predicted local cluster IDs against the anchored document space, producing *global* cluster assignments written back out in CoNLL format.

Two limits of this step are measured rather than assumed (`Analysis_Section.md` §4.4): framing presents only 92.0% of gold mentions to the model, and chaining splits 6.0% of multi-mention clusters. Together they cap the pipeline at CoNLL-F 93.04 even with perfect predictions.

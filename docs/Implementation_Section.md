# 3. Implementation Details

## 3.1 Systems Architecture Overview
The project is built as a complete, end-to-end Python pipeline engineered for maximum efficiency on GPU clusters. The system uses a sequence-to-sequence structure, where Coreference Resolution is framed strictly as an autoregressive generation task rather than a traditional classification loop. The architecture comprises data parsing, generation-based model training, controlled inference execution, and post-processing.

## 3.2 Data Preprocessing and Input Masking (`conll_parser.py` & `preprocessor.py`)
Raw CoNLL formatted files from the TransMuCoRes dataset are first parsed into `Document` and `FrameExample` objects.
During this phase, critical token tagging occurs:
- **Overt Mentions**: Wrapped with `<m>` and `</m>` tags, appended with `#MASK#`.
- **Zero Mentions**: Automatically tagged with `</z>` and appended with `#MASK@` at the inferred syntactic drop-locations.
The documents are then chunked into overlapping *frames* to bypass context-length limitations of LLMs while preserving the narrative flow of large text artifacts.

## 3.3 The Model Stack (`model.py` & `train.py`)
The foundational model operates explicitly via **Llama 3.1 8B Instruct**, which is optimized utilizing `unsloth` for accelerated training logic and Flash Attention 2. 
- The model is loaded in 4-bit precision. 
- A LoRA adapter (Rank=16, Alpha=32) is injected into the attention modules to learn the clustering semantics without disrupting the base language knowledge.
- Training is orchestrated via Hugging Face’s `SFTTrainer` (Supervised Fine-Tuning) and `trl` libraries.

## 3.4 Controlled Inference Engine (`inference.py`)
Because causal Language Models exhibit a tendency for hallucination if unconstrained, the project employs a **Controlled Inference** loop (Section 3.3 of the CorefInst methodology). 
Instead of requesting the LLM to rewrite the entire clustered text, the pipeline identifies the precise location of the `#MASK` placeholders. The generation constraint guarantees that the model iteratively predicts a specific cluster ID at the current active token. The decoded cluster ID is then fed securely into the context block for the next masked generation step, preventing syntax-drifting over long passages.

## 3.5 Cross-Frame Postprocessing (`postprocessor.py`)
Because inference occurs sequentially across bounded frames, the LLM naturally predicts *local* cluster integers relative to current context. To resolve document-level identities, we implement **Algorithm 1**, which iterates through consecutive frames containing textual overlap. A mapping dictionary dynamically coalesces predicted local cluster IDs against the anchored document space, resulting in accurate *global* cluster assignments seamlessly outputted back into CoNLL format.

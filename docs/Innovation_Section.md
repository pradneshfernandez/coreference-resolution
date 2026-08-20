# 2. Innovation

## 2.1 Extending the CorefInst Paradigm to Low-Resource Indian Languages
The primary innovation of this project lies in adapting the state-of-the-art CorefInst architecture—which uses generative decoder-only Large Language Models (LLMs)—specifically for Indian languages (Hindi, Tamil, and Bengali). While foundational LLM-based coreference models have succeeded on high-resource, Indo-European linguistic families via the CorefUD dataset, applying these techniques to the morphologically rich and diverse landscapes of the TransMuCoRes dataset introduces novel challenges. Whether the recipe transfers is the empirical question this project sets out to answer; it has not yet been answered, because the fine-tuning run has not taken place (see `Analysis_Section.md`).

## 2.2 Instruction Engineering for Morphological Complexity
Indian languages present unique linguistic features, such as extensive pro-drop (frequent omission of subjects or objects), complex gender morphology, and agglutinative properties (especially in Tamil). This project designs custom, instruction-tuned prompt formulations that allow the LLM (Llama 3.1 8B) to track textual flow, gender variations, and syntactic dependencies over long passages to accurately infer missing referential entities.

## 2.3 Zero-Mention Contextual Resolution
Indian languages frequently drop pronouns where context supplies the referent, and traditional models fail silently on these hidden targets. The pipeline marks such positions `</z>@MASK` during preprocessing, so the LLM must emit a cluster number for them, and token-by-token decoding lets it condition on entities introduced earlier.

**This capability is currently untested.** The distributed TransMuCoRes files contain no zero mentions — 0 of 582,318 mentions across all three splits, with no trace tokens from which they could be recovered, because the projected OntoNotes and LitBank sources annotate overt mentions only. The code path is implemented and unit-tested against synthetic examples; the corpus does not exercise it. Demonstrating zero-mention resolution would require a corpus that annotates zero anaphora, and no claim about it can be made from these experiments.

## 2.4 Resource-Efficient, Quantized Fine-Tuning 
Modern LLMs demand severe hardware requirements, which limits accessibility in resource-constrained research environments. By utilizing **QLoRA (Quantized Low-Rank Adaptation)** and a 4-bit quantization framework (orchestrated via Unsloth), we reduce the memory footprint significantly. This puts fine-tuning of an 8-billion-parameter model within reach of a single A100 or L4 instance, while retaining the generalization of the base weights. The T4 preset fits in 16 GB but only at a 64-token frame budget, which fragments clusters substantially, so it is intended for smoke runs rather than a headline result.

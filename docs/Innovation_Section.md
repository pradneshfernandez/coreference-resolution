# 2. Innovation

## 2.1 Extending the CorefInst Paradigm to Low-Resource Indian Languages
The primary innovation of this project lies in adapting the state-of-the-art CorefInst architecture—which uses generative decoder-only Large Language Models (LLMs)—specifically for Indian languages (Hindi, Tamil, and Bengali). While foundational LLM-based coreference models have succeeded on high-resource, Indo-European linguistic families via the CorefUD dataset, applying these techniques to the morphologically rich and diverse landscapes of the TransMuCoRes dataset introduces novel challenges that this project successfully mitigates.

## 2.2 Instruction Engineering for Morphological Complexity
Indian languages present unique linguistic features, such as extensive pro-drop (frequent omission of subjects or objects), complex gender morphology, and agglutinative properties (especially in Tamil). This project designs custom, instruction-tuned prompt formulations that allow the LLM (Llama 3.1 8B) to track textual flow, gender variations, and syntactic dependencies over long passages to accurately infer missing referential entities.

## 2.3 Zero-Mention Contextual Resolution
A major innovative aspect of our approach is the explicit and robust handling of *zero mentions*. Because Indian languages frequently drop pronouns when the context implicitly provides the referent, traditional models fail silently on these hidden targets. By using a specially designed `#MASK@` token placeholder inserted syntax-deep into the input matrix during preprocessing, the LLM is forced to output a global cluster ID. The token-by-token generative nature allows it to cross-reference historical entities without relying on explicitly visible structural embeddings.

## 2.4 Resource-Efficient, Quantized Fine-Tuning 
Modern LLMs demand severe hardware requirements, which limits accessibility in resource-constrained research environments. By utilizing **QLoRA (Quantized Low-Rank Adaptation)** and a 4-bit quantization framework (orchestrated via Unsloth), we reduce the memory footprint significantly. This enables full fine-tuning of an 8 Billion parameter foundational model on consumer-grade hardware (like a Google Colab T4 GPU or single A100 instance), while retaining the immense generalization power of the base neural weights.

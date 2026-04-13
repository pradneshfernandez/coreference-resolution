# 1. Literature Survey

## 1.1 Introduction to Coreference Resolution (CR)
Coreference Resolution is a semantic-level natural language processing task that involves identifying and clustering mentions within a text that refer to the same real-world entity. An end-to-end CR system typically performs two primary stages: (1) Mention Detection, which extracts all potential referential expressions, and (2) Coreference Linking, which groups these mentions into coreferential clusters. Accurate CR is fundamental for downstream NLP applications requiring high-level language understanding, such as machine translation, text summarization, and question-answering systems.

## 1.2 Evolution of Coreference Resolution Models
Historically, early neural CR approaches relied on task-specific architectures built upon encoder-only (e.g., BERT) or encoder-decoder language models to generate contextualized word embeddings. These models originally focused heavily on the linking stage before transitioning to end-to-end methods that perform mention detection and linking jointly or sequentially (Lee et al., 2018; Joshi et al., 2020; Straka, 2024).

Despite their strong performance, task-specific architectures have notable limitations. They demand specialized network designs, vast amounts of training data, and extensive training from scratch. This leads to restricted generalizability—and high adaptation costs—when applying these models to new domains, tasks, or languages.

## 1.3 The LLM Paradigm and Instruction Tuning
As Large Language Models (LLMs) comprising mostly decoder-only architectures (e.g., Llama, GPT, Mistral) have revolutionized NLP, their application to CR has become a prominent area of research. Early attempts evaluated these LLMs using zero-shot or few-shot prompting constraints for CR, demonstrating that while LLMs possessed structural knowledge, their out-of-the-box CR accuracy lagged without explicit fine-tuning (Brown et al., 2020; Gan et al., 2024).

Recent frameworks, such as the CorefInst paradigm (Arslan et al., 2026), have addressed this gap through *Instruction Tuning*. By framing CR as a structured generation task via carefully engineered prompts, decoder-only LLMs can be fine-tuned to perform mention clustering sequentially, using token-by-token generation. Instruction-tuned LLMs represent a clear shift from inflexible baseline models toward general-purpose foundational models that adapt to complex cognitive tasks via tailored instructions.

## 1.4 Challenges in Multilingual CR and Zero Mentions
The vast majority of earlier CR research concentrated almost exclusively on English, aided by comprehensive annotated datasets (e.g., OntoNotes). Multilingual CR has progressed more recently, championed by initiatives like the CorefUD collection, which harmonize coreference corpora across diverse languages into a universal scheme. 

A specific architectural challenge in multilingual CR is the handling of **zero mentions**—dropped pronouns intrinsic to pro-drop languages. While overt mentions are explicitly tokenized in strings, zero mentions are syntactically omitted, meaning a coreference system must infer a latent relation. While recent models (such as CorPipe24) integrate zero mention prediction directly, Instruction-tuned LLMs introduce mechanisms to systematically resolve zero mentions by tracking textual flow and coherence without relying strictly on structural embeddings.

## 1.5 Context for Indian Languages
Building on these advancements, expanding robust instruction-tuned CR frameworks (like CorefInst) to Indian languages (e.g., Hindi, Tamil, Bengali) addresses a crucial gap in NLP research. Datasets such as TransMuCoRes provide the critical infrastructure to fine-tune robust foundational models via QLoRA. This allows parameters to be updated intelligently to capture both standard entity references and complex anaphoric phenomena (such as dropped pronouns and gender morphology variations) inherent in morphologically rich Indian languages.

## 1.6 Datasets for Coreference Resolution
High-quality coreference corpora are essential for training both task-specific architectures and LLMs:
- **TransMuCoRes**: A newly developed multilingual coreference dataset specifically tailored for instruction-tuning on Indian Languages (Hindi, Bengali, Tamil). It maps directly into our project’s pipeline and encapsulates zero mentions correctly.
- **CorefUD v1.2** (Popel et al., 2024): A harmonized collection of 21 datasets spanning 15 languages, formatted uniformly in CoNLL-U. It supports pro-drop languages and serves as the primary benchmark dataset for modern multilingual benchmarking tools.
- **OntoNotes 5.0** (Pradhan et al., 2012): The historical standard English-dominated dataset that framed most of the early neural CR architectural research, though limited by strict licensing guidelines in certain contexts.

## 1.7 References
1. Pamay Arslan, T., Erol, E., and Eryiğit, G. (2026). *CorefInst: Leveraging LLMs for Multilingual Coreference Resolution*. Transactions of the Association for Computational Linguistics (TACL).
2. Barzilay, R., and Lapata, M. (2008). *Modeling local coherence: An entity-based approach*. Computational Linguistics, 34(1):1–34.
3. Bhattacharjee, S., Haque, R., Wenniger, G., and Way, A. (2020). *Investigating query expansion and coreference resolution in question answering on BERT*. International Conference on Applications of Natural Language to Information Systems.
4. Bohnet, B., Alberti, C., and Collins, M. (2023). *Coreference resolution through a seq2seq transition-based system*. Transactions of the Association for Computational Linguistics (TACL), 11:212–226.
5. Brown, T., Mann, B., Ryder, N., et al. (2020). *Language models are few-shot learners*. Advances in Neural Information Processing Systems (NeurIPS), 33:1877–1901.
6. Chen, S., Gu, B., Qu, J., Li, Z., Liu, A., Zhao, L., and Chen, Z. (2021). *Tackling zero pronoun resolution and non-zero coreference resolution jointly*. Proceedings of the 25th Conference on Computational Natural Language Learning (CoNLL).
7. Clark, K., and Manning, C. D. (2016). *Improving coreference resolution by learning entity-level distributed representations*. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL).
8. De Marneffe, M. C., Manning, C. D., Nivre, J., and Zeman, D. (2021). *Universal dependencies*. Computational Linguistics, 47(2):255–308.
9. D’Oosterlinck, K., Bitew, S. K., Papineau, B., et al. (2023). *CAW-coref: Conjunction-aware word-level coreference resolution*. The Sixth Workshop on Computational Models of Reference, Anaphora and Coreference (CRAC).
10. Gan, Y., Poesio, M., and Yu, J. (2024). *Assessing the capabilities of large language models in coreference: An evaluation*. LREC-COLING 2024.
11. Joshi, M., Chen, D., Liu, Y., Weld, D. S., Zettlemoyer, L., and Levy, O. (2020). *SpanBERT: Improving pre-training by representing and predicting spans*. Transactions of the Association for Computational Linguistics (TACL), 8:64–77.

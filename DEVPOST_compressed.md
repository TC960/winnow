Winnow multi-path prompt compression for LLM pipelines
 Tokens meter LLM bill, waste.
 Spoken language, retrieved RAG context, long documents
 redundant redundancy different places. *low-information
 tokens, *irrelevant passages
, *bit-width KV cache
. No single paper kills three.
 instead one compression technique, built Winnow
 pipeline runs several state-art compressors, merges
 decisions, hands result to LLM
 cache compressed. Every stage justified by recent
 paper contribution *compose.
 include every, **parallelizable patchable**
 compression method no public,
 re-implemented paper, LLMLingua
 usable code. Winnow builds on
 **LLMLingua-2** Data Distillation for Efficient Faithful Task-Agnostic Prompt Compression [arXiv:2403. 12968. 12968).
 **LongLLMLingua** Accelerating Enhancing LLMs in Long Context Scenarios via Prompt Compression [arXiv:2310. 06839. 06839)./LLMLingua
-Guided Context Pruning Retrieval-Augmented Generation [arXiv:2503. 10720./abs/2503. 10720) public repo self-implemented
 End-to-End Context Compression Scale [arXiv:2606. 09659./abs/2606. 09659./LeonLixyz/LCLM
 Online Vector Quantization Near-Optimal Distortion Rate-cache compression. 19874./abs/2504. 19874)./OmarHory/turboquant
 **CompactPrompt** Unified Pipeline Prompt Data Compression LLM Workflows [arXiv:2510. 18043./abs/2510. 18043) no public dataset
 Winnow compresses prompt two axes user choose
 push.
. Token-space compression.
, two compressor families
 **LLMLingua-2 LongLLMLingua. extractive token classifier
. LongLLMLingua
 query-aware arm BGE cross-encoder reranker, question-aware
 document selection reordering,
 causal perplexity path compresses survivors question.
 **AttentionRAG.-Guided Context Pruning
 RAG (arXiv:2503. 10720). reformulates query incomplete-answer
 template single "focal blank, runs next-token prediction over
 context chunk, uses focal token attention
 rank tokens sentences top-k.
 answers dropped.
 arms produce per-token decisions. token-by-token
 over original, preserving,
 picks **intersection**
 or,.
 merge canonical token sequence (LLMLingua-2
, string-membership test, fallback LLMLingua-only
 AttentionRAG gates text.
. Model-space compression answer generated.
 compressed string, user chooses
. compressed prompt hosted API Anthropic
 Claude OpenAI GPT,. savings immediate
 provider-agnostic, compressed prompt text.
-hosted TurboQuant. Route Qwen model, generating
 **TurboQuant**-compressed KV cache Research, ICLR 2026,
 arXiv:2504. 19874) data-free random-rotation quantizer drops KV
 cache to ~4 bits (3–4× smaller `DynamicCache subclass, validated
 Mistral-7B Qwen2.5-14B.
-hosted TurboQuant. Route-to-End Context
 Compression Scale, latent-context encoder→adapter→decoder
 checkpoints, weeks old build. LCLM compresses context
 latent decoder consumes
 shrinking KV cache length. TurboQuant
 compresses per entry decoder cache.
 orthogonal, savings fewer KV entries fewer bits.
 token-level pruning, sequence-length
 bit-width compression three independent papers stacked end to end.
 workers Modal GPUs. LLMLingua-2/LongLLMLingua, AttentionRAG,
 TurboQuant,+TurboQuant run Modal A100 worker,
 FastAPI server lifecycle load startup
 no cold-start cost. LLMLingua AttentionRAG
 routes Qwen-TurboQuant-TurboQuant
 flag.
 merge. Python normalizes
 LLMLingua labels, reconstructs char-span
, tests AttentionRAG sentence-spans overlap,
 applies union/intersection rule, splices survivors
 preserving original spacing.
 drop-in `DynamicCache subclass Lloyd-Max Gaussian
quantization optional outlier-channel path slots
 HF model_key_values, LCLM decoder.
 Next. js app shows raw. compressed diff, token
 counts, % $ saved, A/B Q&A box same question
 prompts.
 Challenges
 **Pairing TurboQuant. low-bit. 5-bit
 outlier quantization decoder
. LCLM feeds decoder `inputs_embeds soft tokens, not
 ids, confirm `TQCache survives path encoder
 single prefill without latent memory block.
 merge algorithm. Unionizing/intersecting
 compressors not methods segment text
 differently. align canonical token sequence operate
 char-spans merge order-preserving unambiguous repeated
 words, fallback empty AttentionRAG result erases.
 LongLLMLingua. Choosing right lead anchors causal backbone
 reranker question-aware arm iteration,
 reorder context position bias, causal path
 beats extractive classifier.
 **CompactPrompt no public data. tried implement
 style-metric approach, shipped no public datasets fit
, couldn't reproduce left out
 unvalidated.
Accomplishments proud
 working compressor token-space (LLMLingua AttentionRAG
 model-space (LCLM TurboQuant stacked pipeline.
 novel between heterogeneous compression
 order-preserving destroys text.
 TurboQuant validated end-to-end (3–4× KV compression, output FP16)
 with KV savings multiply.
 big wins from compressing axes stacking
 token count, sequence length, bit-width independent.
 Heterogeneous methods need common representation
,.
 reproducing paper gated by artifacts no public dataset
 no reimplementation.
 Learned routing between intersection/union
 black-box. self-hosted,.
 Custom CUDA kernels TurboQuant memory win latency win.
 Revisiting CompactPrompt evaluation data available.
 Built
 Modal FastAPI. LLMLingua-2 LongLLMLingua AttentionRAG
. 10720)-to Context Compression Scale TurboQuant
. 19874) BGE rerankers OpenAI GPT
 Deepgram

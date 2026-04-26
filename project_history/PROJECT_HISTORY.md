# NLP Project: Curriculum Learning

## Full Project History — Phase-by-Phase

> **Model**: 124M parameter GPT-style Decoder-Only Transformer  
> **GPU**: NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)  
> **Framework**: PyTorch 2.x + HuggingFace Transformers  
> **Tokenizer**: GPT-2 BPE (50,257 vocab, shared across ALL phases — never changed)

---

## Project Overview

This project builds a large language model (124M parameters) from scratch using **Curriculum Learning** — a training strategy that presents data in order of increasing complexity, inspired by how humans learn. The model progresses from simple children's stories → formal encyclopedic text → diverse web/reasoning data.

**Final pipeline**:
```
TinyStories → Wikipedia → Mixed Corpus
  Phase 1       Phase 2      Phase 3
```

---

## Architecture Design Decisions

### Why Decoder-Only (GPT-style)?

We chose a **decoder-only** Transformer rather than the full encoder-decoder architecture (T5, BART) or encoder-only (BERT). Reasons:

1. **Task fit**: We are doing causal language modeling (next-token prediction) — exactly what decoder-only is designed for. Encoder-decoder would require target sequences at inference time.
2. **Parameter efficiency**: A decoder-only model has NO cross-attention layers → no wasted parameters. PyTorch's `nn.TransformerDecoderLayer` has 3 sub-layers (self-attn → cross-attn → FFN), but cross-attention is meaningless without an encoder. We built custom `GPTBlock` (self-attn + FFN only) to avoid this waste.
3. **Scaling laws**: Decoder-only models (GPT-3, LLaMA, Mistral) have been shown to scale better for generation tasks.

### Architecture Parameters (fixed across ALL phases)

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `d_model` | 768 | GPT-2 Medium size — fits 16GB VRAM with batch_size=16 + fp16 |
| `n_heads` | 12 | Standard for 768 dim (head_dim = 64) |
| `n_layers` | 12 | GPT-2 Medium depth |
| `d_ff` | 3072 | 4× d_model (GPT-2 standard FF expansion) |
| `max_seq_len` | 512 | Memory-compute tradeoff — 1024 would halve batch size |
| Attention | Causal (masked) | `is_causal=True` in `F.scaled_dot_product_attention` |
| Weight tying | Yes | `lm_head.weight = token_emb.weight` → saves ~38M params |
| Positional emb | Learned | GPT-2 style (not RoPE/ALiBi — simpler for fixed-length) |
| Normalization | Pre-LayerNorm | Better training stability at scale vs Post-LN |

**Total trainable parameters: ~124M** (weight tying is NOT double-counted)

---

## Phase 1: TinyStories Pre-Training

### Goal
Teach the model basic grammar, sentence structure, narrative patterns, and simple vocabulary using children's short stories.

### Dataset: TinyStories
- ~2.12 million short children's stories
- ~475 million tokens total
- Average story ≈ 224 tokens
- Vocabulary: simple, child-friendly
- **Why chosen**: Easiest possible distribution — model learns syntax before semantics. Classic curriculum learning first step.

### Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Learning Rate | 3e-4 | Standard GPT-2 LR for pre-training |
| LR Min | 3e-5 | Cosine decay floor |
| Batch Size | 16 | Fits RTX 4060 Ti with fp16 |
| Grad Accumulation | 8 | Effective batch = 128 |
| Warmup Steps | 2,000 | Model starts from random init — needs longer warmup |
| Epochs | 3 | Full passes over ~475M tokens |
| Dropout | 0.1 | Standard regularization |
| Weight Decay | 0.1 | AdamW standard |

### Results

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 1.7124 | 1.3411 | 3.82 |
| 2 | 1.2817 | 1.2359 | 3.44 |
| **3** | **1.1877** | **1.1878** | **3.28** ✅ |

- **Final Val PPL: 3.28** — excellent! The model learned basic language structure.
- Low perplexity on simple data was expected; validates architecture correctness.
- Train/Val loss converged together (no overfitting) — good regularization.

---

## Phase 2: Wikipedia Pre-Training (Curriculum Step)

### Goal
Transfer Phase 1 language skills to formal, knowledge-rich text. The model adapts to much higher vocabulary diversity and factual content.

### Why This is a Curriculum Step
- Phase 1 → Phase 2 represents a **massive domain shift**: children's stories → encyclopedic text
- Wikipedia has 10× longer documents, 5× larger vocabulary, no story structure
- By warm-starting from Phase 1, the model keeps its language skills and only needs to adapt to new domain — much faster than training from scratch

### Dataset: Wikipedia EN
- English Wikipedia snapshot (November 2023)
- **1 billion tokens** used for training (~4.0 GB binary int32 file)
- **10 million tokens** for validation (~40 MB)
- Streamed and tokenized on-the-fly, saved as memory-mapped binary

### Hyperparameter Changes from Phase 1

| Parameter | Phase 1 | Phase 2 | Reason |
|-----------|---------|---------|--------|
| LR | 3e-4 | **1e-4** | 3× lower — model already trained, prevent forgetting |
| LR Min | 3e-5 | **1e-5** | Same 10× ratio |
| Warmup Steps | 2,000 | **1,000** | Half — model is no longer starting from scratch |

### Results

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 3.6598 | 3.2047 | 24.65 |
| 2 | 3.1639 | 3.0623 | 21.38 |
| **3** | **3.0690** | **3.0188** | **20.47** ✅ |

- The jump from ~3.28 PPL to ~20+ PPL when domain shifted shows the **curriculum necessity**: without Phase 1 warm-start, training from scratch on Wikipedia would take much longer.

---

## Phase 3: Mixed Corpus Pre-Training (Curriculum Step)

### Goal
Generalize the model beyond Wikipedia to diverse web text, Q&A reasoning, and long-form books. Build a strong general-purpose pre-trained model.

### Dataset Composition (~2 billion tokens)

| Source | Tokens | % | Reasoning |
|--------|--------|---|-----------|
| **OpenWebText** | 700M | 35% | High-quality web prose |
| **C4** | 600M | 30% | Filtered Common Crawl |
| **StackExchange** | 500M | 25% | Q&A reasoning chains |
| **Books** | 200M | 10% | Long-form continuity |

### Hyperparameter Changes from Phase 2

| Parameter | Phase 2 | Phase 3 | Reason |
|-----------|---------|---------|--------|
| LR | 1e-4 | **5e-5** | 2× lower — even more careful |
| LR Min | 1e-5 | **5e-6** | Same 10× ratio |
| Warmup Steps | 1,000 | **500** | Very short |
| Batch Size | 16 | **12** | Reduced due to pipeline overhead |

### Results

| Epoch | Train Loss | Val Loss | Val PPL |
|-------|-----------|---------|---------|
| 1 | 3.2825 | 3.5026 | 33.20 |
| 2 | 3.1424 | 3.4311 | 30.91 |
| **3** | **3.0921** | **3.4051** | **30.12** ✅ |

- Phase 3 produced the **best general-purpose model** — capable of web text, code-adjacent text, Q&A patterns.

---

## Key Lessons Learned

1. **Cross-attention is wasteful** for a decoder-only LM: PyTorch's default `TransformerDecoderLayer` has unused cross-attention. We built `GPTBlock` from scratch to avoid this.
2. **Curriculum order matters**: The PPL jump on Wikipedia (~3.28 → 20.47) shows the model starts each new domain with high uncertainty — but it converges much faster thanks to warm-starting.
3. **Lower LR at each phase**: Each curriculum step uses lower LR (3e-4 → 1e-4 → 5e-5) to protect previously learned representations.
4. **Streaming is essential at scale**: Phase 3's C4 data (600M tokens) had to be streamed rather than downloaded fully.

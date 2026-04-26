# Curriculum Learning for Language Modeling

> A 124M parameter GPT-style Language Model trained via a 3-phase Curriculum Learning pipeline — from children's stories to Wikipedia to the open web.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Report](https://img.shields.io/badge/Academic_Report-PDF-red?logo=adobeacrobatreader)](Academic_Report.pdf)

---

## Overview

This project implements **Curriculum Learning** to pre-train a 124-million parameter Decoder-Only Transformer from scratch. Rather than training on a massive, unstructured corpus all at once, the model is sequentially exposed to datasets of increasing complexity — mirroring how humans acquire language.

```
Phase 1: TinyStories (~475M tokens)  →  Basic grammar & narrative structure
                 ↓
Phase 2: Wikipedia (~1B tokens)      →  Formal, encyclopedic prose & factual knowledge
                 ↓
Phase 3: Mixed Web Corpus (~2B tokens) → General-purpose language understanding
```

Each phase warm-starts from the previous phase's best checkpoint, ensuring knowledge is built upon rather than overwritten.

---

## Architecture

The model is a custom **GPT-style Decoder-Only Transformer**, matching the GPT-2 Medium parameterization:

| Hyperparameter | Value |
|---|---|
| Total Parameters | ~124 Million |
| Embedding Dimension ($d_{model}$) | 768 |
| Attention Heads | 12 (head dim: 64) |
| Transformer Layers | 12 |
| Feed-Forward Dimension | 3072 (4× expansion) |
| Max Sequence Length | 512 tokens |
| Tokenizer | GPT-2 BPE (50,257 vocab) |
| Normalization | Pre-LayerNorm |
| Weight Tying | ✅ (saves ~38M params) |

> **Why Decoder-Only?** Encoder-decoder models include cross-attention which is entirely redundant for autoregressive generation. The custom `GPTBlock` in `src/model.py` uses only self-attention + feed-forward layers, ensuring optimal parameter efficiency.

**Hardware:** Trained on an **NVIDIA GeForce RTX 4060 Ti (16 GB VRAM)** using FP16/AMP mixed precision, fused AdamW, and gradient accumulation.

---

## Training Phases

### Phase 1 — TinyStories Pre-Training
- **Dataset:** TinyStories (~475M tokens, ~2.12M short stories)
- **Goal:** Teach basic English syntax, subject-verb agreement, and narrative structure from scratch
- **LR:** 3e-4 → 3e-5 (cosine decay) | **Warmup:** 2,000 steps | **Epochs:** 3 | **Batch:** 128

### Phase 2 — Wikipedia Fine-Tuning
- **Dataset:** English Wikipedia Nov 2023 (~1B tokens, 10M token holdout)
- **Goal:** Transfer syntactic knowledge to formal, knowledge-rich encyclopedic prose
- **LR:** 1e-4 (3× reduced to protect Phase 1 representations) | **Warmup:** 1,000 steps

### Phase 3 — Mixed Web Corpus
- **Dataset:** ~2B tokens — 35% OpenWebText, 30% C4, 25% StackExchange, 10% Books
- **Goal:** Generalize to diverse web text, technical jargon, Q&A, and long-form narratives
- **LR:** 5e-5 | **Warmup:** 500 steps | **Dropout:** 0.05

---

## Results

| Metric | Direction | Phase 1 (TinyStories) | Phase 2 (Wikipedia) | Phase 3 (Mixed Corpus) |
|---|---|---|---|---|
| **Perplexity (PPL)** | ↓ lower | **3.247** | 19.781 | 28.175 |
| **BLEU-4** | ↑ higher | 0.2379 | 0.2216 | **0.2692** |
| **ROUGE-L (F₁)** | ↑ higher | 0.0532 | 0.0634 | **0.0711** |
| **Distinct-1** | ↑ higher | 0.4520 | **0.5967** | 0.5761 |
| **Distinct-2** | ↑ higher | 0.8354 | **0.9647** | 0.9600 |
| **Self-BLEU** | ↓ lower | 2.8354 | 1.4333 | **1.2472** |
| **Repetition Rate** | ↓ lower | 0.0893 | **0.0000** | **0.0000** |

**Key Takeaways:**
- PPL increases with corpus complexity — this is **expected and desirable**, not a regression
- BLEU-4 and ROUGE-L improve progressively, confirming expanding expressiveness across phases
- Repetition drops to **zero** from Phase 2 onwards — richer corpora eliminate degenerate loops
- Phase 3 achieves the best balance: high diversity + zero repetition + best BLEU/ROUGE

---

## Project Structure

```
nlpProj/
├── src/
│   ├── model.py              # GPT-style decoder-only transformer architecture
│   └── dataset.py            # Dataset loading and tokenization utilities
├── configs/
│   ├── phase1.py             # Phase 1 training hyperparameters
│   ├── phase2.py             # Phase 2 training hyperparameters
│   └── phase3.py             # Phase 3 training hyperparameters
├── logs/                     # Training logs & loss history JSONs
├── project_history/          # PROJECT_HISTORY.md — detailed development log
├── to_upload/                # Model weights (uploaded separately — see below)
├── train_phase1.py           # Phase 1 training script (TinyStories)
├── train_phase2.py           # Phase 2 training script (Wikipedia)
├── train_phase3.py           # Phase 3 training script (Mixed Corpus)
├── download_data.py          # Dataset download & preprocessing
├── prepare_phase3_data.py    # Mixed corpus preparation & tokenization
├── evaluate_all_phases.py    # Full cross-phase evaluation pipeline
├── quick_inference.py        # Quick single-prompt inference
├── run_inference.py          # Batch inference runner
├── inference_phase1.py       # Phase 1 specific inference
├── inference_phase2.py       # Phase 2 specific inference
├── phase3_inference.py       # Phase 3 specific inference
├── diagnose.py               # Environment & GPU diagnostics
├── eval_results.json         # Full evaluation results (JSON)
├── eval_results_table.txt    # Human-readable evaluation summary
├── Academic_Report.tex       # Full academic report (LaTeX source)
└── Academic_Report.pdf       # Compiled academic report
```

---

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sparshagra/NLP_Proj_LM.git
cd NLP_Proj_LM
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Model Weights
The trained model weights (~1.4 GB each) are hosted on Google Drive:

📦 **[Download Model Weights](https://drive.google.com/drive/folders/1baRRw3U-WkwnbaQ4aibeN9UAFKQ7UgdA)**

Place the downloaded files into the `to_upload/` directory:
```
to_upload/
├── phase1_model_weights.pt
├── phase2_model_weights.pt
└── phase3_model_weights.pt
```

---

## Usage

### Quick Inference
```bash
python quick_inference.py
```

### Run Batch Inference
```bash
python run_inference.py
```

### Phase-Specific Inference
```bash
# Phase 1 (TinyStories style)
python inference_phase1.py

# Phase 2 (Wikipedia style)
python inference_phase2.py

# Phase 3 (Web/Mixed style)
python phase3_inference.py
```

### Run Full Evaluation
```bash
python evaluate_all_phases.py
```

### Download & Prepare Data
```bash
# Download datasets for all phases
python download_data.py

# Prepare Phase 3 mixed corpus
python prepare_phase3_data.py
```

### Train from Scratch
```bash
python train_phase1.py   # ~475M tokens, TinyStories
python train_phase2.py   # ~1B tokens, Wikipedia (warm-starts from Phase 1)
python train_phase3.py   # ~2B tokens, Mixed Corpus (warm-starts from Phase 2)
```

---

## Sample Outputs

**Prompt:** *"The history of the Roman Empire began"* *(Phase 3)*
> *"The history of the Roman Empire began in the late third century BCE. The first emperor had been appointed by the Romans, but he did not want to have a regent. He was forced into a pact with his enemies and was forced to flee the city after being captured by an army under Caesar's rule..."*

**Prompt:** *"Recent advancements in machine learning have shown that"* *(Phase 3)*
> *"Recent advancements in machine learning have shown that they can be used for both real and human-created tasks. One of the most important advances is to use machine learning as a tool, allowing a developer to identify features which make an impact on his/her business..."*

---

## Academic Report

The full academic report (methodology, architecture details, training dynamics, qualitative & quantitative analysis) is available as:
- **PDF:** [`Academic_Report.pdf`](Academic_Report.pdf)
- **LaTeX source:** [`Academic_Report.tex`](Academic_Report.tex)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

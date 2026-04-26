"""
Phase 1 Configuration — TinyStories (roneneldan/TinyStories)
─────────────────────────────────────────────────────────────
Dataset:
  • ~2.12 M stories, ~475 M tokens (avg story ≈ 224 tokens)
  • Simple children's vocabulary — ideal Phase 1 curriculum data

Model — 124 M parameter GPT-style decoder-only Transformer:
  d_model=768, n_heads=12, n_layers=12, d_ff=3072
  Tokenizer: GPT-2 pretrained (50,257 vocab) — no custom training needed

GPU target: RTX 4060 Ti 16 GB
  fp16 + batch_size=16, grad_accum=8 → effective batch 128
  Estimated VRAM usage: ~8-10 GB
"""

import os

DATA_ROOT  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CKPT_ROOT  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
LOG_ROOT   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


class Phase1Config:

    PHASE       = 1
    PHASE_NAME  = "phase1_tinystories"

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_DIR    = os.path.join(DATA_ROOT, "phase1_tinystories")
    OUTPUT_DIR  = os.path.join(CKPT_ROOT, "phase1")
    LOG_DIR     = LOG_ROOT

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Use GPT-2 pretrained tokenizer: battle-tested, covers all English text,
    # same vocab will be reused in Phase 2 so no retraining needed.
    TOKENIZER_NAME = "gpt2"          # loaded via AutoTokenizer.from_pretrained
    VOCAB_SIZE     = 50_257          # GPT-2 vocab size

    # ── Model (124 M effective params, no cross-attention waste) ──────────────
    D_MODEL      = 768
    N_HEADS      = 12
    N_LAYERS     = 12
    D_FF         = 3072
    MAX_SEQ_LEN  = 512
    DROPOUT      = 0.1

    # ── Training ─────────────────────────────────────────────────────────────
    # GPU-optimised for RTX 4060 Ti 16 GB with 124M param fp16 model
    # Effective batch size = BATCH_SIZE × GRAD_ACCUM = 16 × 8 = 128
    BATCH_SIZE   = 16
    GRAD_ACCUM   = 8              # accumulate to effective batch of 128
    LR           = 3e-4
    LR_MIN       = 3e-5           # cosine decay floor
    WEIGHT_DECAY = 0.1
    BETA1        = 0.9
    BETA2        = 0.95           # GPT-standard
    EPOCHS       = 3
    WARMUP_STEPS = 2_000
    CLIP_GRAD    = 1.0
    LOG_EVERY    = 100            # steps
    SAVE_EVERY   = 1              # epochs
    SEED         = 42

    # ── Dataset loading ───────────────────────────────────────────────────────
    NUM_WORKERS  = 4
    PREFETCH     = 2              # DataLoader prefetch_factor
    MAX_STORIES  = None           # None = full dataset

    # ── Generation (inference) ────────────────────────────────────────────────
    GEN_MAX_NEW    = 300
    GEN_TEMP       = 0.8
    GEN_TOP_K      = 50
    GEN_TOP_P      = 0.95
    GEN_REP_PENALTY = 1.3


cfg = Phase1Config()

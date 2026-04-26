"""
Phase 3 Configuration — Mixed Web/Reasoning Corpus
═══════════════════════════════════════════════════
Curriculum Learning Design:
  Phase 1 (done) : TinyStories  → simple grammar, narrative, basic vocab
  Phase 2 (done) : Wikipedia EN → formal prose, rich vocabulary
  Phase 3 (this) : Mixed corpus → general web text, reasoning, long-form books

Dataset composition (~2–3B tokens total):
  OpenWebText  35%  ~700M tokens  → diverse web prose
  C4           30%  ~600M tokens  → filtered Common Crawl (streamed; no full download)
  StackExchange 25% ~500M tokens  → Q&A reasoning chains (valuable!)
  Books (BC)   10%  ~200M tokens  → long-form narrative continuity

Curriculum techniques:
  1. Warm-start   — load Phase 2 best_model.pt weights
  2. LR cooldown  — 5e-5 (even lower) to protect prior knowledge
  3. Same GPT-2 tokenizer — vocabulary continuity across all phases
  4. Same 124M architecture — weights load directly

Storage estimate:
  Phase 3 data : ~7.5 GB (binary token files, int32)
  Phase 3 ckpts: ~5.6 GB (3 epochs)
  Already used : ~17 GB (phase1/2 data + ckpts)
  Free available: ~61 GB
"""

import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CKPT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
LOG_ROOT  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


class Phase3Config:

    PHASE      = 3
    PHASE_NAME = "phase3_mixed"

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_DIR   = os.path.join(DATA_ROOT, "phase3_mixed")
    OUTPUT_DIR = os.path.join(CKPT_ROOT, "phase3")
    LOG_DIR    = LOG_ROOT

    # Merged binary token files (produced by prepare_phase3_data.py)
    TRAIN_BIN  = os.path.join(DATA_ROOT, "phase3_mixed", "train_tokens.bin")
    VAL_BIN    = os.path.join(DATA_ROOT, "phase3_mixed", "val_tokens.bin")

    # ── Curriculum: Resume from Phase 2 ──────────────────────────────────────
    PHASE2_CKPT    = os.path.join(CKPT_ROOT, "phase2", "best_model.pt")
    PHASE1_TOK_DIR = os.path.join(CKPT_ROOT, "phase1", "tokenizer")

    # ── Tokenizer (SAME as Phase 1 & 2 — must never change) ──────────────────
    TOKENIZER_NAME = "gpt2"
    VOCAB_SIZE     = 50_257

    # ── Model — IDENTICAL to Phase 1 & 2 (124M params) ───────────────────────
    # DO NOT change these — weights load directly from Phase 2
    D_MODEL     = 768
    N_HEADS     = 12
    N_LAYERS    = 12
    D_FF        = 3072
    MAX_SEQ_LEN = 512
    DROPOUT     = 0.05      # slightly reduced (model is stable after 2 phases)

    # ── Training — curriculum-tuned hyperparameters ───────────────────────────
    # Very low LR to gently adapt to new domain without forgetting.
    BATCH_SIZE   = 12
    GRAD_ACCUM   = 11           # effective batch = 132 (≈128, same training dynamics)
    LR           = 5e-5         # lower than Phase 2 (1e-4)
    LR_MIN       = 5e-6         # decay floor
    WEIGHT_DECAY = 0.1
    BETA1        = 0.9
    BETA2        = 0.95
    EPOCHS       = 3
    WARMUP_STEPS = 500          # very short — model is already well-trained
    CLIP_GRAD    = 1.0
    LOG_EVERY    = 200
    SAVE_EVERY   = 1
    SEED         = 42

    # ── Dataset I/O ───────────────────────────────────────────────────────────
    NUM_WORKERS  = 4
    PREFETCH     = 2
    BLOCK_SIZE   = MAX_SEQ_LEN  # 512 tokens per example

    # ── Token targets per source (used by prepare_phase3_data.py) ─────────────
    # Total: 2.0B train tokens → ~8.0 GB on disk (int32)
    TARGET_OWT_TOKENS   =  700_000_000   # OpenWebText  35%
    TARGET_C4_TOKENS    =  600_000_000   # C4           30%  (streamed, not downloaded)
    TARGET_SE_TOKENS    =  500_000_000   # StackExchange 25%
    TARGET_BOOKS_TOKENS =  200_000_000   # BookCorpus   10%
    TARGET_VAL_TOKENS   =   20_000_000   # ~1% for validation

    # ── Generation (inference) ────────────────────────────────────────────────
    GEN_MAX_NEW     = 400
    GEN_TEMP        = 0.8
    GEN_TOP_K       = 50
    GEN_TOP_P       = 0.95
    GEN_REP_PENALTY = 1.3


cfg = Phase3Config()

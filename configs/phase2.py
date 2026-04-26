"""
Phase 2 Configuration — Wikipedia EN (Curriculum: TinyStories → Wikipedia)
═══════════════════════════════════════════════════════════════════════════
Curriculum Learning Design:
  Phase 1 (done) : TinyStories  → simple grammar, narrative, basic vocab
  Phase 2 (this) : Wikipedia EN → formal prose, rich vocabulary, diverse topics

Curriculum techniques applied:
  1. Warm-start   — load Phase 1 best_model.pt weights (no random reinit)
  2. LR cooldown  — 1e-4 (was 3e-4) preserving Phase 1 knowledge
  3. LR cosine    — decay to 1e-5 (was 3e-5) for stable convergence
  4. Same tokenizer (GPT-2) — vocabulary continuity across all phases
  5. Same 124M architecture — weights transfer exactly (no shape mismatch)

Data source: wikimedia/wikipedia (20231101.en) via HuggingFace streaming
(Equivalent content to Kaggle: ffatty/plaintext-wikipedia-full-english)

Stored as memory-mapped binary token files for fast training I/O:
  data/phase2_wikipedia/train_tokens.bin  (~4.0 GB, 1B int32 tokens)
  data/phase2_wikipedia/val_tokens.bin    (~40  MB, 10M int32 tokens)
Run: python download_data.py --phase 2    (takes ~2-4h to stream & tokenise)
"""

import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CKPT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
LOG_ROOT  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


class Phase2Config:

    PHASE      = 2
    PHASE_NAME = "phase2_wikipedia"

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_DIR   = os.path.join(DATA_ROOT, "phase2_wikipedia")
    OUTPUT_DIR = os.path.join(CKPT_ROOT, "phase2")
    LOG_DIR    = LOG_ROOT

    # Binary token cache paths (written by download_data.py --phase 2)
    TRAIN_BIN  = os.path.join(DATA_ROOT, "phase2_wikipedia", "train_tokens.bin")
    VAL_BIN    = os.path.join(DATA_ROOT, "phase2_wikipedia", "val_tokens.bin")

    # ── Curriculum: Resume from Phase 1 ──────────────────────────────────────
    # CRITICAL: must load Phase 1 weights — this IS the curriculum step
    PHASE1_CKPT    = os.path.join(CKPT_ROOT, "phase1", "best_model.pt")
    PHASE1_TOK_DIR = os.path.join(CKPT_ROOT, "phase1", "tokenizer")

    # ── Tokenizer (SAME as Phase 1 — must never change across curriculum) ─────
    TOKENIZER_NAME = "gpt2"
    VOCAB_SIZE     = 50_257

    # ── Model (IDENTICAL to Phase 1 — weights load directly, no arch change) ──
    D_MODEL     = 768
    N_HEADS     = 12
    N_LAYERS    = 12
    D_FF        = 3072
    MAX_SEQ_LEN = 512
    DROPOUT     = 0.1

    # ── Training — curriculum-tuned hyperparameters ───────────────────────────
    # Lower LR than Phase 1 (3e-4 → 1e-4): adapt gently to harder domain.
    # Shorter warmup: model already trained, we start from a good point.
    # Same effective batch size as Phase 1 (16 × 8 = 128).
    BATCH_SIZE   = 16
    GRAD_ACCUM   = 8            # effective batch = 128
    LR           = 1e-4         # 3× lower than Phase 1 (3e-4)
    LR_MIN       = 1e-5         # 3× lower than Phase 1 (3e-5)
    WEIGHT_DECAY = 0.1
    BETA1        = 0.9
    BETA2        = 0.95         # GPT-standard
    EPOCHS       = 3
    WARMUP_STEPS = 1_000        # shorter than Phase 1 (2000) — already trained
    CLIP_GRAD    = 1.0
    LOG_EVERY    = 200          # steps between log prints
    SAVE_EVERY   = 1            # save checkpoint every epoch
    SEED         = 42

    # ── Dataset I/O ───────────────────────────────────────────────────────────
    NUM_WORKERS  = 4
    PREFETCH     = 2
    BLOCK_SIZE   = MAX_SEQ_LEN  # 512 tokens per example

    # ── Target token counts for the download script ───────────────────────────
    TARGET_TRAIN_TOKENS = 1_000_000_000   # 1B  → ~4.0 GB on disk
    TARGET_VAL_TOKENS   =    10_000_000   # 10M → ~40 MB on disk

    # ── Generation (inference) ────────────────────────────────────────────────
    GEN_MAX_NEW     = 300
    GEN_TEMP        = 0.8
    GEN_TOP_K       = 50
    GEN_TOP_P       = 0.95
    GEN_REP_PENALTY = 1.3


cfg = Phase2Config()

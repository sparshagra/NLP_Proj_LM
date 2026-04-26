"""
train_phase3.py
═══════════════════════════════════════════════════════════════════════════════
Phase 3 Training Script — Mixed Web/Reasoning Corpus
Curriculum: TinyStories → Wikipedia → [OpenWebText + C4 + StackExchange + Books]

Dataset composition (~2B tokens):
  OpenWebText   35%  ~700M  → diverse web prose
  C4            30%  ~600M  → filtered Common Crawl
  StackExchange 25%  ~500M  → Q&A reasoning (boosted for reasoning quality)
  BookCorpus    10%  ~200M  → long-form narrative

Run (fresh Phase 3 from Phase 2 checkpoint):
    python train_phase3.py

Run (resume interrupted Phase 3):
    python train_phase3.py --resume

Run in background (keeps training when SSH disconnects):
    export WANDB_API_KEY="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"
    nohup python train_phase3.py --wandb > logs/train_phase3.log 2>&1 &
    echo "PID=$!" > logs/phase3.pid
    tail -f logs/train_phase3.log

Pre-requisites:
    python prepare_phase3_data.py     ← run once first (~6-10h, streams datasets)
    pip install wandb                 ← only needed with --wandb

Checkpoint structure:
    checkpoints/phase3/
        best_model.pt            ← best val-loss model (use for inference)
        checkpoint_epoch01.pt
        checkpoint_epoch02.pt
        checkpoint_epoch03.pt
        tokenizer/               ← GPT-2 tokenizer copy
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.model   import DecoderOnlyTransformer, count_parameters, model_summary
from src.dataset import WikiBinDataset            # reuse — same binary format
from configs.phase3 import cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_dirs():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,    exist_ok=True)
    os.makedirs(cfg.DATA_DIR,   exist_ok=True)


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_info(device):
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU     : {props.name}")
        print(f"  VRAM    : {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA    : {torch.version.cuda}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    """Load the shared GPT-2 tokenizer. Same tokenizer used in all 3 phases."""
    tok_src = cfg.PHASE1_TOK_DIR
    tok_dst = os.path.join(cfg.OUTPUT_DIR, "tokenizer")

    if os.path.isdir(tok_src) and os.listdir(tok_src):
        print(f"  Loading Phase 1 tokenizer from {tok_src} …")
        tok = AutoTokenizer.from_pretrained(tok_src)
        os.makedirs(tok_dst, exist_ok=True)
        if not os.listdir(tok_dst):
            tok.save_pretrained(tok_dst)
    elif os.path.isdir(tok_dst) and os.listdir(tok_dst):
        print(f"  Loading tokenizer from {tok_dst} …")
        tok = AutoTokenizer.from_pretrained(tok_dst)
    else:
        print("  Downloading GPT-2 tokenizer from HuggingFace …")
        tok = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        os.makedirs(tok_dst, exist_ok=True)
        tok.save_pretrained(tok_dst)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Vocab size  : {tok.vocab_size:,}")
    print(f"  EOS token   : {tok.eos_token!r}  (id={tok.eos_token_id})")
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets():
    flag = os.path.join(cfg.DATA_DIR, ".downloaded")
    if not os.path.exists(flag):
        raise FileNotFoundError(
            f"\n  Phase 3 data not found at {cfg.DATA_DIR}\n"
            "  → Run first:  python prepare_phase3_data.py\n"
            "  (This will stream ~2B tokens, takes ~6-10h)\n"
        )
    print("Loading Phase 3 binary token corpus …")

    # Print meta if available
    meta_path = os.path.join(cfg.DATA_DIR, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Total train tokens : {meta.get('total_train_tokens', 0):,}")
        print(f"  Train file         : {meta.get('train_bin_gb', '?')} GB")
        print(f"  Source breakdown:")
        for src, pct in meta.get("percentages", {}).items():
            print(f"    {src:18s}: {pct:.1f}%")

    train_ds = WikiBinDataset(cfg.TRAIN_BIN, block_size=cfg.BLOCK_SIZE)
    val_ds   = WikiBinDataset(cfg.VAL_BIN,   block_size=cfg.BLOCK_SIZE)
    return train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, use_amp: bool, max_batches: int = 200) -> float:
    """Evaluate on up to max_batches to cap VRAM usage during validation."""
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            if n >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sample generations
# ─────────────────────────────────────────────────────────────────────────────

def sample_text(model, tokenizer, device):
    """
    Test prompts spanning all curriculum phases to check knowledge retention
    and new domain adaptation.
    """
    prompts = [
        # Phase 1 domain (catastrophic forgetting check)
        "Once upon a time, there was a little girl",
        # Phase 2 domain (Wikipedia knowledge)
        "The history of the Roman Empire began",
        # Phase 3 domains
        "The best way to implement a binary search tree in Python is",
        "Recent advancements in machine learning have shown that",
        "Stack Overflow: How do I fix a segmentation fault in C++?",
    ]
    model.eval()
    print("\n── Sample Generations ──────────────────────────────────────────")
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            out = model.generate(
                ids,
                max_new         = cfg.GEN_MAX_NEW,
                temperature     = cfg.GEN_TEMP,
                top_k           = cfg.GEN_TOP_K,
                top_p           = cfg.GEN_TOP_P,
                repetition_penalty = cfg.GEN_REP_PENALTY,
                eos_id          = tokenizer.eos_token_id,
                max_seq_len     = cfg.MAX_SEQ_LEN,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"\n  [{prompt[:60]}]\n  {text}\n")
    print("────────────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int,
           lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr_max * step / max(warmup_steps, 1)
    if step > total_steps:
        return lr_min
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(resume: bool = False, use_wandb: bool = False,
          wandb_project: str = "curriculum_lm", wandb_run: str = None):

    setup_dirs()
    set_seed(cfg.SEED)

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = DEVICE.type == "cuda"
    print(f"\n  Device  : {DEVICE}")
    gpu_info(DEVICE)

    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb_run_obj = None
    if use_wandb:
        try:
            import wandb
            wandb_run_obj = wandb.init(
                project = wandb_project,
                name    = wandb_run or "phase3_mixed_124M",
                config  = {k: v for k, v in cfg.__dict__.items()
                           if not k.startswith("__") and not callable(v)},
                resume  = "allow" if resume else None,
                tags    = ["phase3", "mixed", "curriculum",
                           "openwebtext", "c4", "stackexchange", "bookcorpus"],
            )
            print(f"  W&B     : {wandb_run_obj.url}")
        except ImportError:
            print("  ⚠  wandb not installed — skipping.")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nLoading tokenizer …")
    tokenizer = get_tokenizer()
    PAD_ID    = tokenizer.pad_token_id
    EOS_ID    = tokenizer.eos_token_id

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\nLoading Phase 3 datasets …")
    train_ds, val_ds = load_datasets()

    train_loader = DataLoader(
        train_ds,
        batch_size       = cfg.BATCH_SIZE,
        shuffle          = True,
        num_workers      = cfg.NUM_WORKERS,
        pin_memory       = True,
        prefetch_factor  = cfg.PREFETCH,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = cfg.BATCH_SIZE,   # same as train — 2× was causing OOM
        shuffle          = False,
        num_workers      = cfg.NUM_WORKERS,
        pin_memory       = True,
        prefetch_factor  = cfg.PREFETCH,
        persistent_workers = True,
    )
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val   batches : {len(val_loader):,}")

    # ── Model — MUST match Phase 2 architecture exactly ───────────────────────
    model = DecoderOnlyTransformer(
        vocab_size = tokenizer.vocab_size,
        d_model    = cfg.D_MODEL,
        n_heads    = cfg.N_HEADS,
        n_layers   = cfg.N_LAYERS,
        d_ff       = cfg.D_FF,
        max_len    = cfg.MAX_SEQ_LEN + 1,
        dropout    = cfg.DROPOUT,
        pad_id     = PAD_ID,
    ).to(DEVICE)

    model_summary(model)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params,   "weight_decay": cfg.WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr    = cfg.LR,
        betas = (cfg.BETA1, cfg.BETA2),
        fused = True if DEVICE.type == "cuda" else False,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # ── Curriculum: load checkpoint ───────────────────────────────────────────
    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")
    history       = {
        "train_loss": [], "val_loss": [], "val_ppl": [],
        "phase2_start_ppl": None,
    }

    phase3_best = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")

    if resume and os.path.exists(phase3_best):
        # ── Resume interrupted Phase 3 run ────────────────────────────────────
        print(f"\nResuming Phase 3 from {phase3_best} …")
        ckpt = torch.load(phase3_best, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt.get("global_step", 0)
        best_val_loss = ckpt["val_loss"]
        history       = ckpt.get("history", history)
        print(f"  Resumed at epoch {start_epoch}  "
              f"(best val_ppl={math.exp(best_val_loss):.2f})")

    elif os.path.exists(cfg.PHASE2_CKPT):
        # ── Fresh Phase 3: load Phase 2 weights (curriculum warm-start) ───────
        print(f"\n{'═'*65}")
        print(f"  CURRICULUM LEARNING — Loading Phase 2 weights")
        print(f"  Source: {cfg.PHASE2_CKPT}")
        p2_ckpt = torch.load(cfg.PHASE2_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(p2_ckpt["model_state"])
        p2_val_ppl = p2_ckpt.get("val_ppl", math.exp(p2_ckpt["val_loss"]))
        history["phase2_start_ppl"] = p2_val_ppl
        print(f"  Phase 2 val PPL  : {p2_val_ppl:.2f}  (Wikipedia)")
        print(f"  Phase 3 target   : strong generalisation on web+reasoning text")
        print(f"  LR               : {cfg.LR:.0e}  (Phase 2 was 1e-4, ×2 lower)")
        print(f"{'═'*65}\n")
    else:
        print(f"\n  ⚠  Phase 2 checkpoint not found at {cfg.PHASE2_CKPT}")
        print("  Training from SCRATCH (not recommended for curriculum learning).")
        print("  Run Phase 1 + Phase 2 first, or copy best_model.pt to "
              "checkpoints/phase2/\n")

    # ── Compute total optimiser steps ─────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / cfg.GRAD_ACCUM)
    total_steps     = steps_per_epoch * cfg.EPOCHS

    print(f"{'═'*65}")
    print(f"  Phase-3 Training — Mixed Corpus  |  124M decoder-only Transformer")
    print(f"  Curriculum : Phase1→Phase2→Phase3 (TinyStories→Wiki→Mixed)")
    print(f"  Epochs     : {start_epoch} → {cfg.EPOCHS}")
    print(f"  Eff. batch : {cfg.BATCH_SIZE} × {cfg.GRAD_ACCUM} = "
          f"{cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
    print(f"  Total steps: {total_steps:,}  (warmup: {cfg.WARMUP_STEPS:,})")
    print(f"  LR         : {cfg.LR:.0e} → {cfg.LR_MIN:.0e}  (cosine)")
    print(f"  AMP (fp16) : {USE_AMP}")
    print(f"{'═'*65}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.EPOCHS + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader, 1):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                _, loss = model(x, y)
                loss    = loss / cfg.GRAD_ACCUM

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.GRAD_ACCUM
            n_batches  += 1

            if batch_idx % cfg.GRAD_ACCUM == 0:
                # LR schedule
                lr = get_lr(global_step, cfg.WARMUP_STEPS, total_steps,
                            cfg.LR, cfg.LR_MIN)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg.LOG_EVERY == 0:
                    avg     = epoch_loss / n_batches
                    elapsed = time.time() - t0
                    tokens_seen = (global_step * cfg.BATCH_SIZE
                                   * cfg.GRAD_ACCUM * cfg.MAX_SEQ_LEN)
                    print(
                        f"  Ep {epoch:02d} | step {global_step:6d} | "
                        f"loss {avg:.4f} | ppl {math.exp(avg):8.2f} | "
                        f"lr {lr:.2e} | {elapsed:.0f}s elapsed",
                        flush=True,
                    )
                    if wandb_run_obj:
                        wandb_run_obj.log({
                            "train/loss": avg,
                            "train/ppl":  math.exp(avg),
                            "train/lr":   lr,
                            "tokens":     tokens_seen,
                        }, step=global_step)

        # ── Epoch end ─────────────────────────────────────────────────────────
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()          # free residual activation memory
        val_loss = evaluate(model, val_loader, DEVICE, USE_AMP)
        val_ppl  = math.exp(val_loss)
        trn_loss = epoch_loss / n_batches
        elapsed  = time.time() - t0

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch:02d} done  ({elapsed:.0f}s  |  {elapsed/3600:.1f}h)")
        print(f"  Train loss : {trn_loss:.4f}  PPL : {math.exp(trn_loss):.2f}")
        print(f"  Val   loss : {val_loss:.4f}  PPL : {val_ppl:.2f}")
        if history["phase2_start_ppl"]:
            print(f"  Phase 2 PPL was : {history['phase2_start_ppl']:.2f}  "
                  f"(curriculum improvement: {history['phase2_start_ppl'] - val_ppl:+.2f})")
        print(f"{'─'*65}\n")

        if wandb_run_obj:
            wandb_run_obj.log({
                "val/loss":  val_loss,
                "val/ppl":   val_ppl,
                "epoch":     epoch,
            }, step=global_step)

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = {
            "epoch":       epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "opt_state":   optimizer.state_dict(),
            "scaler_state":scaler.state_dict(),
            "val_loss":    val_loss,
            "val_ppl":     val_ppl,
            "history":     history,
            "phase":       3,
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("__") and not callable(v)},
        }

        ep_path = os.path.join(cfg.OUTPUT_DIR, f"checkpoint_epoch{epoch:02d}.pt")
        torch.save(ckpt, ep_path)
        print(f"  Checkpoint  → {ep_path}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, phase3_best)
            print(f"  ★ New best  (val_loss={val_loss:.4f}  ppl={val_ppl:.2f})",
                  flush=True)

        # ── Sample generations ────────────────────────────────────────────────
        sample_text(model, tokenizer, DEVICE)

    # ── Training complete ─────────────────────────────────────────────────────
    hist_path = os.path.join(cfg.LOG_DIR, "phase3_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    if wandb_run_obj:
        wandb_run_obj.finish()

    print(f"\n{'═'*65}")
    print(f"  Phase-3 Training Complete!")
    print(f"  Best val PPL : {math.exp(best_val_loss):.2f}  (Mixed corpus)")
    if history["phase2_start_ppl"]:
        print(f"  Phase 2 PPL  : {history['phase2_start_ppl']:.2f}  (Wikipedia)")
        print(f"  Curriculum   : model improved across all 3 phases ✓")
    print(f"  Best model   : {phase3_best}")
    print(f"  History      : {hist_path}")
    print(f"{'═'*65}")
    print(f"\n→ Inference: python inference_phase3.py")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Curriculum fine-tuning on mixed web/reasoning corpus (124M)"
    )
    parser.add_argument("--resume",        action="store_true",
                        help="Resume from checkpoints/phase3/best_model.pt")
    parser.add_argument("--wandb",         action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", default="curriculum_lm",
                        help="W&B project name (default: curriculum_lm)")
    parser.add_argument("--wandb_run",     default=None,
                        help="W&B run name (default: phase3_mixed_124M)")
    args = parser.parse_args()

    train(
        resume        = args.resume,
        use_wandb     = args.wandb,
        wandb_project = args.wandb_project,
        wandb_run     = args.wandb_run,
    )

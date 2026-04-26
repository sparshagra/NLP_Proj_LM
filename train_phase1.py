"""
Phase 1 Training Script — TinyStories (124M decoder-only Transformer)
══════════════════════════════════════════════════════════════════════
Run (fresh):
    python train_phase1.py

Run (resume from checkpoint):
    python train_phase1.py --resume

Run offline (keeps training even if SSH disconnects):
    nohup python train_phase1.py > logs/train_phase1.log 2>&1 &
    tail -f logs/train_phase1.log      # watch progress from another terminal

With Weights & Biases logging:
    python train_phase1.py --wandb --wandb_project my_project --wandb_run phase1_124M

Pre-requisites:
    python download_data.py --phase 1     ← run once to download TinyStories
    pip install wandb                     ← only needed if using --wandb
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
from src.model    import DecoderOnlyTransformer, count_parameters, model_summary
from src.dataset  import ArrowLMDataset
from configs.phase1 import cfg

try:
    from datasets import load_from_disk
except ImportError:
    raise ImportError("Run: pip install datasets")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_dirs():
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.LOG_DIR,    exist_ok=True)


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
# 1. Dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_arrow_splits():
    flag = os.path.join(cfg.DATA_DIR, ".downloaded")
    if not os.path.exists(flag):
        raise FileNotFoundError(
            f"\n  Dataset not found at {cfg.DATA_DIR}\n"
            "  → Run:  python download_data.py --phase 1\n"
        )
    print("Loading TinyStories from disk …")
    ds = load_from_disk(cfg.DATA_DIR)
    train_split = ds["train"]
    val_split   = ds["validation"]
    print(f"  Train: {len(train_split):,} stories")
    print(f"  Val  : {len(val_split):,}  stories")
    return train_split, val_split


# ─────────────────────────────────────────────────────────────────────────────
# 2. Tokenizer — GPT-2 pretrained (no training needed)
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer():
    tok_cache = os.path.join(cfg.OUTPUT_DIR, "tokenizer")
    if os.path.isdir(tok_cache) and os.listdir(tok_cache):
        print(f"Loading cached tokenizer from {tok_cache} …")
        tok = AutoTokenizer.from_pretrained(tok_cache)
    else:
        print(f"Downloading GPT-2 tokenizer ({cfg.TOKENIZER_NAME}) …")
        tok = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        os.makedirs(tok_cache, exist_ok=True)
        tok.save_pretrained(tok_cache)
        print(f"  Tokenizer cached → {tok_cache}")

    # GPT-2 has no pad token by default — use eos as pad (common practice)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"  Vocab size  : {tok.vocab_size:,}")
    print(f"  EOS token   : {tok.eos_token!r}  (id={tok.eos_token_id})")
    print(f"  PAD token   : {tok.pad_token!r}  (id={tok.pad_token_id})")
    return tok


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, scaler_enabled) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=scaler_enabled):
                _, loss = model(x, y)
            total_loss += loss.item()
            n += 1
    return total_loss / max(n, 1)


def sample_text(model, tokenizer, device, n_samples=3):
    prompts = [
        "Once upon a time",
        "The little rabbit",
        "One day in the forest",
    ]
    model.eval()
    print("\n── Sample Generations ──────────────────────────────────")
    with torch.no_grad():
        for prompt in prompts[:n_samples]:
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
            print(f"\n  [{prompt}]\n  {text}\n")
    print("────────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LR schedule: linear warmup + cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int,
           lr_max: float, lr_min: float) -> float:
    if step < warmup_steps:                           # linear warmup
        return lr_max * step / max(warmup_steps, 1)
    if step > total_steps:                            # past end — hold at min
        return lr_min
    # cosine decay
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(resume: bool = False, use_wandb: bool = False,
          wandb_project: str = "curriculum_lm", wandb_run: str = None):

    setup_dirs()
    set_seed(cfg.SEED)

    # ── Device setup ─────────────────────────────────────────────────────────
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP      = DEVICE.type == "cuda"
    print(f"\n  Device  : {DEVICE}")
    gpu_info(DEVICE)

    # Enable TF32 on Ampere+ GPUs (faster matmuls, no accuracy loss)
    if DEVICE.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True   # auto-optimise kernels

    # ── Wandb ─────────────────────────────────────────────────────────────────
    wandb_run_obj = None
    if use_wandb:
        try:
            import wandb
            wandb_run_obj = wandb.init(
                project = wandb_project,
                name    = wandb_run or f"phase1_124M",
                config  = {k: v for k, v in cfg.__dict__.items()
                           if not k.startswith("__") and not callable(v)},
                resume  = "allow" if resume else None,
            )
            print(f"  W&B     : {wandb_run_obj.url}")
        except ImportError:
            print("  ⚠  wandb not installed — skipping. Run: pip install wandb")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_split, val_split = load_arrow_splits()
    tokenizer = get_tokenizer()

    PAD_ID = tokenizer.pad_token_id
    EOS_ID = tokenizer.eos_token_id
    BOS_ID = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else EOS_ID

    print("\nBuilding datasets …")
    train_ds = ArrowLMDataset(
        train_split, tokenizer, cfg.MAX_SEQ_LEN,
        bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
        max_stories=cfg.MAX_STORIES,
    )
    val_ds = ArrowLMDataset(
        val_split, tokenizer, cfg.MAX_SEQ_LEN,
        bos_id=BOS_ID, eos_id=EOS_ID, pad_id=PAD_ID,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size    = cfg.BATCH_SIZE,
        shuffle       = True,
        num_workers   = cfg.NUM_WORKERS,
        pin_memory    = True,
        prefetch_factor = cfg.PREFETCH,
        persistent_workers = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size    = cfg.BATCH_SIZE * 2,     # can use larger batch for eval
        shuffle       = False,
        num_workers   = cfg.NUM_WORKERS,
        pin_memory    = True,
        prefetch_factor = cfg.PREFETCH,
        persistent_workers = True,
    )
    print(f"  Train batches : {len(train_loader):,}")
    print(f"  Val   batches : {len(val_loader):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Optimizer (AdamW with GPT-standard betas) ─────────────────────────────
    # Separate weight-decay groups: skip biases and LayerNorm params
    decay_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() < 2]
    optimizer = AdamW(
        [
            {"params": decay_params,   "weight_decay": cfg.WEIGHT_DECAY},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.LR,
        betas=(cfg.BETA1, cfg.BETA2),
        fused=True if DEVICE.type == "cuda" else False,  # fused AdamW (faster)
    )

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 1
    global_step   = 0
    best_val_loss = float("inf")
    history       = {"train_loss": [], "val_loss": [], "val_ppl": []}

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    if resume and os.path.exists(ckpt_path):
        print(f"\nResuming from {ckpt_path} …")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch   = ckpt["epoch"] + 1
        global_step   = ckpt.get("global_step", 0)
        best_val_loss = ckpt["val_loss"]
        history       = ckpt.get("history", history)
        print(f"  Resumed at epoch {start_epoch}  "
              f"(best val_ppl={math.exp(best_val_loss):.2f})")
    elif resume:
        print("  ⚠  No checkpoint found — starting from scratch.")

    # ── Compute total optimiser steps for LR schedule ─────────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / cfg.GRAD_ACCUM)
    total_steps     = steps_per_epoch * cfg.EPOCHS

    print(f"\n{'═'*65}")
    print(f"  Phase-1 Training — TinyStories  |  124M decoder-only Transformer")
    print(f"  Epochs        : {start_epoch} → {cfg.EPOCHS}")
    print(f"  Effective batch: {cfg.BATCH_SIZE} × {cfg.GRAD_ACCUM} = "
          f"{cfg.BATCH_SIZE * cfg.GRAD_ACCUM}")
    print(f"  Total steps   : {total_steps:,}  (warmup: {cfg.WARMUP_STEPS:,})")
    print(f"  AMP (fp16)    : {USE_AMP}")
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
                # ── LR update (manual cosine+warmup) ─────────────────────────
                lr = get_lr(global_step, cfg.WARMUP_STEPS, total_steps,
                            cfg.LR, cfg.LR_MIN)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
                global_step += 1

                if global_step % cfg.LOG_EVERY == 0:
                    avg = epoch_loss / n_batches
                    elapsed = time.time() - t0
                    tokens_seen = global_step * cfg.BATCH_SIZE * cfg.GRAD_ACCUM * cfg.MAX_SEQ_LEN
                    print(f"  Ep {epoch:02d} | step {global_step:6d} | "
                          f"loss {avg:.4f} | ppl {math.exp(avg):8.2f} | "
                          f"lr {lr:.2e} | {elapsed:.0f}s elapsed")

                    if wandb_run_obj:
                        wandb_run_obj.log({
                            "train/loss": avg,
                            "train/ppl":  math.exp(avg),
                            "train/lr":   lr,
                            "tokens":     tokens_seen,
                        }, step=global_step)

        # ── Epoch end ─────────────────────────────────────────────────────────
        val_loss = evaluate(model, val_loader, DEVICE, USE_AMP)
        val_ppl  = math.exp(val_loss)
        trn_loss = epoch_loss / n_batches
        elapsed  = time.time() - t0

        history["train_loss"].append(trn_loss)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)

        print(f"\n{'─'*65}")
        print(f"  Epoch {epoch:02d} done  ({elapsed:.0f}s  |  "
              f"{elapsed/3600:.1f}h)")
        print(f"  Train loss : {trn_loss:.4f}  PPL : {math.exp(trn_loss):.2f}")
        print(f"  Val   loss : {val_loss:.4f}  PPL : {val_ppl:.2f}")
        print(f"{'─'*65}\n")

        if wandb_run_obj:
            wandb_run_obj.log({
                "val/loss": val_loss,
                "val/ppl":  val_ppl,
                "epoch":    epoch,
            }, step=global_step)

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = {
            "epoch":        epoch,
            "global_step":  global_step,
            "model_state":  model.state_dict(),
            "opt_state":    optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_loss":     val_loss,
            "val_ppl":      val_ppl,
            "history":      history,
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("__") and not callable(v)},
        }

        ep_path = os.path.join(cfg.OUTPUT_DIR, f"checkpoint_epoch{epoch:02d}.pt")
        torch.save(ckpt, ep_path)
        print(f"  Checkpoint  → {ep_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, ckpt_path)
            print(f"  ★ New best  (val_loss={val_loss:.4f}  ppl={val_ppl:.2f})")

        # ── Sample generations after each epoch ──────────────────────────────
        sample_text(model, tokenizer, DEVICE)

    # ── Final ─────────────────────────────────────────────────────────────────
    hist_path = os.path.join(cfg.LOG_DIR, "phase1_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    if wandb_run_obj:
        wandb_run_obj.finish()

    print(f"\n{'═'*65}")
    print(f"  Phase-1 Training Complete!")
    print(f"  Best val PPL : {math.exp(best_val_loss):.2f}")
    print(f"  Best model   : {ckpt_path}")
    print(f"  History      : {hist_path}")
    print(f"{'═'*65}")
    print(f"\n→ Next: python train_phase2.py")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1: Train 124M decoder-only Transformer on TinyStories"
    )
    parser.add_argument("--resume",         action="store_true",
                        help="Resume from checkpoints/phase1/best_model.pt")
    parser.add_argument("--wandb",          action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project",  default="curriculum_lm",
                        help="W&B project name (default: curriculum_lm)")
    parser.add_argument("--wandb_run",      default=None,
                        help="W&B run name (default: phase1_124M)")
    args = parser.parse_args()

    train(
        resume        = args.resume,
        use_wandb     = args.wandb,
        wandb_project = args.wandb_project,
        wandb_run     = args.wandb_run,
    )

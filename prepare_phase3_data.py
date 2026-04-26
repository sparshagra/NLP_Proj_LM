"""
prepare_phase3_data.py
══════════════════════
Phase 3 dataset preparation for curriculum learning.

Streams four datasets and writes merged int32 binary token files:

  OpenWebText   35%  ~700M tokens   Skylion007/openwebtext
  C4            30%  ~600M tokens   allenai/c4  (STREAMED — never fully downloaded)
  StackExchange 25%  ~500M tokens   HuggingFaceH4/stack-exchange-preferences
  Wikipedia     10%  ~200M tokens   wikimedia/wikipedia 20231101.en
                               (BookCorpus deprecated — uses old loading script)

Total target: ~2.0B train tokens + ~20M val tokens
Disk usage  : ~8.0 GB  (int32, 4 bytes per token)

Storage strategy:
  • C4 is enormous (~800 GB). We STREAM it and stop at 600M tokens.
  • All other datasets are downloaded via HuggingFace cache then streamed
    through our tokeniser. We only keep the .bin output; HF cache is
    deleted after each source to reclaim disk. Pass --keep-cache to skip.

Output:
  data/phase3_mixed/train_tokens.bin   merged, shuffled train corpus
  data/phase3_mixed/val_tokens.bin     ~1% held-out validation
  data/phase3_mixed/meta.json          statistics

Run:
  python prepare_phase3_data.py

Run in background (recommended — takes 4-10h):
  nohup python prepare_phase3_data.py > logs/prepare_phase3.log 2>&1 &
  echo "PID=$!" > logs/prepare_phase3.pid
  tail -f logs/prepare_phase3.log
"""

import argparse
import json
import os
import random
import shutil
import struct
import sys
import tempfile
import time

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
LOG_ROOT     = os.path.join(PROJECT_ROOT, "logs")
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")

PHASE3_DIR   = os.path.join(DATA_ROOT, "phase3_mixed")
TRAIN_BIN    = os.path.join(PHASE3_DIR, "train_tokens.bin")
VAL_BIN      = os.path.join(PHASE3_DIR, "val_tokens.bin")
META_JSON    = os.path.join(PHASE3_DIR, "meta.json")
DONE_FLAG    = os.path.join(PHASE3_DIR, ".downloaded")

# Per-source token targets
TARGET = {
    "openwebtext":   700_000_000,   # 35%
    "c4":            600_000_000,   # 30%
    "stackexchange": 500_000_000,   # 25%
    "bookcorpus":    200_000_000,   # 10%
}
TARGET_VAL  = 20_000_000           # ~1% val
VAL_MODULO  = 100                  # every 100th document goes to val


# ── Utilities ─────────────────────────────────────────────────────────────────

def get_tokenizer():
    from transformers import AutoTokenizer
    tok_cache = os.path.join(CKPT_ROOT, "phase1", "tokenizer")
    if os.path.isdir(tok_cache) and os.listdir(tok_cache):
        tok = AutoTokenizer.from_pretrained(tok_cache)
        print(f"  Loaded tokenizer from Phase 1 cache: {tok_cache}")
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("  Downloaded GPT-2 tokenizer from HuggingFace")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"  Vocab: {tok.vocab_size:,}  EOS={tok.eos_token_id}")
    return tok


def disk_free_gb() -> float:
    st = os.statvfs(PROJECT_ROOT)
    return st.f_bavail * st.f_frsize / 1e9


def human(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.3f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    return str(n)


def write_ids(f_train, f_val, ids: list[int],
              train_tokens: int, val_tokens: int, doc_idx: int,
              target_train: int):
    """Route a tokenised document to train or val file."""
    if doc_idx % VAL_MODULO == 0 and val_tokens < TARGET_VAL:
        f_val.write(struct.pack(f"{len(ids)}i", *ids))
        val_tokens += len(ids)
    elif train_tokens < target_train:
        f_train.write(struct.pack(f"{len(ids)}i", *ids))
        train_tokens += len(ids)
    return train_tokens, val_tokens


def progress(source: str, tok_count: int, target: int,
             val_count: int, doc_idx: int, t0: float, disk_used_gb: float):
    elapsed  = time.time() - t0
    speed    = tok_count / max(elapsed, 1) / 1e6
    pct      = min(tok_count / target * 100, 100)
    print(
        f"  [{source:12s}] docs:{doc_idx:>8,}  "
        f"tokens:{human(tok_count):>10}  ({pct:5.1f}%)  "
        f"val:{human(val_count):>8}  "
        f"disk:{disk_used_gb:.1f}GB free  "
        f"{speed:.1f}Mtok/s  [{elapsed/3600:.1f}h]",
        flush=True,
    )


# ── Source 1: OpenWebText ──────────────────────────────────────────────────────

def stream_openwebtext(tok, f_train, f_val,
                       train_tokens: int, val_tokens: int) -> tuple[int, int]:
    from datasets import load_dataset

    TARGET_THIS = TARGET["openwebtext"]
    print(f"\n{'═'*68}")
    print(f"  [1/4] OpenWebText — Skylion007/openwebtext")
    print(f"  Target : {human(TARGET_THIS)} train tokens")
    print(f"{'═'*68}")

    EOS = tok.eos_token_id
    BOS = tok.bos_token_id if tok.bos_token_id is not None else EOS

    ds = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=True,
    )

    local_train = 0
    local_val   = 0
    doc_idx     = 0
    t0          = time.time()

    for doc in ds:
        if local_train >= TARGET_THIS and (val_tokens + local_val) >= TARGET_VAL:
            break

        text = doc.get("text", "")
        if not text or len(text) < 50:
            continue

        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        doc_idx += 1

        if doc_idx % VAL_MODULO == 0 and (val_tokens + local_val) < TARGET_VAL:
            f_val.write(struct.pack(f"{len(ids)}i", *ids))
            local_val   += len(ids)
        elif local_train < TARGET_THIS:
            f_train.write(struct.pack(f"{len(ids)}i", *ids))
            local_train += len(ids)

        if doc_idx % 10_000 == 0:
            progress("openwebtext", local_train, TARGET_THIS,
                     val_tokens + local_val, doc_idx,
                     t0, disk_free_gb())

    elapsed = time.time() - t0
    print(f"\n  ✔  OpenWebText done — {human(local_train)} train tokens "
          f"| {human(local_val)} val tokens | {elapsed/3600:.1f}h")
    return train_tokens + local_train, val_tokens + local_val


# ── Source 2: C4 (STREAMED ONLY — never downloads full dataset) ───────────────

def stream_c4(tok, f_train, f_val,
              train_tokens: int, val_tokens: int) -> tuple[int, int]:
    from datasets import load_dataset

    TARGET_THIS = TARGET["c4"]
    print(f"\n{'═'*68}")
    print(f"  [2/4] C4 — allenai/c4  (STREAMED — no full download)")
    print(f"  Target : {human(TARGET_THIS)} train tokens  (~2.4 GB)")
    print(f"  Strategy: streaming the 'en' split, stop at threshold")
    print(f"{'═'*68}")

    EOS = tok.eos_token_id
    BOS = tok.bos_token_id if tok.bos_token_id is not None else EOS

    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )

    local_train = 0
    local_val   = 0
    doc_idx     = 0
    t0          = time.time()

    for doc in ds:
        if local_train >= TARGET_THIS and (val_tokens + local_val) >= TARGET_VAL:
            break
        if local_train >= TARGET_THIS and (val_tokens + local_val) < TARGET_VAL:
            # Only collect remaining val tokens
            text = doc.get("text", "")
            if not text or len(text) < 50:
                continue
            ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
            doc_idx += 1
            if doc_idx % VAL_MODULO == 0:
                f_val.write(struct.pack(f"{len(ids)}i", *ids))
                local_val += len(ids)
            continue

        text = doc.get("text", "")
        if not text or len(text) < 50:
            continue

        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        doc_idx += 1

        if doc_idx % VAL_MODULO == 0 and (val_tokens + local_val) < TARGET_VAL:
            f_val.write(struct.pack(f"{len(ids)}i", *ids))
            local_val += len(ids)
        elif local_train < TARGET_THIS:
            f_train.write(struct.pack(f"{len(ids)}i", *ids))
            local_train += len(ids)

        if doc_idx % 10_000 == 0:
            progress("c4", local_train, TARGET_THIS,
                     val_tokens + local_val, doc_idx,
                     t0, disk_free_gb())

    elapsed = time.time() - t0
    print(f"\n  ✔  C4 done — {human(local_train)} train tokens "
          f"| {human(local_val)} val tokens | {elapsed/3600:.1f}h")
    return train_tokens + local_train, val_tokens + local_val


# ── Source 3: StackExchange ───────────────────────────────────────────────────

def stream_stackexchange(tok, f_train, f_val,
                         train_tokens: int, val_tokens: int) -> tuple[int, int]:
    from datasets import load_dataset

    TARGET_THIS = TARGET["stackexchange"]
    print(f"\n{'═'*68}")
    print(f"  [3/4] StackExchange — HuggingFaceH4/stack-exchange-preferences")
    print(f"  Target : {human(TARGET_THIS)} train tokens  (reasoning-rich Q&A)")
    print(f"{'═'*68}")

    EOS = tok.eos_token_id
    BOS = tok.bos_token_id if tok.bos_token_id is not None else EOS

    # This dataset has question + ranked answers; we concatenate Q + best answer
    ds = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True,
    )

    local_train = 0
    local_val   = 0
    doc_idx     = 0
    t0          = time.time()

    for doc in ds:
        if local_train >= TARGET_THIS and (val_tokens + local_val) >= TARGET_VAL:
            break

        # ── Robustly extract text from the SE preference format ──────────────
        # Schema variant 1: {question, answers: [{text, pm_score}, ...]}
        # Schema variant 2: {question, human_ref_A, human_ref_B, ...}
        # Schema variant 3: flat text field
        text = ""
        question = (
            doc.get("question") or
            doc.get("prompt") or
            doc.get("context") or
            ""
        )
        if not question and "text" in doc:
            # Plain text fallback
            text = doc["text"] or ""
        else:
            # Try to pick the highest-scored answer
            best_answer = ""
            answers = doc.get("answers", [])
            if isinstance(answers, list) and answers:
                try:
                    scored = [(a.get("pm_score", 0), a.get("text", ""))
                              for a in answers if isinstance(a, dict)]
                    if scored:
                        best_answer = max(scored, key=lambda kv: kv[0])[1]
                    elif isinstance(answers[0], str):
                        best_answer = answers[0]
                except Exception:
                    best_answer = str(answers[0]) if answers else ""
            # Fall back to human_ref_A (RLHF-style layout)
            if not best_answer:
                best_answer = doc.get("human_ref_A") or doc.get("response") or ""
            text = f"{question}\n\n{best_answer}".strip() if best_answer else question.strip()

        if len(text) < 50:
            continue

        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        doc_idx += 1

        if doc_idx % VAL_MODULO == 0 and (val_tokens + local_val) < TARGET_VAL:
            f_val.write(struct.pack(f"{len(ids)}i", *ids))
            local_val += len(ids)
        elif local_train < TARGET_THIS:
            f_train.write(struct.pack(f"{len(ids)}i", *ids))
            local_train += len(ids)

        if doc_idx % 5_000 == 0:
            progress("stackexchange", local_train, TARGET_THIS,
                     val_tokens + local_val, doc_idx,
                     t0, disk_free_gb())

    elapsed = time.time() - t0
    print(f"\n  ✔  StackExchange done — {human(local_train)} train tokens "
          f"| {human(local_val)} val tokens | {elapsed/3600:.1f}h")
    return train_tokens + local_train, val_tokens + local_val


# ── Source 4: Wikipedia (long-form books substitute) ─────────────────────────
#
# bookcorpus/bookcorpus and pg19 both fail with:
#   RuntimeError: Dataset scripts are no longer supported, but found bookcorpus.py
# Replacement: wikimedia/wikipedia "20231101.en" — long-form, coherent articles
# that provide equivalent narrative-rich text for language model training.

def stream_bookcorpus(tok, f_train, f_val,
                      train_tokens: int, val_tokens: int) -> tuple[int, int]:
    from datasets import load_dataset

    TARGET_THIS = TARGET["bookcorpus"]
    print(f"\n{'═'*68}")
    print(f"  [4/4] Wikipedia (books substitute) — wikimedia/wikipedia 20231101.en")
    print(f"  Target : {human(TARGET_THIS)} train tokens  (long-form articles, ≥500 chars)")
    print(f"  Note   : bookcorpus/bookcorpus deprecated (old loading script).")
    print(f"           Wikipedia provides equivalent long-form narrative text.")
    print(f"{'═'*68}")

    EOS = tok.eos_token_id
    BOS = tok.bos_token_id if tok.bos_token_id is not None else EOS

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )

    local_train = 0
    local_val   = 0
    doc_idx     = 0
    t0          = time.time()

    for doc in ds:
        if local_train >= TARGET_THIS and (val_tokens + local_val) >= TARGET_VAL:
            break

        text = doc.get("text", "").strip()
        # Only use substantial articles (skip stubs < 500 chars)
        if len(text) < 500:
            continue

        ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
        doc_idx += 1

        if doc_idx % VAL_MODULO == 0 and (val_tokens + local_val) < TARGET_VAL:
            f_val.write(struct.pack(f"{len(ids)}i", *ids))
            local_val += len(ids)
        elif local_train < TARGET_THIS:
            f_train.write(struct.pack(f"{len(ids)}i", *ids))
            local_train += len(ids)

        if doc_idx % 10_000 == 0:
            progress("wikipedia", local_train, TARGET_THIS,
                     val_tokens + local_val, doc_idx,
                     t0, disk_free_gb())

    elapsed = time.time() - t0
    print(f"\n  ✔  Wikipedia (books) done — {human(local_train)} train tokens "
          f"| {human(local_val)} val tokens | {elapsed/3600:.1f}h")
    return train_tokens + local_train, val_tokens + local_val


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Phase 3 mixed corpus (~2B tokens)"
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed sources (not yet implemented; "
                             "re-run is idempotent via the .downloaded flag)")
    args = parser.parse_args()

    # ── Sanity checks ─────────────────────────────────────────────────────────
    free_gb = disk_free_gb()
    print(f"\n  Disk free : {free_gb:.1f} GB")
    if free_gb < 12:
        print(f"  ⚠  Warning: less than 12 GB free. "
              f"Phase 3 data needs ~8 GB. Proceeding anyway …")

    if os.path.exists(DONE_FLAG):
        print(f"\n  ✔  Phase 3 data already prepared at {PHASE3_DIR}")
        print(f"     Delete {DONE_FLAG} to re-run.\n")
        return

    os.makedirs(PHASE3_DIR, exist_ok=True)
    os.makedirs(LOG_ROOT,   exist_ok=True)

    # Remove partial files from any previous interrupted run
    for p in [TRAIN_BIN, VAL_BIN]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed partial file: {p}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nLoading tokenizer …")
    tok = get_tokenizer()

    total_train = 0
    total_val   = 0
    t_global    = time.time()
    source_stats = {}

    print(f"\n{'═'*68}")
    print(f"  Phase 3 Data Preparation")
    print(f"  Target : ~2.0B train + ~20M val tokens")
    print(f"  Output : {PHASE3_DIR}")
    print(f"  Disk   : {free_gb:.1f} GB free (~8 GB needed)")
    print(f"{'═'*68}\n")

    with open(TRAIN_BIN, "wb") as f_train, open(VAL_BIN, "wb") as f_val:

        # 1. OpenWebText ──────────────────────────────────────────────────────
        before_train = total_train
        before_val   = total_val
        total_train, total_val = stream_openwebtext(tok, f_train, f_val,
                                                    total_train, total_val)
        source_stats["openwebtext"] = {
            "train_tokens": total_train - before_train,
            "val_tokens":   total_val - before_val,
        }
        print(f"  Disk free after OWT : {disk_free_gb():.1f} GB", flush=True)

        # 2. C4 ───────────────────────────────────────────────────────────────
        before_train = total_train
        before_val   = total_val
        total_train, total_val = stream_c4(tok, f_train, f_val,
                                           total_train, total_val)
        source_stats["c4"] = {
            "train_tokens": total_train - before_train,
            "val_tokens":   total_val - before_val,
        }
        print(f"  Disk free after C4  : {disk_free_gb():.1f} GB", flush=True)

        # 3. StackExchange ────────────────────────────────────────────────────
        before_train = total_train
        before_val   = total_val
        total_train, total_val = stream_stackexchange(tok, f_train, f_val,
                                                      total_train, total_val)
        source_stats["stackexchange"] = {
            "train_tokens": total_train - before_train,
            "val_tokens":   total_val - before_val,
        }
        print(f"  Disk free after SE  : {disk_free_gb():.1f} GB", flush=True)

        # 4. BookCorpus ───────────────────────────────────────────────────────
        before_train = total_train
        before_val   = total_val
        total_train, total_val = stream_bookcorpus(tok, f_train, f_val,
                                                   total_train, total_val)
        source_stats["bookcorpus"] = {
            "train_tokens": total_train - before_train,
            "val_tokens":   total_val - before_val,
        }
        print(f"  Disk free after BC  : {disk_free_gb():.1f} GB", flush=True)

    # ── Final stats ───────────────────────────────────────────────────────────
    elapsed     = time.time() - t_global
    train_gb    = os.path.getsize(TRAIN_BIN) / 1e9
    val_mb      = os.path.getsize(VAL_BIN)   / 1e6

    meta = {
        "total_train_tokens": total_train,
        "total_val_tokens":   total_val,
        "train_bin_gb":       round(train_gb, 3),
        "val_bin_mb":         round(val_mb,   1),
        "elapsed_hours":      round(elapsed / 3600, 2),
        "tokenizer":          "gpt2",
        "block_size":         512,
        "sources":            source_stats,
        "percentages": {
            k: round(v["train_tokens"] / max(total_train, 1) * 100, 1)
            for k, v in source_stats.items()
        },
    }

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    open(DONE_FLAG, "w").close()

    print(f"\n{'═'*68}")
    print(f"  Phase 3 Data Preparation Complete!")
    print(f"{'═'*68}")
    print(f"  Total train tokens : {human(total_train)}")
    print(f"  Total val   tokens : {human(total_val)}")
    print(f"  Train file         : {train_gb:.2f} GB")
    print(f"  Val   file         : {val_mb:.1f} MB")
    print(f"  Total time         : {elapsed/3600:.1f}h")
    print(f"\n  Source breakdown:")
    for src, stats in source_stats.items():
        pct = stats["train_tokens"] / max(total_train, 1) * 100
        print(f"    {src:18s}: {human(stats['train_tokens']):>10}  ({pct:.1f}%)")
    print(f"\n  Disk free now: {disk_free_gb():.1f} GB")
    print(f"  Meta: {META_JSON}")
    print(f"{'═'*68}")
    print(f"\n→ Next: nohup python train_phase3.py --wandb > logs/train_phase3.log 2>&1 &")


if __name__ == "__main__":
    main()

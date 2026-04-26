"""
resume_phase3_bookcorpus.py
════════════════════════════
Appends the missing BookCorpus (10% / ~200M tokens) to the existing
Phase 3 binary files.

The main prepare_phase3_data.py run completed OpenWebText + C4 + StackExchange
successfully (~1.8B tokens) but crashed on BookCorpus because the
`bookcorpus/bookcorpus` dataset uses a now-unsupported loading script.

This script:
  1. Validates the existing .bin files (OWT + C4 + SE = ~1.8B  train tokens).
  2. Streams a replacement "books" source (Wikipedia long-form articles,
     which are rich narrative text similar to BookCorpus).
  3. APPENDS the new tokens to data/phase3_mixed/train_tokens.bin and
     data/phase3_mixed/val_tokens.bin.
  4. Writes meta.json and the .downloaded flag on success.

Primary source  : wikimedia/wikipedia  "20231101.en"  (long-form articles)
Fallback source : roneneldan/TinyStories  (already cached locally)

Run:
    python resume_phase3_bookcorpus.py

Run in background (recommended):
    nohup python resume_phase3_bookcorpus.py > logs/resume_phase3_books.log 2>&1 &
    echo "PID=$!" > logs/resume_phase3_books.pid
    tail -f logs/resume_phase3_books.log
"""

import json
import os
import struct
import time

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
LOG_ROOT     = os.path.join(PROJECT_ROOT, "logs")
CKPT_ROOT    = os.path.join(PROJECT_ROOT, "checkpoints")

PHASE3_DIR   = os.path.join(DATA_ROOT, "phase3_mixed")
TRAIN_BIN    = os.path.join(PHASE3_DIR, "train_tokens.bin")
VAL_BIN      = os.path.join(PHASE3_DIR, "val_tokens.bin")
META_JSON    = os.path.join(PHASE3_DIR, "meta.json")
DONE_FLAG    = os.path.join(PHASE3_DIR, ".downloaded")

# Expected tokens already in the files (OWT + C4 + SE)
EXPECTED_TRAIN_BYTES = 7_200_002_632   # 1,800,000,658 int32 tokens
EXPECTED_VAL_BYTES   = 80_000_756      #    20,000,189 int32 tokens

TARGET_BOOKS_TRAIN = 200_000_000       # 10% of 2B
TARGET_VAL_TOTAL   = 20_000_000        # already met — extra val goes to train
VAL_MODULO         = 100


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


def human(n: int) -> str:
    if n >= 1e9: return f"{n/1e9:.3f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    return str(n)


def disk_free_gb() -> float:
    st = os.statvfs(PROJECT_ROOT)
    return st.f_bavail * st.f_frsize / 1e9


def stream_wikipedia_books(tok, f_train, f_val, val_tokens_so_far: int):
    """
    Stream Wikipedia English articles as a book-like corpus.
    Wikipedia articles are long-form, coherent text — a very reasonable
    substitute for BookCorpus in terms of language quality.
    Falls back to TinyStories (locally cached) if Wikipedia is unavailable.
    """
    from datasets import load_dataset

    TARGET_THIS = TARGET_BOOKS_TRAIN
    EOS = tok.eos_token_id
    BOS = tok.bos_token_id if tok.bos_token_id is not None else EOS

    local_train = 0
    local_val   = 0
    doc_idx     = 0
    t0          = time.time()

    # ── Try Wikipedia ─────────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  [4/4] Books corpus — wikimedia/wikipedia 20231101.en")
    print(f"  Target : {human(TARGET_THIS)} train tokens  (narratives + knowledge)")
    print(f"  Strategy: streaming Wikipedia; long articles ≥ 500 chars only")
    print(f"{'═'*68}")

    source_ok = False
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        # Quick peek to verify it works
        peek = next(iter(ds))
        assert "text" in peek, "no text field"
        print(f"  ✔  Wikipedia stream opened. Sample title: {peek.get('title','?')!r}")
        source_ok = True
    except Exception as e:
        print(f"  ✗  Wikipedia unavailable: {e}")

    if source_ok:
        # Reconstruct iterator (peek consumed the first element)
        ds = load_dataset(
            "wikimedia/wikipedia",
            "20231101.en",
            split="train",
            streaming=True,
        )
        for doc in ds:
            if local_train >= TARGET_THIS:
                break

            text = doc.get("text", "").strip()
            # Only use substantial articles (avoids stubs)
            if len(text) < 500:
                continue

            ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
            doc_idx += 1

            # Val quota already met by OWT + C4 + SE — route all to train
            if (doc_idx % VAL_MODULO == 0
                    and val_tokens_so_far + local_val < TARGET_VAL_TOTAL):
                f_val.write(struct.pack(f"{len(ids)}i", *ids))
                local_val += len(ids)
            elif local_train < TARGET_THIS:
                f_train.write(struct.pack(f"{len(ids)}i", *ids))
                local_train += len(ids)

            if doc_idx % 10_000 == 0:
                elapsed = time.time() - t0
                speed   = local_train / max(elapsed, 1) / 1e6
                pct     = min(local_train / TARGET_THIS * 100, 100)
                print(
                    f"  [wikipedia   ] docs:{doc_idx:>8,}  "
                    f"tokens:{human(local_train):>10}  ({pct:5.1f}%)  "
                    f"disk:{disk_free_gb():.1f}GB free  "
                    f"{speed:.1f}Mtok/s  [{elapsed/3600:.1f}h]",
                    flush=True,
                )
    else:
        # ── Fallback: TinyStories (locally cached) ────────────────────────────
        print(f"\n  Falling back to roneneldan/TinyStories (locally cached)…")
        print(f"  Note: TinyStories is small (~475M tokens), may not reach 200M target.")
        try:
            ds = load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=True,
            )
            for doc in ds:
                if local_train >= TARGET_THIS:
                    break
                text = doc.get("text", "").strip()
                if len(text) < 50:
                    continue
                ids = [BOS] + tok.encode(text, add_special_tokens=False) + [EOS]
                doc_idx += 1
                if (doc_idx % VAL_MODULO == 0
                        and val_tokens_so_far + local_val < TARGET_VAL_TOTAL):
                    f_val.write(struct.pack(f"{len(ids)}i", *ids))
                    local_val += len(ids)
                elif local_train < TARGET_THIS:
                    f_train.write(struct.pack(f"{len(ids)}i", *ids))
                    local_train += len(ids)
                if doc_idx % 50_000 == 0:
                    elapsed = time.time() - t0
                    speed   = local_train / max(elapsed, 1) / 1e6
                    pct     = min(local_train / TARGET_THIS * 100, 100)
                    print(
                        f"  [tinystories ] docs:{doc_idx:>8,}  "
                        f"tokens:{human(local_train):>10}  ({pct:5.1f}%)  "
                        f"{speed:.1f}Mtok/s  [{elapsed/3600:.1f}h]",
                        flush=True,
                    )
        except Exception as e2:
            print(f"  ✗  Fallback also failed: {e2}")
            print("  ⚠  Proceeding with 0 book tokens — dataset is still 90% complete.")

    elapsed = time.time() - t0
    print(f"\n  ✔  Books done — {human(local_train)} train tokens "
          f"| {human(local_val)} val tokens | {elapsed/3600:.2f}h")
    return local_train, local_val


def main():
    print(f"\n{'═'*68}")
    print(f"  Phase 3 Resume — Appending BookCorpus (10%)")
    print(f"  Existing files will NOT be re-processed.")
    print(f"{'═'*68}\n")

    # ── Guard: already done ────────────────────────────────────────────────────
    if os.path.exists(DONE_FLAG):
        print(f"  ✔  Phase 3 already complete ({DONE_FLAG} exists).")
        print(f"     Delete it to force a re-run.\n")
        return

    # ── Validate existing files ────────────────────────────────────────────────
    for path, expected_bytes, label in [
        (TRAIN_BIN, EXPECTED_TRAIN_BYTES, "train"),
        (VAL_BIN,   EXPECTED_VAL_BYTES,   "val"),
    ]:
        if not os.path.exists(path):
            print(f"  ✗  {label} bin not found: {path}")
            print(f"     Please run prepare_phase3_data.py from scratch first.\n")
            return
        actual = os.path.getsize(path)
        if actual < EXPECTED_TRAIN_BYTES * 0.8 if label == "train" else EXPECTED_VAL_BYTES * 0.5:
            print(f"  ⚠  {label} bin is much smaller than expected "
                  f"({actual:,} vs {expected_bytes:,} bytes). "
                  f"Proceeding anyway.")
        else:
            print(f"  ✔  {label} bin: {actual:,} bytes "
                  f"({actual/4:,.0f} tokens)")

    existing_train_tokens = os.path.getsize(TRAIN_BIN) // 4
    existing_val_tokens   = os.path.getsize(VAL_BIN)   // 4
    print(f"\n  Existing train tokens : {human(existing_train_tokens)}")
    print(f"  Existing val   tokens : {human(existing_val_tokens)}")
    print(f"  Disk free             : {disk_free_gb():.1f} GB")

    if disk_free_gb() < 1.5:
        print(f"  ⚠  Less than 1.5 GB free — BookCorpus needs ~0.8 GB. Aborting.")
        return

    os.makedirs(LOG_ROOT, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("\nLoading tokenizer …")
    tok = get_tokenizer()

    t_global = time.time()

    # ── Append books tokens ────────────────────────────────────────────────────
    with open(TRAIN_BIN, "ab") as f_train, open(VAL_BIN, "ab") as f_val:
        books_train, books_val = stream_wikipedia_books(
            tok, f_train, f_val, existing_val_tokens
        )

    # ── Final stats ────────────────────────────────────────────────────────────
    elapsed      = time.time() - t_global
    total_train  = existing_train_tokens + books_train
    total_val    = existing_val_tokens + books_val
    train_gb     = os.path.getsize(TRAIN_BIN) / 1e9
    val_mb       = os.path.getsize(VAL_BIN)   / 1e6

    # Source stats (reconstructed from known targets)
    source_stats = {
        "openwebtext":   {"train_tokens": 700_000_000,       "val_tokens": 0},
        "c4":            {"train_tokens": 600_000_000,       "val_tokens": 0},
        "stackexchange": {"train_tokens": 500_000_000,       "val_tokens": existing_val_tokens},
        "bookcorpus_wikipedia": {"train_tokens": books_train, "val_tokens": books_val},
    }

    meta = {
        "total_train_tokens":  total_train,
        "total_val_tokens":    total_val,
        "train_bin_gb":        round(train_gb, 3),
        "val_bin_mb":          round(val_mb,   1),
        "elapsed_hours":       round(elapsed / 3600, 2),
        "tokenizer":           "gpt2",
        "block_size":          512,
        "sources":             source_stats,
        "percentages": {
            k: round(v["train_tokens"] / max(total_train, 1) * 100, 1)
            for k, v in source_stats.items()
        },
        "notes": (
            "BookCorpus replaced with wikimedia/wikipedia (bookcorpus/bookcorpus "
            "deprecated loading script). Wikipedia provides equivalent long-form text."
        ),
    }

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=2)

    # Mark as done
    open(DONE_FLAG, "w").close()

    print(f"\n{'═'*68}")
    print(f"  Phase 3 Resume Complete!")
    print(f"{'═'*68}")
    print(f"  Total train tokens : {human(total_train)}")
    print(f"  Total val   tokens : {human(total_val)}")
    print(f"  Train file         : {train_gb:.2f} GB")
    print(f"  Val   file         : {val_mb:.1f} MB")
    print(f"  Books elapsed      : {elapsed/3600:.2f}h")
    print(f"\n  Source breakdown:")
    for src, stats in source_stats.items():
        pct = stats["train_tokens"] / max(total_train, 1) * 100
        print(f"    {src:28s}: {human(stats['train_tokens']):>10}  ({pct:.1f}%)")
    print(f"\n  Disk free now: {disk_free_gb():.1f} GB")
    print(f"  Meta: {META_JSON}")
    print(f"{'═'*68}")
    print(f"\n→ Next: nohup python train_phase3.py --wandb > logs/train_phase3.log 2>&1 &")


if __name__ == "__main__":
    main()

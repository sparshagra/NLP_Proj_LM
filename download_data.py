"""
download_data.py
────────────────
One-time script to download and cache datasets for curriculum phases.
Run ONCE on the SSH machine; training scripts load from disk every run.

Usage:
    python download_data.py --phase 1     # TinyStories (Arrow format, ~1.8 GB)
    python download_data.py --phase 2     # Wikipedia EN (binary .bin, ~4.04 GB)
    python download_data.py --phase all   # both

Phase mapping:
    Phase 1 → roneneldan/TinyStories          (children's stories, Arrow)
    Phase 2 → wikimedia/wikipedia 20231101.en (formal text, binary token cache)
               ↳ Equivalent to Kaggle: ffatty/plaintext-wikipedia-full-english
                 but uses HuggingFace streaming — no Kaggle credentials needed.

Storage estimate (after both phases downloaded):
    Phase 1 data  : ~1.8 GB   (Arrow)
    Phase 2 data  : ~4.1 GB   (train_tokens.bin + val_tokens.bin)
    Phase 1 ckpts : ~5.6 GB   (already saved)
    Phase 2 ckpts : ~5.6 GB   (3 epochs × 1.4 GB + best_model.pt)
    ─────────────────────────
    Total est.    : ~17 GB   (well within the 70 GB disk)
"""

import argparse
import os
import struct
import sys
import time

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — TinyStories (Arrow format, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def download_phase1():
    save_dir = os.path.join(DATA_ROOT, "phase1_tinystories")
    flag     = os.path.join(save_dir, ".downloaded")

    if os.path.exists(flag):
        print("✔  Phase 1 (TinyStories) already cached at:", save_dir)
        return

    from datasets import load_dataset

    print("\n" + "═" * 62)
    print("  Downloading Phase 1 — TinyStories (roneneldan/TinyStories)")
    print("═" * 62)
    os.makedirs(save_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")
    ds.save_to_disk(save_dir)
    open(flag, "w").close()
    print(f"\n✔  Saved to: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Wikipedia EN — streaming binary token cache
# ─────────────────────────────────────────────────────────────────────────────

def download_phase2():
    """
    Streams English Wikipedia from HuggingFace (wikimedia/wikipedia, 20231101.en),
    tokenises articles with the GPT-2 tokenizer (same as Phase 1), and writes
    raw int32 token arrays to two binary files:

        data/phase2_wikipedia/train_tokens.bin  — 1 billion tokens  (~4.0 GB)
        data/phase2_wikipedia/val_tokens.bin    — 10 million tokens  (~40 MB)
        data/phase2_wikipedia/meta.txt          — token/article stats
        data/phase2_wikipedia/.downloaded       — sentinel flag

    Token encoding: raw little-endian int32 (compatible with np.memmap int32).

    Why not Kaggle directly?
        The Kaggle dataset (ffatty/plaintext-wikipedia-full-english) is the
        same content (~20 GB compressed). Downloading it requires Kaggle API
        credentials and would download the full dump.  HuggingFace streaming
        lets us take exactly as many tokens as we want (~4 GB instead of ~20 GB).

    Estimated time: 2–4 hours (network + tokenisation bound).
    """
    save_dir  = os.path.join(DATA_ROOT, "phase2_wikipedia")
    flag      = os.path.join(save_dir, ".downloaded")
    train_bin = os.path.join(save_dir, "train_tokens.bin")
    val_bin   = os.path.join(save_dir, "val_tokens.bin")

    if os.path.exists(flag):
        print("✔  Phase 2 (Wikipedia) already cached at:", save_dir)
        return

    os.makedirs(save_dir, exist_ok=True)

    # ── Token targets ──────────────────────────────────────────────────────────
    TARGET_TRAIN = 1_000_000_000   # 1B tokens → ~4.0 GB
    TARGET_VAL   =    10_000_000   # 10M tokens → ~40 MB

    # Remove any partial files from previous interrupted runs
    for p in [train_bin, val_bin]:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed partial file: {p}")

    # ── Tokenizer (load from Phase 1 cache if available) ──────────────────────
    print("Loading GPT-2 tokenizer …")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Run: pip install transformers")

    tok_cache = os.path.join(
        os.path.dirname(__file__), "checkpoints", "phase1", "tokenizer"
    )
    if os.path.isdir(tok_cache) and os.listdir(tok_cache):
        tok = AutoTokenizer.from_pretrained(tok_cache)
        print(f"  Loaded from Phase 1 cache: {tok_cache}")
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
        print("  Downloaded GPT-2 tokenizer from HuggingFace")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    EOS_ID = tok.eos_token_id
    BOS_ID = tok.bos_token_id if tok.bos_token_id is not None else EOS_ID
    print(f"  Vocab: {tok.vocab_size:,}  BOS={BOS_ID}  EOS={EOS_ID}")

    # ── Stream Wikipedia ───────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print("  Streaming wikimedia/wikipedia (20231101.en) …")
    print("  (Same content as Kaggle: ffatty/plaintext-wikipedia-full-english)")
    print(f"  Target  : {TARGET_TRAIN / 1e9:.1f}B train + "
          f"{TARGET_VAL / 1e6:.0f}M val tokens")
    print(f"  Disk est: ~{(TARGET_TRAIN + TARGET_VAL) * 4 / 1e9:.2f} GB")
    print("═" * 62 + "\n")

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Run: pip install datasets")

    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        streaming=True,
    )["train"]   # Wikipedia only has a 'train' split; we carve val ourselves

    train_tokens  = 0
    val_tokens    = 0
    articles_seen = 0
    t0            = time.time()

    with open(train_bin, "wb") as f_train, open(val_bin, "wb") as f_val:
        for article in ds:
            # Stop when both targets are met
            if train_tokens >= TARGET_TRAIN and val_tokens >= TARGET_VAL:
                break

            text = article.get("text", "")
            if not text or len(text) < 200:      # skip stubs
                continue

            # Tokenise the article
            ids = tok.encode(text, add_special_tokens=False)
            ids = [BOS_ID] + ids + [EOS_ID]
            articles_seen += 1

            # Route 1% of articles to validation (round-robin)
            if articles_seen % 100 == 0 and val_tokens < TARGET_VAL:
                f_val.write(struct.pack(f"{len(ids)}i", *ids))
                val_tokens += len(ids)
            elif train_tokens < TARGET_TRAIN:
                f_train.write(struct.pack(f"{len(ids)}i", *ids))
                train_tokens += len(ids)

            # Progress every 5 000 articles
            if articles_seen % 5_000 == 0:
                elapsed = time.time() - t0
                speed   = (train_tokens + val_tokens) / elapsed / 1e6
                pct_t   = min(train_tokens / TARGET_TRAIN * 100, 100)
                pct_v   = min(val_tokens   / TARGET_VAL   * 100, 100)
                size_gb = (
                    (os.path.getsize(train_bin) if os.path.exists(train_bin) else 0) +
                    (os.path.getsize(val_bin)   if os.path.exists(val_bin)   else 0)
                ) / 1e9
                print(
                    f"  Art:{articles_seen:>7,}  "
                    f"Train:{train_tokens / 1e9:.3f}B ({pct_t:5.1f}%)  "
                    f"Val:{val_tokens / 1e6:.1f}M ({pct_v:5.1f}%)  "
                    f"Disk:{size_gb:.2f}GB  "
                    f"{speed:.1f}Mtok/s  [{elapsed / 3600:.1f}h]",
                    flush=True,
                )

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed  = time.time() - t0
    train_gb = os.path.getsize(train_bin) / 1e9
    val_mb   = os.path.getsize(val_bin)   / 1e6

    print(f"\n{'═' * 62}")
    print(f"  Wikipedia download complete!")
    print(f"  Articles processed : {articles_seen:,}")
    print(f"  Train tokens       : {train_tokens:,}  ({train_gb:.2f} GB)")
    print(f"  Val   tokens       : {val_tokens:,}  ({val_mb:.1f} MB)")
    print(f"  Total time         : {elapsed / 3600:.1f}h")
    print(f"{'═' * 62}")

    # Write meta file for reference
    meta_path = os.path.join(save_dir, "meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"dataset=wikimedia/wikipedia:20231101.en\n")
        f.write(f"equivalent_kaggle=ffatty/plaintext-wikipedia-full-english\n")
        f.write(f"tokenizer=gpt2\n")
        f.write(f"train_tokens={train_tokens}\n")
        f.write(f"val_tokens={val_tokens}\n")
        f.write(f"articles={articles_seen}\n")
        f.write(f"train_bin={train_bin}\n")
        f.write(f"val_bin={val_bin}\n")

    open(flag, "w").close()
    print(f"\n✔  Cached to: {save_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download curriculum datasets (Phase 1: TinyStories, Phase 2: Wikipedia)"
    )
    parser.add_argument(
        "--phase",
        default="1",
        choices=["1", "2", "all"],
        help="Which dataset to download (default: 1)",
    )
    args = parser.parse_args()

    if args.phase in ("1", "all"):
        download_phase1()
    if args.phase in ("2", "all"):
        download_phase2()

    print("\n✔  All requested datasets ready.\n")


if __name__ == "__main__":
    main()

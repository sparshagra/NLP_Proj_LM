"""
evaluate_all_phases.py
══════════════════════════════════════════════════════════════════════════
Comprehensive evaluation of all three curriculum-learning checkpoints.

Metrics computed per phase:
  1. Perplexity (PPL)      — exp(mean cross-entropy loss on held-out val set)
  2. BLEU-4               — 4-gram precision vs. reference continuations
  3. ROUGE-L (F1)         — LCS-based overlap vs. reference continuations
  4. Distinct-1 / Distinct-2 — lexical / phrase-level diversity
  5. Self-BLEU            — inter-output diversity (lower = more diverse)
  6. Repetition Rate      — fraction of tokens that are exact repeats

Usage:
    python evaluate_all_phases.py            # runs all three phases
    python evaluate_all_phases.py --phase 1  # single phase
    python evaluate_all_phases.py --no_gpu   # force CPU

Outputs:
    eval_results.json       — machine-readable results
    eval_results_table.txt  — ASCII summary table
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.model import DecoderOnlyTransformer

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("pip install transformers")

try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("pip install rouge-score")

try:
    import sacrebleu
except ImportError:
    raise ImportError("pip install sacrebleu")

import nltk

# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions — all hyperparameters kept in one place
# ─────────────────────────────────────────────────────────────────────────────

PHASE_CONFIGS = {
    1: dict(
        name        = "Phase 1 — TinyStories",
        ckpt        = "checkpoints/phase1/best_model.pt",
        tok_dir     = "checkpoints/phase1/tokenizer",
        data_type   = "hf_arrow",           # HuggingFace Arrow dataset
        data_dir    = "data/phase1_tinystories",
        val_split   = "validation",
        text_col    = "text",
        # Model architecture (124 M)
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_len=512, dropout=0.0,
        reported_ppl = 3.28,
    ),
    2: dict(
        name        = "Phase 2 — Wikipedia",
        ckpt        = "checkpoints/phase2/best_model.pt",
        tok_dir     = "checkpoints/phase1/tokenizer",   # same tokenizer throughout
        data_type   = "bin",                # pre-tokenised memory-mapped binary
        val_bin     = "data/phase2_wikipedia/val_tokens.bin",
        # Model architecture (124 M)
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_len=512, dropout=0.0,
        reported_ppl = 20.47,
    ),
    3: dict(
        name        = "Phase 3 — Mixed Corpus",
        ckpt        = "checkpoints/phase3/best_model.pt",
        tok_dir     = "checkpoints/phase1/tokenizer",
        data_type   = "bin",
        val_bin     = "data/phase3_mixed/val_tokens.bin",
        # Model architecture (124 M)
        d_model=768, n_heads=12, n_layers=12, d_ff=3072,
        max_len=512, dropout=0.0,
        reported_ppl = 30.12,
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# Fixed evaluation prompts (same across all phases for fair comparison)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_PROMPTS = [
    # Phase 1 style — simple narrative
    "Once upon a time there was a little girl named",
    "The small rabbit looked up at the sky and",
    "One day in the forest, the animals decided to",
    # Phase 2 style — encyclopedic
    "According to recent scientific research, the primary cause of",
    "The history of the Roman Empire began when",
    "Machine learning is a subfield of artificial intelligence that",
    # Phase 3 style — web / technical
    "The best way to implement a binary search tree in Python is",
    "Recent advancements in natural language processing have shown",
    "To deploy a web application at scale, engineers typically",
    # Cross-domain
    "Climate change is affecting the world because",
]

# Corresponding "reference" continuations — these are gold text fragments
# used for BLEU / ROUGE evaluation. Deliberately distinct from prompts.
REFERENCE_CONTINUATIONS = [
    "named Lily who loved to play in the garden with her friends.",
    "she saw a beautiful rainbow and felt very happy inside.",
    "they all agreed to have a picnic by the river together.",
    "is a combination of genetic and environmental factors over time.",
    "Julius Caesar crossed the Rubicon river in 49 BC with his legion.",
    "uses statistical techniques and algorithms to build systems from data.",
    "to use a Node class containing a value and left and right child pointers.",
    "transformer architectures have dramatically improved text understanding tasks.",
    "use containerization tools like Docker and orchestration platforms like Kubernetes.",
    "of rising greenhouse gas emissions from industrial and agricultural activities.",
]

assert len(EVAL_PROMPTS) == len(REFERENCE_CONTINUATIONS), "Mismatch in prompt/reference lists"

# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(cfg: dict, device: torch.device):
    """Load tokenizer + model from a phase config dict."""
    base = os.path.dirname(__file__)

    # Tokenizer
    tok_dir = os.path.join(base, cfg["tok_dir"])
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = DecoderOnlyTransformer(
        vocab_size = tokenizer.vocab_size,
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        d_ff       = cfg["d_ff"],
        max_len    = cfg["max_len"] + 1,
        dropout    = cfg["dropout"],
        pad_id     = tokenizer.pad_token_id,
    ).to(device)

    ckpt_path = os.path.join(base, cfg["ckpt"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  ✓ Loaded checkpoint: {ckpt_path}")
    print(f"    Epoch: {ckpt.get('epoch', '?')}  |  Saved val PPL: {ckpt.get('val_ppl', '?')}")
    return model, tokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Metric 1 — Perplexity (on held-out validation tokens)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_perplexity(model, tokenizer, cfg: dict, device: torch.device,
                        max_batches: int = 100, block_size: int = 512) -> float:
    """
    Compute perplexity on the held-out validation set.
    For Phase 1 (HF Arrow format) we sample text and re-tokenise.
    For Phases 2/3 we read directly from the pre-tokenised binary files.
    """
    base = os.path.dirname(__file__)
    total_loss  = 0.0
    total_count = 0

    if cfg["data_type"] == "bin":
        # Memory-mapped binary file of int32 tokens
        val_path = os.path.join(base, cfg["val_bin"])
        data = np.memmap(val_path, dtype=np.int32, mode="r")
        n_blocks = min(max_batches, len(data) // block_size)
        indices = np.random.default_rng(42).choice(
            len(data) // block_size, size=n_blocks, replace=False
        )
        for idx in indices:
            chunk  = data[idx * block_size : (idx + 1) * block_size]
            tokens = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
            x, y   = tokens[:, :-1], tokens[:, 1:]
            if x.shape[1] < 2:
                continue
            _, loss = model(x, y)
            if loss is not None and not torch.isnan(loss):
                total_loss  += loss.item() * y.numel()
                total_count += y.numel()

    else:
        # HuggingFace Arrow dataset (Phase 1)
        try:
            from datasets import load_from_disk
        except ImportError:
            raise ImportError("pip install datasets")

        data_path = os.path.join(base, cfg["data_dir"])
        ds = load_from_disk(data_path)
        val_ds = ds[cfg["val_split"]]
        rng = np.random.default_rng(42)
        indices = rng.choice(min(len(val_ds), max_batches * 4),
                             size=min(max_batches, len(val_ds)), replace=False)
        for idx in indices:
            text = val_ds[int(idx)][cfg["text_col"]]
            if not text or len(text) < 20:
                continue
            enc = tokenizer(text, return_tensors="pt",
                            max_length=block_size, truncation=True)
            tokens = enc["input_ids"].to(device)
            if tokens.shape[1] < 4:
                continue
            x, y = tokens[:, :-1], tokens[:, 1:]
            _, loss = model(x, y)
            if loss is not None and not torch.isnan(loss):
                total_loss  += loss.item() * y.numel()
                total_count += y.numel()

    if total_count == 0:
        return float("inf")
    avg_loss = total_loss / total_count
    return math.exp(avg_loss)

# ─────────────────────────────────────────────────────────────────────────────
# Text generation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: torch.device,
                  max_new: int = 80, temperature: float = 0.8,
                  top_k: int = 50, top_p: float = 0.95,
                  rep_penalty: float = 1.3) -> str:
    """Generate a completion string for a given prompt string."""
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new            = max_new,
        temperature        = temperature,
        top_k              = top_k,
        top_p              = top_p,
        repetition_penalty = rep_penalty,
        eos_id             = tokenizer.eos_token_id,
        max_seq_len        = 512,
    )
    # Return only the newly generated part (exclude the prompt)
    prompt_len = ids.shape[1]
    new_ids    = out[0, prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

# ─────────────────────────────────────────────────────────────────────────────
# Metric 2 — BLEU-4
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu4(hypotheses: list[str], references: list[str]) -> float:
    """
    Corpus BLEU-4 using sacrebleu.
    hypotheses : list of generated strings
    references : list of reference strings (one per hypothesis)
    """
    # sacrebleu expects refs as list-of-lists
    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score   # 0-100 scale

# ─────────────────────────────────────────────────────────────────────────────
# Metric 3 — ROUGE-L (F1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_rouge_l(hypotheses: list[str], references: list[str]) -> float:
    """Average ROUGE-L F1 across all hypothesis/reference pairs."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for hyp, ref in zip(hypotheses, references):
        s = scorer.score(ref, hyp)
        scores.append(s["rougeL"].fmeasure)
    return float(np.mean(scores)) if scores else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Metric 4 — Distinct-1 and Distinct-2
# ─────────────────────────────────────────────────────────────────────────────

def compute_distinct(texts: list[str]):
    """
    Distinct-1: ratio of unique unigrams to total unigrams.
    Distinct-2: ratio of unique bigrams  to total bigrams.
    Both in [0, 1]; higher → more diverse output.
    """
    all_tokens   = []
    all_bigrams  = []
    for text in texts:
        words = text.lower().split()
        all_tokens.extend(words)
        all_bigrams.extend(zip(words[:-1], words[1:]))

    d1 = len(set(all_tokens))  / max(len(all_tokens),  1)
    d2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    return float(d1), float(d2)

# ─────────────────────────────────────────────────────────────────────────────
# Metric 5 — Self-BLEU  (measures inter-output diversity)
# ─────────────────────────────────────────────────────────────────────────────

def compute_self_bleu(texts: list[str]) -> float:
    """
    For each text treat all other texts as references.
    Lower Self-BLEU → more diverse set of outputs.
    """
    if len(texts) < 2:
        return 0.0
    scores = []
    for i, hyp in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        try:
            result = sacrebleu.corpus_bleu([hyp], [refs])
            scores.append(result.score)
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores))

# ─────────────────────────────────────────────────────────────────────────────
# Metric 6 — Repetition Rate
# ─────────────────────────────────────────────────────────────────────────────

def compute_repetition_rate(texts: list[str], ngram: int = 4) -> float:
    """
    Fraction of n-grams in each text that are repeated within that same text.
    Averaged over all texts. Higher → more repetitive / degenerate output.
    """
    rates = []
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) < ngram + 1:
            rates.append(0.0)
            continue
        grams = [tuple(tokens[i:i+ngram]) for i in range(len(tokens)-ngram+1)]
        unique = len(set(grams))
        total  = len(grams)
        rates.append(1.0 - unique / max(total, 1))
    return float(np.mean(rates)) if rates else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Phase evaluator — runs all metrics for one phase
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_phase(phase_num: int, device: torch.device,
                   ppl_batches: int = 80) -> dict:
    cfg = PHASE_CONFIGS[phase_num]
    print(f"\n{'═'*65}")
    print(f"  Evaluating {cfg['name']}")
    print(f"{'═'*65}")

    t0 = time.time()
    model, tokenizer = load_model(cfg, device)

    # ── 1. Perplexity ────────────────────────────────────────────────────────
    print(f"\n  [1/6] Computing Perplexity on held-out validation set …")
    ppl = compute_perplexity(model, tokenizer, cfg, device, max_batches=ppl_batches)
    print(f"        PPL = {ppl:.4f}  (reported during training: {cfg['reported_ppl']})")

    # ── Generate completions for all eval prompts ────────────────────────────
    print(f"\n  [2/6] Generating completions for {len(EVAL_PROMPTS)} prompts …")
    hypotheses = []
    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        hyp = generate_text(model, tokenizer, prompt, device,
                            max_new=80, temperature=0.8,
                            top_k=50, top_p=0.95, rep_penalty=1.3)
        hypotheses.append(hyp)
        print(f"         [{i:2d}/{len(EVAL_PROMPTS)}] {prompt[:45]!r:<47} → {hyp[:60]!r}")

    # ── 2. BLEU-4 ─────────────────────────────────────────────────────────────
    print(f"\n  [3/6] Computing BLEU-4 …")
    bleu4 = compute_bleu4(hypotheses, REFERENCE_CONTINUATIONS)
    print(f"        BLEU-4 = {bleu4:.4f}")

    # ── 3. ROUGE-L ────────────────────────────────────────────────────────────
    print(f"\n  [4/6] Computing ROUGE-L …")
    rouge_l = compute_rouge_l(hypotheses, REFERENCE_CONTINUATIONS)
    print(f"        ROUGE-L (F1) = {rouge_l:.4f}")

    # ── 4. Distinct-1 / Distinct-2 ────────────────────────────────────────────
    print(f"\n  [5/6] Computing Distinct-1 / Distinct-2 …")
    d1, d2 = compute_distinct(hypotheses)
    print(f"        Distinct-1 = {d1:.4f}  |  Distinct-2 = {d2:.4f}")

    # ── 5. Self-BLEU ──────────────────────────────────────────────────────────
    print(f"\n  [6/6] Computing Self-BLEU and Repetition Rate …")
    self_bleu = compute_self_bleu(hypotheses)
    rep_rate  = compute_repetition_rate(hypotheses, ngram=4)
    print(f"        Self-BLEU       = {self_bleu:.4f}")
    print(f"        Repetition Rate = {rep_rate:.4f}")

    elapsed = time.time() - t0
    print(f"\n  ✓ Phase {phase_num} evaluation complete in {elapsed:.1f}s")

    return {
        "phase"           : phase_num,
        "name"            : cfg["name"],
        "perplexity"      : round(ppl,       4),
        "reported_ppl"    : cfg["reported_ppl"],
        "bleu4"           : round(bleu4,     4),
        "rouge_l"         : round(rouge_l,   4),
        "distinct_1"      : round(d1,        4),
        "distinct_2"      : round(d2,        4),
        "self_bleu"       : round(self_bleu, 4),
        "repetition_rate" : round(rep_rate,  4),
        "hypotheses"      : hypotheses,
        "elapsed_s"       : round(elapsed,   1),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ASCII table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]):
    header = (
        f"{'Phase':<26} | {'PPL':>8} | {'BLEU-4':>8} | "
        f"{'ROUGE-L':>8} | {'Dist-1':>7} | {'Dist-2':>7} | "
        f"{'Self-BLEU':>10} | {'Rep.Rate':>9}"
    )
    sep = "─" * len(header)
    print(f"\n{'═'*len(header)}")
    print("  EVALUATION SUMMARY")
    print(f"{'═'*len(header)}")
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['name']:<26} | {r['perplexity']:>8.3f} | {r['bleu4']:>8.4f} | "
            f"{r['rouge_l']:>8.4f} | {r['distinct_1']:>7.4f} | {r['distinct_2']:>7.4f} | "
            f"{r['self_bleu']:>10.4f} | {r['repetition_rate']:>9.4f}"
        )
    print(sep)
    print("  ↓ PPL better  ↑ BLEU/ROUGE better  ↑ Distinct better  ↓ Self-BLEU/RepRate better")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all curriculum phases")
    parser.add_argument("--phase",    type=int, default=None, choices=[1, 2, 3],
                        help="Evaluate only a single phase (default: all)")
    parser.add_argument("--no_gpu",   action="store_true",
                        help="Force CPU evaluation")
    parser.add_argument("--ppl_batches", type=int, default=80,
                        help="Number of validation batches for PPL (default 80)")
    parser.add_argument("--out",      type=str, default="eval_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    device = torch.device("cpu") if args.no_gpu else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    phases_to_run = [args.phase] if args.phase else [1, 2, 3]
    all_results   = []

    for p in phases_to_run:
        res = evaluate_phase(p, device, ppl_batches=args.ppl_batches)
        all_results.append(res)

    print_summary_table(all_results)

    # Save JSON (strip large hypothesis lists for readability)
    out_path = os.path.join(os.path.dirname(__file__), args.out)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    # Also save clean ASCII table
    table_path = out_path.replace(".json", "_table.txt")
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_summary_table(all_results)
    with open(table_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Table  saved → {table_path}")


if __name__ == "__main__":
    main()

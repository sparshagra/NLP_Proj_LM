"""
Phase 2 Inference Script — Compare Phase 1 vs Phase 2 model outputs
════════════════════════════════════════════════════════════════════
Demonstrates curriculum learning: the model progressively improves from
simple story generation (Phase 1) to encyclopedic text (Phase 2).

Usage:
    python inference_phase2.py                  # compare both phases
    python inference_phase2.py --phase 2        # Phase 2 only
    python inference_phase2.py --phase 1        # Phase 1 only
    python inference_phase2.py --prompt "The universe began"
    python inference_phase2.py --interactive    # interactive REPL

Outputs:
    - Side-by-side Phase 1 vs Phase 2 generation for each prompt
    - Perplexity comparison on held-out test sentences
    - Visual curriculum improvement summary
"""

import argparse
import math
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from src.model import DecoderOnlyTransformer

PROJ_ROOT = os.path.dirname(__file__)
CKPT_ROOT = os.path.join(PROJ_ROOT, "checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# Prompts that demonstrate curriculum progress
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS_PHASE1_DOMAIN = [
    "Once upon a time, there was a little girl named Lily",
    "The puppy was very happy when",
    "Tom and his friend went to the park and",
]

PROMPTS_PHASE2_DOMAIN = [
    "The history of the Roman Empire began",
    "Photosynthesis is the biological process by which",
    "According to recent scientific research,",
    "The French Revolution was a period of",
    "Quantum mechanics describes the behavior of",
]

TEST_SENTENCES_PERPLEXITY = [
    # Wikipedia-style sentences (Phase 2 domain)
    "The mitochondria are organelles found in eukaryotic cells that generate most of the cell's supply of adenosine triphosphate.",
    "The Renaissance was a period of European cultural, artistic, political and scientific rebirth that followed the Middle Ages.",
    "Isaac Newton formulated the laws of motion and universal gravitation that formed the dominant scientific viewpoint for centuries.",
    # Story-style sentences (Phase 1 domain — catastrophic forgetting check)
    "Once upon a time there was a little bunny who lived in a forest.",
    "The children played happily in the garden while their mother watched from the window.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(phase: int, device: torch.device) -> tuple:
    """Load model + tokenizer for a given phase."""
    ckpt_dir = os.path.join(CKPT_ROOT, f"phase{phase}")
    ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
    tok_path  = os.path.join(ckpt_dir, "tokenizer")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"\n  Phase {phase} checkpoint not found: {ckpt_path}\n"
            f"  → Run training first:  python train_phase{phase}.py\n"
        )

    print(f"  Loading Phase {phase} model from {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    # Load tokenizer
    if os.path.isdir(tok_path) and os.listdir(tok_path):
        tok = AutoTokenizer.from_pretrained(tok_path)
    else:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = DecoderOnlyTransformer(
        vocab_size = tok.vocab_size,
        d_model    = cfg["D_MODEL"],
        n_heads    = cfg["N_HEADS"],
        n_layers   = cfg["N_LAYERS"],
        d_ff       = cfg["D_FF"],
        max_len    = cfg["MAX_SEQ_LEN"] + 1,
        dropout    = 0.0,          # no dropout at inference
        pad_id     = tok.pad_token_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_ppl = ckpt.get("val_ppl", math.exp(ckpt["val_loss"]))
    domain  = "TinyStories" if phase == 1 else "Wikipedia"
    print(f"    val PPL: {val_ppl:.2f}  ({domain})")
    return model, tok, val_ppl


# ─────────────────────────────────────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate(model, tokenizer, prompt: str, device,
             max_new: int = 200, temperature: float = 0.8,
             top_k: int = 50, top_p: float = 0.95,
             rep_penalty: float = 1.3) -> str:
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
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity on a sentence
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sentence_perplexity(model, tokenizer, text: str, device) -> float:
    ids = tokenizer.encode(text, return_tensors="pt").to(device)
    if ids.shape[1] < 2:
        return float("inf")
    x, y  = ids[:, :-1], ids[:, 1:]
    _, loss = model(x, y)
    return math.exp(loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Main comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(device, max_new: int = 150):
    print("\n" + "═" * 70)
    print("  CURRICULUM LEARNING — Phase 1 vs Phase 2 Comparison")
    print("═" * 70)

    models = {}
    ppls   = {}

    for phase in [1, 2]:
        ckpt_dir = os.path.join(CKPT_ROOT, f"phase{phase}")
        if not os.path.exists(os.path.join(ckpt_dir, "best_model.pt")):
            print(f"  ⚠  Phase {phase} model not found — skipping.")
            continue
        m, tok, ppl = load_model(phase, device)
        models[phase] = (m, tok)
        ppls[phase]   = ppl

    if not models:
        print("  No models found. Run training first.")
        return

    # ── Perplexity comparison table ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  PERPLEXITY COMPARISON (lower = better)")
    print(f"{'─' * 70}")
    header = f"  {'Sentence':<50}"
    for ph in sorted(models):
        header += f"  Ph{ph} PPL"
    print(header)
    print("  " + "─" * 68)

    for sentence in TEST_SENTENCES_PERPLEXITY:
        row = f"  {sentence[:48]:<50}"
        for ph in sorted(models):
            m, tok = models[ph]
            ppl = sentence_perplexity(m, tok, sentence, device)
            row += f"  {ppl:8.2f}"
        print(row)

    print(f"\n  Overall val PPL (on training domain):")
    for ph in sorted(ppls):
        domain = "TinyStories" if ph == 1 else "Wikipedia"
        print(f"    Phase {ph} ({domain}): {ppls[ph]:.2f}")

    # ── Generation comparison ─────────────────────────────────────────────────
    all_prompts = [
        ("Phase 1 domain", PROMPTS_PHASE1_DOMAIN),
        ("Phase 2 domain", PROMPTS_PHASE2_DOMAIN),
    ]

    for domain_name, prompts in all_prompts:
        print(f"\n{'═' * 70}")
        print(f"  GENERATIONS — {domain_name.upper()}")
        print(f"{'═' * 70}")

        for prompt in prompts:
            print(f"\n  Prompt: \"{prompt}\"")
            print(f"  {'─' * 66}")
            for ph in sorted(models):
                m, tok = models[ph]
                text = generate(m, tok, prompt, device, max_new=max_new)
                label = f"Phase {ph}"
                # Wrap long lines
                words   = text.split()
                lines   = []
                line    = ""
                for w in words:
                    if len(line) + len(w) + 1 > 62:
                        lines.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                if line:
                    lines.append(line)
                print(f"  [{label}] {lines[0] if lines else ''}")
                for l in lines[1:]:
                    print(f"  {'':9}{l}")
                print()

    print(f"{'═' * 70}")
    print("  CURRICULUM SUMMARY")
    print(f"{'═' * 70}")
    print("  Phase 1 → TinyStories: learns basic grammar, simple vocabulary,")
    print("            narrative structure (PPL on TinyStories val: "
          f"{ppls.get(1, '?'):.2f})")
    print("  Phase 2 → Wikipedia: extends to formal prose, richer vocabulary,")
    print("            encyclopedic knowledge (PPL on Wikipedia val: "
          f"{ppls.get(2, '?'):.2f})")
    print("  The model demonstrates progressive linguistic capability improvement.")
    print(f"{'═' * 70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def interactive(phase: int, device):
    model, tok, ppl = load_model(phase, device)
    print(f"\n  Phase {phase} model loaded (val PPL {ppl:.2f})")
    print("  Type a prompt and press Enter.  Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        text = generate(model, tok, prompt, device)
        print(f"\n  {text}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2 Inference: compare Phase 1 vs Phase 2 model outputs"
    )
    parser.add_argument("--phase",       type=int, default=0,
                        help="Phase to load (1 or 2). Default 0 = compare both.")
    parser.add_argument("--prompt",      default=None,
                        help="Single custom prompt to generate from.")
    parser.add_argument("--interactive", action="store_true",
                        help="Start an interactive prompt REPL.")
    parser.add_argument("--max_new",     type=int, default=150,
                        help="Max new tokens to generate (default: 150).")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")

    if args.interactive:
        phase = args.phase if args.phase in (1, 2) else 2
        interactive(phase, DEVICE)
    elif args.prompt:
        phase = args.phase if args.phase in (1, 2) else 2
        model, tok, ppl = load_model(phase, DEVICE)
        text = generate(model, tok, args.prompt, DEVICE, max_new=args.max_new)
        print(f"\n  Phase {phase} output:\n  {text}\n")
    else:
        run_comparison(DEVICE, max_new=args.max_new)

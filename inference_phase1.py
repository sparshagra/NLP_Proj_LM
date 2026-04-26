"""
Phase 1 Inference Script — 124M Decoder-Only Transformer
══════════════════════════════════════════════════════════

Usage:
    python inference_phase1.py                                # example prompts
    python inference_phase1.py --prompt "Once upon a time"   # single prompt
    python inference_phase1.py --interactive                  # REPL mode
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.model         import DecoderOnlyTransformer, count_parameters
from configs.phase1    import cfg

try:
    from transformers import AutoTokenizer
except ImportError:
    raise ImportError("Run: pip install transformers")


# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(device):
    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_dir = os.path.join(cfg.OUTPUT_DIR, "tokenizer")
    if os.path.isdir(tok_dir) and os.listdir(tok_dir):
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded  (vocab={tokenizer.vocab_size:,})")

    # ── Model ─────────────────────────────────────────────────────────────────
    PAD_ID = tokenizer.pad_token_id
    model  = DecoderOnlyTransformer(
        vocab_size = tokenizer.vocab_size,
        d_model    = cfg.D_MODEL,
        n_heads    = cfg.N_HEADS,
        n_layers   = cfg.N_LAYERS,
        d_ff       = cfg.D_FF,
        max_len    = cfg.MAX_SEQ_LEN + 1,
        dropout    = 0.0,          # no dropout at inference
        pad_id     = PAD_ID,
    ).to(device)

    ckpt_path = os.path.join(cfg.OUTPUT_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"\n  Checkpoint not found: {ckpt_path}\n"
            "  → Train first: python train_phase1.py\n"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_params = count_parameters(model)
    print(f"✓ Model loaded from {ckpt_path}")
    print(f"  Parameters  : {n_params / 1e6:.1f} M")
    print(f"  Trained for : {ckpt['epoch']} epoch(s)")
    print(f"  Best val PPL: {ckpt.get('val_ppl', '?'):.2f}")
    print(f"  Device      : {device}")
    return model, tokenizer


def generate(model, tokenizer, prompt: str, device,
             max_new: int, temperature: float, top_k: int,
             top_p: float = 0.95, rep_penalty: float = 1.3) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new            = max_new,
            temperature        = temperature,
            top_k              = top_k,
            top_p              = top_p,
            repetition_penalty = rep_penalty,
            eos_id             = tokenizer.eos_token_id,
            max_seq_len        = cfg.MAX_SEQ_LEN,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


EXAMPLE_PROMPTS = [
    "Once upon a time",
    "The little rabbit",
    "One day in the forest",
    "Sarah was very excited",
    "The old treasure map was",
]


# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference for Phase 1 — 124M decoder-only Transformer"
    )
    parser.add_argument("--prompt",      type=str,   default=None)
    parser.add_argument("--max_new",     type=int,   default=cfg.GEN_MAX_NEW)
    parser.add_argument("--temperature", type=float, default=cfg.GEN_TEMP)
    parser.add_argument("--top_k",       type=int,   default=cfg.GEN_TOP_K)
    parser.add_argument("--top_p",       type=float, default=cfg.GEN_TOP_P)
    parser.add_argument("--rep_penalty", type=float, default=cfg.GEN_REP_PENALTY)
    parser.add_argument("--device",      type=str,   default=None)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("\n" + "="*70)
    print("  Phase 1 — Inference  |  124M Decoder-Only Transformer")
    print("="*70 + "\n")

    model, tokenizer = load_model_and_tokenizer(device)

    gen_kwargs = dict(
        max_new     = args.max_new,
        temperature = args.temperature,
        top_k       = args.top_k,
        top_p       = args.top_p,
        rep_penalty = args.rep_penalty,
    )

    # ── Single prompt mode ────────────────────────────────────────────────────
    if args.prompt:
        print(f"\n📝 Prompt: {args.prompt}\n")
        print(generate(model, tokenizer, args.prompt, device, **gen_kwargs))
        print()
        return

    # ── Interactive REPL mode ─────────────────────────────────────────────────
    if args.interactive:
        print("Interactive mode — type a prompt and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input("Prompt> ").strip()
                if not prompt:
                    continue
                print()
                print(generate(model, tokenizer, prompt, device, **gen_kwargs))
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        return

    # ── Default: run example prompts ─────────────────────────────────────────
    print("🎯 Generating from example prompts:\n")
    for i, prompt in enumerate(EXAMPLE_PROMPTS, 1):
        print(f"{i}. Prompt: {prompt}\n")
        text = generate(model, tokenizer, prompt, device, **gen_kwargs)
        print(text)
        print("\n" + "─"*70 + "\n")


if __name__ == "__main__":
    main()

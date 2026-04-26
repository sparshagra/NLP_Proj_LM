#!/usr/bin/env python3
"""
run_inference.py
=============================================================================
DEFINITIVE INFERENCE SCRIPT — uses Phase 3 best model (PPL=30.12)

Usage:
    python3 run_inference.py
    python3 run_inference.py --interactive
"""

import argparse, os, sys, torch, time
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import DecoderOnlyTransformer
from configs.phase3 import cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Load best available model ────────────────────────────────────────────────
def load_best_model():
    ckpt_path = "checkpoints/phase3/best_model.pt"
    ckpt_name = "Phase3-BestModel"
    
    tok_dirs = [
        "checkpoints/phase3/tokenizer",
        "checkpoints/phase1/tokenizer",
    ]

    if not os.path.exists(ckpt_path):
        print("ERROR: No checkpoint found."); sys.exit(1)

    tok = None
    for d in tok_dirs:
        if os.path.isdir(d) and os.listdir(d):
            tok = AutoTokenizer.from_pretrained(d); break
    if tok is None:
        tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading {ckpt_name} from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    val_ppl = ckpt.get("val_ppl", "?")
    print(f"  val_ppl = {val_ppl:.2f}" if isinstance(val_ppl, float) else f"  val_ppl = {val_ppl}")

    model = DecoderOnlyTransformer(
        vocab_size=tok.vocab_size, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
        n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF, max_len=cfg.MAX_SEQ_LEN+1,
        dropout=0.0, pad_id=tok.pad_token_id,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok, ckpt_name


def generate(model, tok, prompt, max_new=250, temperature=0.75, top_k=50, top_p=0.92, rep=1.3):
    ids = tok.encode(prompt, return_tensors="pt",
                     truncation=True, max_length=cfg.MAX_SEQ_LEN - max_new).to(DEVICE)
    pl = ids.shape[1]
    with torch.no_grad():
        out = model.generate(ids, max_new=max_new, min_new=20,
            temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=rep, eos_id=tok.eos_token_id,
            max_seq_len=cfg.MAX_SEQ_LEN)
    new = out[0, pl:].tolist()
    eos = tok.eos_token_id
    if eos in new:
        new = new[:new.index(eos)]
    resp = tok.decode(new, skip_special_tokens=False).strip()
    return resp


DEMOS = [
    "Machine learning is a subfield of artificial intelligence that",
    "The French city of Paris is famous for",
    "Python is a programming language known for",
    "The ocean covers about 71% of Earth's surface, and",
    "Exercise has many health benefits, including",
]


def run_demos(model, tok, model_name):
    print(f"\n{'='*65}")
    print(f"  TEXT COMPLETION DEMO  |  {model_name}")
    print(f"  Model: 124M decoder-only Transformer  |  Device: {DEVICE}")
    print(f"{'='*65}\n")
    for prompt in DEMOS:
        t0 = time.time()
        resp = generate(model, tok, prompt)
        elapsed = time.time() - t0
        print(f"  PROMPT      : {prompt}")
        print(f"  COMPLETION  : {resp[:400]}")
        print(f"  ({elapsed:.1f}s {'─'*50}")
        print()


def interactive(model, tok):
    print(f"\n{'='*65}")
    print("  INTERACTIVE MODE  (type 'quit' to exit)")
    print(f"{'='*65}\n")
    while True:
        try:
            prompt = input("Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q", ""):
                break
            print("\nGenerating...\n")
            resp = generate(model, tok, prompt)
            print(f"Completion:\n  {resp}\n")
            print("-" * 60)
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Inference for 124M curriculum LM")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--max_new",     type=int,   default=250)
    args = parser.parse_args()

    print(f"\n  Device : {DEVICE}")
    model, tok, name = load_best_model()
    run_demos(model, tok, name)
    if args.interactive:
        interactive(model, tok)


if __name__ == "__main__":
    main()

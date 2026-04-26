"""
diagnose.py  — one-shot diagnostic for the instruction-tuned model.
Run: python diagnose.py
"""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(__file__))

CKPT = "checkpoints/instruction_tuned/best_model.pt"
TINY = "checkpoints/instruction_tuned/inference_model.pt"   # weights-only, fast

print("="*60)
print("STEP 1: Check checkpoint file")
print(f"  File exists : {os.path.exists(CKPT)}")
print(f"  File size   : {os.path.getsize(CKPT)/1e6:.0f} MB")
print(f"  Tiny exists : {os.path.exists(TINY)}")

# ── If the tiny (weights-only) file doesn't exist, create it ────────────────
if not os.path.exists(TINY):
    print("\nSTEP 2: Creating fast inference checkpoint (one-time, ~2 min)...")
    t0 = time.time()
    full = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s  |  val_ppl={full['val_ppl']:.4f}")
    slim = {
        "model_state": full["model_state"],
        "val_ppl"    : full["val_ppl"],
        "val_loss"   : full["val_loss"],
    }
    torch.save(slim, TINY)
    print(f"  Saved: {TINY}  ({os.path.getsize(TINY)/1e6:.0f} MB)")
else:
    print("  Tiny checkpoint already exists — skipping creation.")

# ── Load the tiny checkpoint (fast) ─────────────────────────────────────────
print("\nSTEP 3: Loading weights-only checkpoint...")
t0 = time.time()
slim = torch.load(TINY, map_location="cpu", weights_only=False)
print(f"  Loaded in {time.time()-t0:.1f}s  |  val_ppl={slim['val_ppl']:.4f}")

# ── Build model ─────────────────────────────────────────────────────────────
print("\nSTEP 4: Building model...")
from src.model import DecoderOnlyTransformer
from transformers import AutoTokenizer
import configs.phase_it as cfg_mod
cfg = cfg_mod.cfg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

tok = AutoTokenizer.from_pretrained("checkpoints/phase1/tokenizer")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS_ID = tok.eos_token_id
print(f"  EOS token id: {EOS_ID}  text: {repr(tok.eos_token)}")

model = DecoderOnlyTransformer(
    vocab_size=tok.vocab_size, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
    n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF, max_len=cfg.MAX_SEQ_LEN + 1,
    dropout=0.0, pad_id=EOS_ID,
).to(device)
model.load_state_dict(slim["model_state"])
model.eval()
print("  Model loaded ✓")

# ── Logit diagnostic BEFORE generation ──────────────────────────────────────
print("\nSTEP 5: Top-10 next tokens after '### Response:\\n'")
SYSTEM = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.")
prompt = f"{SYSTEM}\n\n### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
ids = tok.encode(prompt, return_tensors="pt").to(device)
print(f"  Prompt length: {ids.shape[1]} tokens")

with torch.no_grad():
    logits, _ = model(ids)
    last = logits[0, -1, :].float()
    topk = torch.topk(last, 15)

print("  Rank | Token ID | Token Text          | Logit")
print("  " + "-"*55)
for rank, (val, idx) in enumerate(zip(topk.values, topk.indices)):
    tok_text = repr(tok.decode([idx.item()]))
    eos_flag = " ← EOS" if idx.item() == EOS_ID else ""
    print(f"  #{rank+1:2d}  | {idx.item():8d} | {tok_text:20s} | {val.item():.3f}{eos_flag}")

# ── Raw generation ───────────────────────────────────────────────────────────
print("\nSTEP 6: Generate 30 tokens (argmax greedy, no EOS stop)")
with torch.no_grad():
    x = ids.clone()
    generated = []
    for i in range(30):
        logits, _ = model(x)
        next_tok = logits[0, -1, :].argmax(-1).item()
        generated.append(next_tok)
        x = torch.cat([x, torch.tensor([[next_tok]], device=device)], dim=1)

print("  Raw token IDs:", generated)
print("  Decoded tokens:")
for i, t in enumerate(generated):
    print(f"    [{i:2d}] id={t:6d}  text={repr(tok.decode([t]))}")

full_text = tok.decode(generated, skip_special_tokens=False)
print(f"\n  Full generation: {repr(full_text)}")
print(f"  After strip()  : {repr(full_text.strip())}")

print("\n" + "="*60)
print("DIAGNOSTIC COMPLETE")

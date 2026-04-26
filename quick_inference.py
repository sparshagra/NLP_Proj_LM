"""
quick_inference.py  — fast inference using the slim checkpoint
Root cause fix: token 198 (newline) has logit 11 at position 0.
Fix: suppress newline + EOS tokens for first 10 positions.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

from src.model import DecoderOnlyTransformer
from transformers import AutoTokenizer
import configs.phase_it as cfg_mod
import torch.nn.functional as F

cfg    = cfg_mod.cfg
TINY   = "checkpoints/instruction_tuned/inference_model.pt"
SYSTEM = ("Below is an instruction that describes a task. "
          "Write a response that appropriately completes the request.")

# ---------- load -------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok    = AutoTokenizer.from_pretrained("checkpoints/phase1/tokenizer")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS_ID = tok.eos_token_id

slim  = torch.load(TINY, map_location=device, weights_only=False)
model = DecoderOnlyTransformer(
    vocab_size=tok.vocab_size, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
    n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF, max_len=cfg.MAX_SEQ_LEN + 1,
    dropout=0.0, pad_id=EOS_ID,
).to(device)
model.load_state_dict(slim["model_state"])
model.eval()
print(f"  Model loaded  (val_ppl={slim['val_ppl']:.2f}  device={device})\n")

# ---------- generate ---------------------------------------------------
# SUPPRESS_IDS: tokens to block for first N steps
# Token 198 = '\n'  (the newline collapse token — logit 11 vs 2.4 for others)
# Token 628 = '\n\n'
# Token 220 = ' ' (leading space)
# EOS_ID    = 50256 (don't stop immediately)
SUPPRESS_IDS  = {198, 628, 220, EOS_ID}
SUPPRESS_STEPS = 10   # how many positions to block them

def generate(instruction, input_text="", max_new=200, temperature=0.85,
             top_k=50, top_p=0.92, rep_penalty=1.3):
    if input_text.strip():
        prompt = (f"{SYSTEM}\n\n### Instruction:\n{instruction.strip()}\n\n"
                  f"### Input:\n{input_text.strip()}\n\n### Response:\n")
    else:
        prompt = f"{SYSTEM}\n\n### Instruction:\n{instruction.strip()}\n\n### Response:\n"

    ids = tok.encode(prompt, return_tensors="pt").to(device)
    prompt_len = ids.shape[1]
    generated  = []

    with torch.no_grad():
        x = ids.clone()
        for step in range(max_new):
            logits, _ = model(x[:, -cfg.MAX_SEQ_LEN:])
            logits     = logits[0, -1, :].float()

            # Repetition penalty
            if rep_penalty != 1.0:
                for tid in set(x[0].tolist()):
                    logits[tid] = (logits[tid] / rep_penalty
                                   if logits[tid] > 0
                                   else logits[tid] * rep_penalty)

            # Suppress newline / EOS for first SUPPRESS_STEPS tokens
            if step < SUPPRESS_STEPS:
                for tid in SUPPRESS_IDS:
                    logits[tid] = -1e9

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = -1e9

            # Top-p
            if top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                remove = cum - F.softmax(sorted_l, dim=-1) > top_p
                sorted_l[remove] = -1e9
                logits = torch.zeros_like(logits).scatter_(0, sorted_i, sorted_l)

            probs  = F.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1).item()
            generated.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

            if nxt == EOS_ID and step >= 10:
                break

    # Decode: stop at first EOS, skip leading/trailing whitespace
    if EOS_ID in generated:
        generated = generated[:generated.index(EOS_ID)]
    text = tok.decode(generated, skip_special_tokens=False).strip()
    return text if text else "(no output)"


# ---------- demo -------------------------------------------------------
demos = [
    ("What is the capital of France?", ""),
    ("Explain what machine learning is in simple terms.", ""),
    ("Write a short poem about the ocean.", ""),
    ("List 3 benefits of drinking water.", ""),
    ("Translate the following to Spanish.", "Hello, how are you today?"),
    ("Write a Python function to reverse a string.", ""),
    ("What causes rain?", ""),
]

print("="*65)
print("  INSTRUCTION-TUNED 124M MODEL  —  DEMO OUTPUTS")
print("="*65)

for instr, inp in demos:
    resp = generate(instr, inp)
    print(f"\n  Instruction: {instr}")
    if inp:
        print(f"  Input:       {inp}")
    print(f"  Response:    {resp[:400]}")
    print("  " + "-"*60)

print("\nDone.")

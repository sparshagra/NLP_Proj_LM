"""
phase3_inference.py  -  inference demo using the Phase 3 model (val_ppl=30.12)
This is our best WORKING model. Loads in ~1 second.
"""
import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from src.model import DecoderOnlyTransformer
from transformers import AutoTokenizer
import configs.phase_it as cfg_mod

cfg    = cfg_mod.cfg
SLIM   = "checkpoints/phase3/inference_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok    = AutoTokenizer.from_pretrained("checkpoints/phase1/tokenizer")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
EOS_ID = tok.eos_token_id

t0   = time.time()
slim = torch.load(SLIM, map_location=device, weights_only=False)
model = DecoderOnlyTransformer(
    vocab_size=tok.vocab_size, d_model=cfg.D_MODEL, n_heads=cfg.N_HEADS,
    n_layers=cfg.N_LAYERS, d_ff=cfg.D_FF, max_len=cfg.MAX_SEQ_LEN + 1,
    dropout=0.0, pad_id=EOS_ID,
).to(device)
model.load_state_dict(slim["model_state"])
model.eval()
print(f"[Phase 3 loaded in {time.time()-t0:.1f}s  |  val_ppl={slim['val_ppl']:.2f}  |  device={device}]")


def generate(prompt, max_new=120, temperature=0.85, top_k=50, top_p=0.92, rep=1.3):
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    gen = []
    with torch.no_grad():
        x = ids.clone()
        for step in range(max_new):
            logits, _ = model(x[:, -cfg.MAX_SEQ_LEN:])
            logits = logits[0, -1, :].float()
            for tid in set(x[0].tolist()):
                logits[tid] = logits[tid] / rep if logits[tid] > 0 else logits[tid] * rep
            logits /= max(temperature, 1e-8)
            if top_k > 0:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[-1]
                logits[logits < kth] = -1e9
            if top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                sl[cp - F.softmax(sl, dim=-1) > top_p] = -1e9
                logits = torch.zeros_like(logits).scatter_(0, si, sl)
            probs = F.softmax(logits, dim=-1)
            nxt   = torch.multinomial(probs, 1).item()
            gen.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt == EOS_ID:
                break
    if EOS_ID in gen:
        gen = gen[:gen.index(EOS_ID)]
    return tok.decode(gen, skip_special_tokens=False).strip()


print("\n" + "="*65)
print("  PHASE 3 MODEL  —  124M Decoder-Only Transformer")
print("  Curriculum: TinyStories → Wikipedia → Mixed Corpus")
print("  val PPL = 30.12  (Mixed Web+Reasoning corpus)")
print("="*65)

demos = [
    # Knowledge / QA
    ("The capital of France is",),
    ("Machine learning is a subfield of artificial intelligence that",),
    ("The water cycle describes how water",),
    # Reasoning
    ("To solve the equation 2x + 4 = 10, we",),
    ("The main difference between supervised and unsupervised learning is",),
    # Code / Technical
    ("In Python, a list comprehension allows you to",),
    ("A binary search algorithm works by",),
    # Creative
    ("Once upon a time, in a kingdom by the sea, there lived a",),
    ("The ocean stretched endlessly before her, and she",),
]

for (prompt,) in demos:
    out = generate(prompt)
    print(f"\n  Prompt : {prompt}")
    print(f"  Output : {out[:350]}")
    print("  " + "-"*60)

print("\n" + "="*65)
print("  Done.")

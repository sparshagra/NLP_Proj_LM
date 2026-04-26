"""
make_slim_phase3.py  -  one-time: extract weights from Phase 3 checkpoint
Saves a fast-loading inference-only copy (~480MB instead of 1.4GB).
Run ONCE, then delete this script.
"""
import torch, os, time

SRC = "checkpoints/phase3/best_model.pt"
DST = "checkpoints/phase3/inference_model.pt"

if os.path.exists(DST):
    print(f"Already exists: {DST}  ({os.path.getsize(DST)/1e6:.0f} MB)")
else:
    print(f"Loading {SRC}  ({os.path.getsize(SRC)/1e6:.0f} MB) ...")
    print("This takes ~5-8 min (includes optimizer state). One-time only.")
    t0  = time.time()
    ck  = torch.load(SRC, map_location="cpu", weights_only=False)
    print(f"  Loaded in {time.time()-t0:.1f}s  |  val_ppl={ck.get('val_ppl', '?')}")
    slim = {
        "model_state": ck["model_state"],
        "val_ppl"    : ck.get("val_ppl", 0),
        "val_loss"   : ck.get("val_loss", 0),
        "phase"      : ck.get("phase", 3),
    }
    torch.save(slim, DST)
    print(f"Saved: {DST}  ({os.path.getsize(DST)/1e6:.0f} MB)")

print("Done.")

"""Compare v1 bilinear vs v2 sharp-attention on all-mod + 4 LOMO held-outs."""
import json
from pathlib import Path

ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")


def get(p):
    if not p.exists():
        return None
    return json.load(open(p))


print("=" * 80)
print("All-mod (full training, all 12 modifications seen)")
print(f"{'Model':30s}  {'AUCb':>6s}  {'AUCm':>6s}  {'MCC':>6s}")
print("-" * 60)
for name, sub in [
    ("v1 bilinear", "chemical_v1_bilinear"),
    ("v2 sharp-attn tau=0.25", "chemical_v2_tau0.25_linear_morgan_r2"),
]:
    s = get(ROOT / sub / "test_summary.json")
    if s:
        print(f"  {name:28s}  {s['AUCb']:.3f}  {s['AUCm']:.3f}  {s['MCC']:.3f}")

print()
print("=" * 80)
print("LOMO held-out (tau=0.25 vs v1 baseline)")
print(f"{'Held-out':10s}  {'model':28s}  {'AUCb':>6s}  {'AUCm':>6s}  {'verdict':<10s}")
print("-" * 70)
for mod in ["m6A", "m7G", "Am", "Psi"]:
    s_v1 = get(ROOT / "chemical_v1_bilinear_lomo" / mod / "test_heldout_summary.json")
    s_v2 = get(ROOT / "chemical_v2_tau0.25_linear_morgan_r2_lomo" / mod / "test_heldout_summary.json")
    for label, s in [("v1 bilinear", s_v1), ("v2 sharp-attn", s_v2)]:
        if s:
            v = "PASS" if s["AUCm"] > 0.5 else "FAIL"
            delta = ""
            if label == "v2 sharp-attn" and s_v1:
                delta = f"  Δ={s['AUCm']-s_v1['AUCm']:+.3f}"
            print(f"  {mod:8s}  {label:28s}  {s['AUCb']:.3f}  {s['AUCm']:.3f}  {v:<5s} {delta}")

print()
print("=" * 80)
print("tau sweep on m6A LOMO (does sharper attention help?)")
print(f"{'tau':>5s}  {'AUCb':>6s}  {'AUCm':>6s}  {'verdict':<6s}")
print("-" * 35)
# Include v1 baseline for context
s_v1 = get(ROOT / "chemical_v1_bilinear_lomo/m6A/test_heldout_summary.json")
if s_v1:
    print(f"  (v1)  {s_v1['AUCb']:.3f}  {s_v1['AUCm']:.3f}  baseline")
for tau in ["0.25", "0.5", "1.0"]:
    s = get(ROOT / f"chemical_v2_tau{tau}_linear_morgan_r2_lomo/m6A/test_heldout_summary.json")
    if s:
        v = "PASS" if s["AUCm"] > 0.5 else "FAIL"
        print(f"  {tau:>3s}   {s['AUCb']:.3f}  {s['AUCm']:.3f}  {v}")

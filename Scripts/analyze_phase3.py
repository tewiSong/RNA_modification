"""Aggregate Phase 3 multi-seed results into a mean ± std table."""
import json
from pathlib import Path

import numpy as np

ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")
HELDOUTS = ["m7G", "m6A", "Am", "Psi"]
SEEDS = [1, 2, 3]
MODEL_DIRS = {
    "original":         "original_lomo",
    "chemical_v0":      "chemical_lomo",
    "modid":            "modid_lomo",
    "v1_bilinear":      "chemical_v1_bilinear_lomo",
    "v1_lowrank":       "chemical_v1_lowrank_lomo",
    "v1_hypernetwork":  "chemical_v1_hypernetwork_lomo",
}


def load_metric(name, heldout, seed, metric):
    base = MODEL_DIRS[name]
    if seed == 1:
        subdir = base
    else:
        subdir = f"{base}_seed{seed}"
    path = ROOT / subdir / heldout / "test_heldout_summary.json"
    if not path.exists():
        return None
    return float(json.load(open(path))[metric])


def main():
    print(f"{'Model':18s} {'Held-out':9s} {'AUCb mean±std':>20s} {'AUCb seeds':>30s}")
    print("-" * 88)
    for model_name in MODEL_DIRS:
        for heldout in HELDOUTS:
            vals = [load_metric(model_name, heldout, s, "AUCb") for s in SEEDS]
            vals_present = [v for v in vals if v is not None]
            if not vals_present:
                continue
            arr = np.array(vals_present)
            seeds_str = "[" + ", ".join(f"{v:.3f}" if v is not None else "  -  " for v in vals) + "]"
            print(f"{model_name:18s} {heldout:9s} {arr.mean():.3f} ± {arr.std(ddof=1):.3f}    {seeds_str:>30s}")
    print()
    print("=== Paired comparisons across (mod × seed), AUCb ===")
    pairs = [
        ("v1_lowrank", "chemical_v0"),
        ("v1_hypernetwork", "chemical_v0"),
        ("v1_bilinear", "chemical_v0"),
        ("v1_lowrank", "v1_bilinear"),
        ("chemical_v0", "modid"),
        ("chemical_v0", "original"),
    ]
    from scipy import stats
    for a_name, b_name in pairs:
        diffs = []
        for heldout in HELDOUTS:
            for s in SEEDS:
                a = load_metric(a_name, heldout, s, "AUCb")
                b = load_metric(b_name, heldout, s, "AUCb")
                if a is not None and b is not None:
                    diffs.append(a - b)
        d = np.array(diffs)
        t, p = stats.ttest_rel([], []) if len(d) < 2 else stats.ttest_1samp(d, 0)
        try:
            w_p = stats.wilcoxon(d).pvalue
        except ValueError:
            w_p = float("nan")
        print(f"  {a_name:18s} - {b_name:18s}  n={len(d):2d}  mean_diff={d.mean():+.4f}  t-p={p:.3f}  Wilcoxon-p={w_p:.3f}")


if __name__ == "__main__":
    main()

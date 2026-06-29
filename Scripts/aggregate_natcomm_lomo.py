import csv
import json
from pathlib import Path

import numpy as np


MODS = ["m6A", "m1A", "m5C", "m5U", "m6Am", "Am", "Cm", "Gm", "Um", "I", "m7G", "Psi"]
METRICS = ["Sn", "Sp", "Acc", "MCC", "AUCb", "AUCm"]


ROOT = Path("Results/paper_aligned")
OUT = Path("Results/natcomm_multiseed_summary")


METHOD_DIRS = {
    "baseline": {
        1: "chemical_v3_tau0.4_linear_morgan_r2_bio0_lomo",
        2: "chemical_v3_tau0.4_linear_morgan_r2_bio0_seed2_lomo",
        3: "chemical_v3_tau0.4_linear_morgan_r2_bio0_seed3_lomo",
    },
    "unrestricted": {
        1: "chemical_v2_softlabel_tau0.4_linear_g0.2_lomo",
        2: "chemical_v2_softlabel_tau0.4_linear_seed2_g0.2_lomo",
        3: "chemical_v2_softlabel_tau0.4_linear_seed3_g0.2_lomo",
    },
    "tanimoto_filtered": {
        1: "chemical_v2_softlabel_filtered_tau0.4_linear_g0.2_joint_prob_aw1.0_tmin0.45_lomo",
        2: "chemical_v2_softlabel_filtered_tau0.4_linear_seed2_g0.2_joint_prob_aw1.0_tmin0.45_lomo",
        3: "chemical_v2_softlabel_filtered_tau0.4_linear_seed3_g0.2_joint_prob_aw1.0_tmin0.45_lomo",
    },
    "proposed": {
        1: "chemical_v2_softlabel_filtered_tau0.4_linear_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo",
        2: "chemical_v2_softlabel_filtered_tau0.4_linear_seed2_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo",
        3: "chemical_v2_softlabel_filtered_tau0.4_linear_seed3_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo",
    },
}


def read_summary(method, seed, mod):
    path = ROOT / METHOD_DIRS[method][seed] / mod / "test_heldout_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        return json.load(handle)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in METHOD_DIRS:
        for seed in (1, 2, 3):
            for mod in MODS:
                data = read_summary(method, seed, mod)
                row = {"method": method, "seed": seed, "heldout": mod}
                row.update({metric: float(data[metric]) for metric in METRICS})
                rows.append(row)

    with (OUT / "per_method_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "seed", "heldout"] + METRICS)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in METHOD_DIRS:
        method_rows = [r for r in rows if r["method"] == method]
        for metric in METRICS:
            seed_means = []
            for seed in (1, 2, 3):
                vals = [r[metric] for r in method_rows if r["seed"] == seed]
                seed_means.append(float(np.mean(vals)))
            summary_rows.append({
                "method": method,
                "metric": metric,
                "mean_over_seed_means": float(np.mean(seed_means)),
                "std_over_seed_means": float(np.std(seed_means, ddof=1)),
                "seed1_mean": seed_means[0],
                "seed2_mean": seed_means[1],
                "seed3_mean": seed_means[2],
            })

    with (OUT / "method_seed_mean_summary.csv").open("w", newline="") as handle:
        fieldnames = ["method", "metric", "mean_over_seed_means", "std_over_seed_means", "seed1_mean", "seed2_mean", "seed3_mean"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    paired_rows = []
    for other in ("baseline", "unrestricted", "tanimoto_filtered"):
        for metric in ("AUCm", "AUCb"):
            diffs = []
            for seed in (1, 2, 3):
                for mod in MODS:
                    proposed = read_summary("proposed", seed, mod)[metric]
                    comparator = read_summary(other, seed, mod)[metric]
                    diffs.append(float(proposed) - float(comparator))
            paired_rows.append({
                "comparison": f"proposed-minus-{other}",
                "metric": metric,
                "n_pairs": len(diffs),
                "mean_delta": float(np.mean(diffs)),
                "std_delta": float(np.std(diffs, ddof=1)),
                "min_delta": float(np.min(diffs)),
                "max_delta": float(np.max(diffs)),
            })

    with (OUT / "paired_auc_delta.csv").open("w", newline="") as handle:
        fieldnames = ["comparison", "metric", "n_pairs", "mean_delta", "std_delta", "min_delta", "max_delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(paired_rows)

    md = []
    md.append("# NatComm multi-seed LOMO summary\n")
    md.append("## Mean over twelve held-out modifications, then mean ± sd over three seeds\n")
    for method in METHOD_DIRS:
        aucm = next(r for r in summary_rows if r["method"] == method and r["metric"] == "AUCm")
        aucb = next(r for r in summary_rows if r["method"] == method and r["metric"] == "AUCb")
        md.append(f"- {method}: AUCm {aucm['mean_over_seed_means']:.4f} ± {aucm['std_over_seed_means']:.4f}; "
                  f"AUCb {aucb['mean_over_seed_means']:.4f} ± {aucb['std_over_seed_means']:.4f}\n")
    md.append("\n## Paired deltas\n")
    for row in paired_rows:
        md.append(f"- {row['comparison']} ({row['metric']}): mean delta {row['mean_delta']:.4f} ± {row['std_delta']:.4f}, n={row['n_pairs']}\n")
    (OUT / "summary.md").write_text("".join(md), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()


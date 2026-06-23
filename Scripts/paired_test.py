import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from v0_data import MODIFICATION_NAMES


def block_aucb(prob, label, mod_index):
    positive_indices = np.flatnonzero(label[:, mod_index] == 1)
    start = int(positive_indices[0])
    end = start + positive_indices.shape[0] * 2
    return float(roc_auc_score(label[start:end, mod_index], prob[start:end, mod_index]))


def aucm(prob, label, mod_index):
    return float(roc_auc_score(label[:, mod_index], prob[:, mod_index]))


def load_aucb_matrix(method_dirs):
    """Return dict[(method, seed) -> (12,)] of per-modification AUCb."""
    out = {}
    for method, seed_dirs in method_dirs.items():
        for seed, dir_path in seed_dirs.items():
            path = Path(dir_path) / "test_predictions.npz"
            data = np.load(path)
            prob, label = data["prob"], data["label"]
            row = np.array([block_aucb(prob, label, k) for k in range(label.shape[1])])
            out[(method, seed)] = row
    return out


def paired_compare(matrix, method_a, method_b, seeds, mod_indices=None):
    if mod_indices is None:
        mod_indices = list(range(len(MODIFICATION_NAMES)))
    a_vals = []
    b_vals = []
    for seed in seeds:
        a_vals.append(matrix[(method_a, seed)][mod_indices])
        b_vals.append(matrix[(method_b, seed)][mod_indices])
    a = np.stack(a_vals).ravel()
    b = np.stack(b_vals).ravel()
    diff = a - b
    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    return {
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": len(diff),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "t_stat": float(t_stat),
        "t_p": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods_json", required=True,
        help='JSON file mapping method -> {seed: dir_path}. Example: {"chem_v0": {"1": "Results/.../chemical_seed1", ...}, ...}',
    )
    parser.add_argument("--pairs", nargs="+", required=True, help="Pairs like A:B C:D")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    with open(args.methods_json) as h:
        method_dirs = {k: {str(s): v for s, v in seeds.items()} for k, seeds in json.load(h).items()}

    seeds = sorted({s for d in method_dirs.values() for s in d})
    for method, sd in method_dirs.items():
        assert set(sd) == set(seeds), f"method {method} missing seeds; got {sorted(sd)}"

    matrix = load_aucb_matrix(method_dirs)
    rows = []
    for pair in args.pairs:
        a, b = pair.split(":")
        rows.append(paired_compare(matrix, a, b, seeds))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

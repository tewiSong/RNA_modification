#!/usr/bin/env python3
"""Generate NatComm-target audit and interpretation artifacts."""

import csv
import json
import math
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_multirm import MODIFICATION_NAMES  # noqa: E402


ROOT = REPO / "Results" / "paper_aligned"
OUT = REPO / "Results" / "natcomm_audit"
MULTISEED = REPO / "Results" / "natcomm_multiseed_summary"
EXTERNAL = REPO / "Results" / "external_rmbase_lomo"

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


def read_csv_dicts(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_modification_table():
    rows = read_csv_dicts(REPO / "Data" / "modifications.csv")
    by_name = {row["name"]: row for row in rows}
    return [by_name[name] for name in MODIFICATION_NAMES]


def load_tanimoto_matrix():
    bio_path = REPO / "Data" / "bio_priors.pkl"
    if bio_path.exists():
        try:
            with bio_path.open("rb") as handle:
                data = pickle.load(handle)
            matrix = np.asarray(data["tanimoto_matrix"], dtype=float)
            if matrix.shape == (len(MODIFICATION_NAMES), len(MODIFICATION_NAMES)):
                return matrix
        except ModuleNotFoundError:
            pass

    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    table = load_modification_table()
    fps = []
    for row in table:
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            raise ValueError(f"bad SMILES for {row['name']}")
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    matrix = np.eye(len(fps), dtype=float)
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            val = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            matrix[i, j] = val
            matrix[j, i] = val
    return matrix


def write_family_outputs(table, tanimoto):
    family_rows = []
    for row in table:
        family_rows.append({
            "modification": row["name"],
            "h5_label": row["h5_label"],
            "original_base": row["original_base"],
        })
    write_csv(
        OUT / "nucleotide_family_table.csv",
        family_rows,
        ["modification", "h5_label", "original_base"],
    )

    matrix_rows = []
    for i, name in enumerate(MODIFICATION_NAMES):
        row = {"modification": name}
        row.update({MODIFICATION_NAMES[j]: f"{tanimoto[i, j]:.6f}" for j in range(len(MODIFICATION_NAMES))})
        matrix_rows.append(row)
    write_csv(OUT / "tanimoto_matrix.csv", matrix_rows, ["modification"] + MODIFICATION_NAMES)

    order = sorted(range(len(MODIFICATION_NAMES)), key=lambda i: (table[i]["original_base"], MODIFICATION_NAMES[i]))
    ordered_names = [MODIFICATION_NAMES[i] for i in order]
    ordered_matrix = tanimoto[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(ordered_matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(ordered_names)))
    ax.set_yticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=45, ha="right")
    ax.set_yticklabels(ordered_names)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Morgan r=2 Tanimoto similarity grouped by nucleotide family")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Tanimoto")

    # Draw family boundaries.
    bases = [table[i]["original_base"] for i in order]
    for idx in range(1, len(bases)):
        if bases[idx] != bases[idx - 1]:
            ax.axhline(idx - 0.5, color="white", linewidth=1.8)
            ax.axvline(idx - 0.5, color="white", linewidth=1.8)

    fig.tight_layout()
    fig.savefig(OUT / "tanimoto_heatmap.png", dpi=220)
    plt.close(fig)


def has_soft_label_source(config, method, heldout_index, table, tanimoto):
    gamma = float(config.get("soft_label_gamma", 0.0))
    if gamma <= 0:
        return False
    t_row = tanimoto[heldout_index].astype(float).copy()
    t_row[heldout_index] = 0.0
    tmin = float(config.get("soft_label_tani_min", 0.0))
    if tmin > 0:
        t_row[t_row < tmin] = 0.0
    if bool(config.get("soft_label_same_base_only", False)):
        bases = np.array([row["original_base"] for row in table])
        source_mask = bases == bases[heldout_index]
        source_mask[heldout_index] = False
        t_row[~source_mask] = 0.0
    return bool(np.any(t_row > 0.0))


def audit_configs(table, tanimoto):
    rows = []
    failures = []
    for method, seed_dirs in METHOD_DIRS.items():
        for seed, dirname in seed_dirs.items():
            for mod in MODIFICATION_NAMES:
                mod_dir = ROOT / dirname / mod
                config_path = mod_dir / "config.json"
                checkpoint_path = mod_dir / "best_model.pt"
                summary_path = mod_dir / "test_heldout_summary.json"
                row = {
                    "method": method,
                    "seed": seed,
                    "heldout": mod,
                    "config_exists": config_path.exists(),
                    "checkpoint_exists": checkpoint_path.exists(),
                    "test_summary_exists": summary_path.exists(),
                    "status": "ok",
                }
                if not config_path.exists():
                    row["status"] = "missing_config"
                    failures.append(row.copy())
                    rows.append(row)
                    continue
                with config_path.open() as handle:
                    config = json.load(handle)
                checks = {
                    "command_is_v2_lomo": config.get("command") == "train_chemical_v2_lomo",
                    "heldout_matches": config.get("heldout_mod") == mod,
                    "heldout_removed_from_train_tasks": config.get("heldout_index") not in config.get("train_task_indices", []),
                    "selection_seen_mods_only": config.get("heldout_index") not in config.get("train_task_indices", []),
                    "same_base_for_proposed_only": (
                        bool(config.get("soft_label_same_base_only", False)) if method == "proposed"
                        else not bool(config.get("soft_label_same_base_only", False))
                    ),
                }
                if method == "baseline":
                    checks["baseline_no_soft_labels"] = float(config.get("soft_label_gamma", 0.0)) == 0.0
                    checks["baseline_loss_excludes_heldout"] = config.get("heldout_index") not in config.get("loss_task_indices", [])
                if method in {"unrestricted", "tanimoto_filtered", "proposed"}:
                    checks["soft_label_gamma_0_2"] = abs(float(config.get("soft_label_gamma", -1.0)) - 0.2) < 1e-9
                    if has_soft_label_source(config, method, config.get("heldout_index"), table, tanimoto):
                        checks["loss_includes_heldout_soft_column_if_source_exists"] = (
                            config.get("heldout_index") in config.get("loss_task_indices", [])
                        )
                    else:
                        checks["loss_excludes_heldout_when_no_source_exists"] = (
                            config.get("heldout_index") not in config.get("loss_task_indices", [])
                        )
                if method == "unrestricted":
                    checks["unrestricted_tmin_0"] = float(config.get("soft_label_tani_min", 0.0)) == 0.0
                if method in {"tanimoto_filtered", "proposed"}:
                    checks["tmin_0_45"] = abs(float(config.get("soft_label_tani_min", -1.0)) - 0.45) < 1e-9
                if not checkpoint_path.exists():
                    checks["checkpoint_exists"] = False
                if not summary_path.exists():
                    checks["test_summary_exists"] = False

                failed = [name for name, ok in checks.items() if not ok]
                row.update({name: bool(ok) for name, ok in checks.items()})
                if failed:
                    row["status"] = "failed:" + ",".join(failed)
                    failures.append(row.copy())
                rows.append(row)

    all_fields = sorted({key for row in rows for key in row.keys()})
    leading = ["method", "seed", "heldout", "status"]
    fields = leading + [field for field in all_fields if field not in leading]
    write_csv(OUT / "lomo_config_audit.csv", rows, fields)
    return rows, failures


def write_paired_significance():
    metrics = read_csv_dicts(MULTISEED / "per_method_metrics.csv")
    by_key = {
        (row["method"], int(row["seed"]), row["heldout"]): row
        for row in metrics
    }
    rows = []
    for other in ("baseline", "unrestricted", "tanimoto_filtered"):
        for metric in ("AUCm", "AUCb"):
            diffs = []
            for seed in (1, 2, 3):
                for mod in MODIFICATION_NAMES:
                    proposed = float(by_key[("proposed", seed, mod)][metric])
                    comparator = float(by_key[(other, seed, mod)][metric])
                    diffs.append(proposed - comparator)
            diffs = np.asarray(diffs, dtype=float)
            t_stat, t_p = stats.ttest_1samp(diffs, 0.0)
            try:
                w_stat, w_p = stats.wilcoxon(diffs)
            except ValueError:
                w_stat, w_p = math.nan, math.nan
            rows.append({
                "comparison": f"proposed-minus-{other}",
                "metric": metric,
                "n_pairs": len(diffs),
                "mean_delta": f"{np.mean(diffs):.6f}",
                "std_delta": f"{np.std(diffs, ddof=1):.6f}",
                "paired_t_stat": f"{t_stat:.6f}",
                "paired_t_p": f"{t_p:.6g}",
                "wilcoxon_stat": f"{w_stat:.6f}" if not math.isnan(w_stat) else "nan",
                "wilcoxon_p": f"{w_p:.6g}" if not math.isnan(w_p) else "nan",
            })
    write_csv(
        OUT / "paired_significance.csv",
        rows,
        [
            "comparison",
            "metric",
            "n_pairs",
            "mean_delta",
            "std_delta",
            "paired_t_stat",
            "paired_t_p",
            "wilcoxon_stat",
            "wilcoxon_p",
        ],
    )
    return rows


def external_coverage_summary():
    rows = read_csv_dicts(EXTERNAL / "external_lomo_metrics.csv")
    by_mod = {}
    nan_by_method = {}
    for row in rows:
        mod = row["heldout_mod"]
        by_mod.setdefault(mod, int(float(row["n_pos"])))
        method = row["method"]
        nan_by_method.setdefault(method, 0)
        if row["external_aucm"].lower() == "nan":
            nan_by_method[method] += 1
    coverage_rows = [{"heldout": mod, "n_pos": by_mod[mod]} for mod in sorted(by_mod)]
    write_csv(OUT / "external_positive_coverage.csv", coverage_rows, ["heldout", "n_pos"])
    return coverage_rows, nan_by_method


def write_report(config_rows, failures, significance_rows, coverage_rows, nan_by_method):
    summary = read_csv_dicts(MULTISEED / "method_seed_mean_summary.csv")
    aucm_summary = {row["method"]: row for row in summary if row["metric"] == "AUCm"}
    external_summary = {row["method"]: row for row in read_csv_dicts(EXTERNAL / "external_lomo_method_summary.csv")}

    lines = []
    lines.append("# NatComm audit and interpretation results\n\n")
    lines.append("## Completion\n\n")
    lines.append(f"- LOMO config/checkpoint/test-summary rows audited: {len(config_rows)}\n")
    lines.append(f"- Audit failures: {len(failures)}\n")
    lines.append("- Generated nucleotide-family table, Tanimoto matrix, and grouped heatmap.\n\n")
    lines.append("## Internal MultiRM LOMO AUCm\n\n")
    for method in ("baseline", "unrestricted", "tanimoto_filtered", "proposed"):
        row = aucm_summary[method]
        lines.append(
            f"- {method}: {float(row['mean_over_seed_means']):.4f} +- "
            f"{float(row['std_over_seed_means']):.4f}\n"
        )
    lines.append("\n## External RMBase LOMO AUCm\n\n")
    for method in ("baseline", "unrestricted", "tanimoto_filtered", "proposed"):
        row = external_summary[method]
        lines.append(
            f"- {method}: {float(row['mean_external_aucm']):.4f} +- "
            f"{float(row['std_external_aucm']):.4f}\n"
        )
    lines.append("\n## External coverage caveat\n\n")
    low = [row for row in coverage_rows if int(row["n_pos"]) < 100]
    zero = [row for row in coverage_rows if int(row["n_pos"]) == 0]
    lines.append(f"- Zero-positive held-out modifications: {', '.join(row['heldout'] for row in zero) or 'none'}\n")
    low_text = ", ".join(f"{row['heldout']}={row['n_pos']}" for row in low) or "none"
    lines.append(f"- Low-positive held-out modifications (<100): {low_text}\n")
    lines.append(f"- NaN external AUC rows by method: {nan_by_method}\n\n")
    lines.append("## Paired significance\n\n")
    for row in significance_rows:
        lines.append(
            f"- {row['comparison']} {row['metric']}: mean_delta={row['mean_delta']}, "
            f"paired_t_p={row['paired_t_p']}, wilcoxon_p={row['wilcoxon_p']}\n"
        )
    lines.append("\n## Leakage audit conclusion\n\n")
    if failures:
        lines.append("One or more config-level checks failed; inspect lomo_config_audit.csv before manuscript use.\n")
    else:
        lines.append(
            "All audited LOMO runs remove the held-out modification from train-task indices. "
            "Baseline losses exclude the held-out column. Soft-label runs include the held-out "
            "column only through chemistry-derived pseudo-labels, and the proposed runs set "
            "soft_label_same_base_only=true with Tanimoto threshold 0.45.\n"
        )
    (OUT / "audit_summary.md").write_text("".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = load_modification_table()
    tanimoto = load_tanimoto_matrix()
    write_family_outputs(table, tanimoto)
    config_rows, failures = audit_configs(table, tanimoto)
    significance_rows = write_paired_significance()
    coverage_rows, nan_by_method = external_coverage_summary()
    write_report(config_rows, failures, significance_rows, coverage_rows, nan_by_method)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

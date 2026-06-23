import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

from v0_data import MODIFICATION_NAMES


N_BOOT = 1000
CI_LO = 2.5
CI_HI = 97.5


def block_indices_for_mod(labels, mod_index):
    positive_indices = np.flatnonzero(labels[:, mod_index] == 1)
    assert positive_indices.shape[0] > 0
    start = int(positive_indices[0])
    block = positive_indices.shape[0] * 2
    end = start + block
    assert end <= labels.shape[0]
    return start, end, positive_indices.shape[0]


def delong_auc_ci(labels, scores, alpha=0.95):
    # Closed-form Hanley-McNeil approximation of DeLong variance.
    # For balanced data n_pos == n_neg this is equivalent to the standard
    # DeLong derivation with much less code; sufficient when n=100.
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    n_pos = pos.shape[0]
    n_neg = neg.shape[0]
    auc = roc_auc_score(labels.astype(int), scores)
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1.0) * (q1 - auc * auc)
        + (n_neg - 1.0) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 1e-12)))
    z = stats.norm.ppf(0.5 + alpha / 2.0)
    return float(auc), float(auc - z * se), float(auc + z * se), se


def select_gmean_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gmeans = np.sqrt(tpr * (1.0 - fpr))
    return float(thresholds[int(np.argmax(gmeans))])


def threshold_metrics(labels, scores):
    threshold = select_gmean_threshold(labels, scores)
    predictions = (scores >= threshold).astype(bool)
    labels_bool = labels.astype(bool)
    tp = int((predictions & labels_bool).sum())
    tn = int(((~predictions) & (~labels_bool)).sum())
    fp = int((predictions & (~labels_bool)).sum())
    fn = int(((~predictions) & labels_bool).sum())
    n = labels.shape[0]
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / n
    try:
        mcc = matthews_corrcoef(labels, predictions.astype(np.float32))
    except ValueError:
        mcc = 0.0
    return sn, sp, acc, mcc


def stratified_bootstrap_block(block_labels, block_scores, rng, n_boot):
    pos_mask = block_labels == 1
    pos_scores = block_scores[pos_mask]
    neg_scores = block_scores[~pos_mask]
    pos_labels = block_labels[pos_mask]
    neg_labels = block_labels[~pos_mask]
    n_pos = pos_scores.shape[0]
    n_neg = neg_scores.shape[0]
    sn_list, sp_list, acc_list, mcc_list, auc_list = [], [], [], [], []
    for _ in range(n_boot):
        pi = rng.integers(0, n_pos, size=n_pos)
        ni = rng.integers(0, n_neg, size=n_neg)
        sb = np.concatenate([pos_scores[pi], neg_scores[ni]])
        lb = np.concatenate([pos_labels[pi], neg_labels[ni]])
        sn, sp, acc, mcc = threshold_metrics(lb, sb)
        sn_list.append(sn); sp_list.append(sp); acc_list.append(acc); mcc_list.append(mcc)
        if lb.sum() and (1 - lb).sum():
            auc_list.append(roc_auc_score(lb, sb))
    return {
        "sn": (np.percentile(sn_list, CI_LO), np.percentile(sn_list, CI_HI)),
        "sp": (np.percentile(sp_list, CI_LO), np.percentile(sp_list, CI_HI)),
        "acc": (np.percentile(acc_list, CI_LO), np.percentile(acc_list, CI_HI)),
        "mcc": (np.percentile(mcc_list, CI_LO), np.percentile(mcc_list, CI_HI)),
        "aucb_boot": (np.percentile(auc_list, CI_LO), np.percentile(auc_list, CI_HI)) if auc_list else (np.nan, np.nan),
    }


def per_mod_ci(prob, label, mod_index, rng):
    start, end, n_pos = block_indices_for_mod(label, mod_index)
    block_label = label[start:end, mod_index]
    block_score = prob[start:end, mod_index]
    aucb_point, aucb_lo, aucb_hi, aucb_se = delong_auc_ci(block_label, block_score)

    aucm_label = label[:, mod_index]
    aucm_score = prob[:, mod_index]
    aucm_point, aucm_lo, aucm_hi, aucm_se = delong_auc_ci(aucm_label, aucm_score)

    sn_pt, sp_pt, acc_pt, mcc_pt = threshold_metrics(block_label, block_score)
    boot = stratified_bootstrap_block(block_label, block_score, rng, N_BOOT)

    return {
        "Modification": MODIFICATION_NAMES[mod_index],
        "Sn": sn_pt, "Sn_lo": boot["sn"][0], "Sn_hi": boot["sn"][1],
        "Sp": sp_pt, "Sp_lo": boot["sp"][0], "Sp_hi": boot["sp"][1],
        "Acc": acc_pt, "Acc_lo": boot["acc"][0], "Acc_hi": boot["acc"][1],
        "MCC": mcc_pt, "MCC_lo": boot["mcc"][0], "MCC_hi": boot["mcc"][1],
        "AUCb": aucb_point, "AUCb_lo": aucb_lo, "AUCb_hi": aucb_hi, "AUCb_se": aucb_se,
        "AUCm": aucm_point, "AUCm_lo": aucm_lo, "AUCm_hi": aucm_hi, "AUCm_se": aucm_se,
    }


def mean_aucb_bootstrap(prob, label, mod_indices, rng, n_boot):
    blocks = [block_indices_for_mod(label, k) for k in mod_indices]
    means = []
    for _ in range(n_boot):
        per_mod = []
        for (start, end, _), k in zip(blocks, mod_indices):
            bl = label[start:end, k]
            bs = prob[start:end, k]
            pos_mask = bl == 1
            n_pos = int(pos_mask.sum())
            n_neg = int((~pos_mask).sum())
            pi = rng.integers(0, n_pos, size=n_pos)
            ni = rng.integers(0, n_neg, size=n_neg)
            lb = np.concatenate([bl[pos_mask][pi], bl[~pos_mask][ni]])
            sb = np.concatenate([bs[pos_mask][pi], bs[~pos_mask][ni]])
            per_mod.append(roc_auc_score(lb, sb))
        means.append(float(np.mean(per_mod)))
    return float(np.mean(means)), float(np.percentile(means, CI_LO)), float(np.percentile(means, CI_HI))


def write_csv(rows, out_path):
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, float):
                    formatted[key] = f"{value:.4f}"
                else:
                    formatted[key] = value
            writer.writerow(formatted)


def report_for(predictions_path, mod_indices, seed):
    rng = np.random.default_rng(seed)
    data = np.load(predictions_path)
    prob = data["prob"]
    label = data["label"]
    rows = [per_mod_ci(prob, label, k, rng) for k in mod_indices]

    mean_point, mean_lo, mean_hi = mean_aucb_bootstrap(prob, label, mod_indices, rng, N_BOOT)
    summary_row = {
        "Modification": "Mean(AUCb across mods)",
        "Sn": "", "Sn_lo": "", "Sn_hi": "",
        "Sp": "", "Sp_lo": "", "Sp_hi": "",
        "Acc": "", "Acc_lo": "", "Acc_hi": "",
        "MCC": "", "MCC_lo": "", "MCC_hi": "",
        "AUCb": mean_point, "AUCb_lo": mean_lo, "AUCb_hi": mean_hi, "AUCb_se": "",
        "AUCm": "", "AUCm_lo": "", "AUCm_hi": "", "AUCm_se": "",
    }
    rows.append(summary_row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--mod_indices", default="all", help="comma-separated indices or 'all' or single index for LOMO")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mod_indices == "all":
        mod_indices = list(range(len(MODIFICATION_NAMES)))
    else:
        mod_indices = [int(x) for x in args.mod_indices.split(",")]

    rows = report_for(args.predictions_path, mod_indices, args.seed)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_path)
    summary = {"mean_aucb_ci": [rows[-1]["AUCb_lo"], rows[-1]["AUCb_hi"]], "rows": len(rows) - 1}
    with out_path.with_suffix(".summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

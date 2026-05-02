import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from v0_data import (
    H5_LABELS,
    MODIFICATION_NAMES,
    ORIGINAL_BASES,
    MultiRMSplitDataset,
    build_chemical_feature_matrix,
    inspect_h5,
    load_modification_table,
    read_multirm_split,
)
from v0_models import ChemicalConditionedRM, ModificationIdConditionedRM, SequenceOnlyRM


def parse_args():
    parser = argparse.ArgumentParser(description="Chemical-conditioned MultiRM v0")
    parser.add_argument("--data_path", default="Data/MultiRM_data.h5")
    parser.add_argument("--modifications_path", default="Data/modifications.csv")
    parser.add_argument("--save_dir", default="Results/v0")
    parser.add_argument("--model_type", choices=["chemical", "mod_id", "sequence_only"], default="chemical")
    parser.add_argument("--loss_mode", choices=["all_bce", "compatible_masked_bce"], default="all_bce")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--chem_hidden_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--eval_test", action="store_true")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--selection_metric", choices=["auto", "mean_aucb", "mean_aucm"], default="auto")
    parser.add_argument("--valid_aucb_mode", choices=["same_base_unmodified", "multirm_block"], default="same_base_unmodified")
    parser.add_argument("--test_aucb_mode", choices=["same_base_unmodified", "multirm_block"], default="multirm_block")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(args, chemical_features):
    if args.model_type == "chemical":
        return ChemicalConditionedRM(
            chemical_features=chemical_features,
            hidden_size=args.hidden_size,
            chem_hidden_size=args.chem_hidden_size,
            dropout=args.dropout,
        )
    if args.model_type == "mod_id":
        return ModificationIdConditionedRM(hidden_size=args.hidden_size, dropout=args.dropout)
    if args.model_type == "sequence_only":
        return SequenceOnlyRM(hidden_size=args.hidden_size, dropout=args.dropout)
    raise ValueError(args.model_type)


def compute_loss(logits, labels, compatibility_mask, loss_mode):
    if loss_mode == "all_bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    positive_outside_mask = ((labels > 0.5) & (compatibility_mask < 0.5)).sum()
    assert int(positive_outside_mask.item()) == 0
    unreduced = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    denominator = compatibility_mask.sum()
    assert float(denominator.item()) > 0.0
    return (unreduced * compatibility_mask).sum() / denominator


def train_one_epoch(model, loader, optimizer, args):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for x, labels, compatibility_mask in loader:
        x = x.to(args.device)
        labels = labels.to(args.device)
        compatibility_mask = compatibility_mask.to(args.device)

        logits = model(x)
        loss = compute_loss(logits, labels, compatibility_mask, args.loss_mode)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.shape[0]
        total_examples += x.shape[0]
    return total_loss / total_examples


def collect_predictions(model, loader, device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for x, labels, _ in loader:
            logits = model(x.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return probabilities, labels


def resolve_selection_metric(args):
    if args.selection_metric != "auto":
        return args.selection_metric
    if args.loss_mode == "compatible_masked_bce":
        return "mean_aucb"
    return "mean_aucm"


def canonical_eval_base(base):
    if base == "U":
        return "T"
    return base


def select_gmean_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    gmeans = np.sqrt(tpr * (1.0 - fpr))
    best_index = int(np.argmax(gmeans))
    return float(thresholds[best_index])


def calculate_binary_metrics(labels, scores):
    threshold = select_gmean_threshold(labels, scores)
    predictions = scores >= threshold
    labels_bool = labels.astype(bool)
    true_positive = int((predictions & labels_bool).sum())
    true_negative = int(((~predictions) & (~labels_bool)).sum())
    false_positive = int((predictions & (~labels_bool)).sum())
    false_negative = int(((~predictions) & labels_bool).sum())

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    accuracy = (true_positive + true_negative) / labels.shape[0]
    mcc = matthews_corrcoef(labels, predictions.astype(np.float32))
    return {
        "threshold": threshold,
        "sn": float(sensitivity),
        "sp": float(specificity),
        "acc": float(accuracy),
        "mcc": float(mcc),
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
    }


def evaluate_multirm_auc(probabilities, labels, center_bases, aucb_mode):
    rows = []
    unmodified = labels.sum(axis=1) == 0
    center_bases = np.array(center_bases)

    for index, name in enumerate(MODIFICATION_NAMES):
        positive_mask = labels[:, index] == 1
        if aucb_mode == "same_base_unmodified":
            original_base = canonical_eval_base(ORIGINAL_BASES[index])
            aucb_negative_mask = unmodified & (center_bases == original_base)
            aucb_mask = positive_mask | aucb_negative_mask
        else:
            block_size = labels.shape[0] // len(MODIFICATION_NAMES)
            assert labels.shape[0] == block_size * len(MODIFICATION_NAMES)
            block_start = index * block_size
            block_end = (index + 1) * block_size
            aucb_mask = np.zeros(labels.shape[0], dtype=bool)
            aucb_mask[block_start:block_end] = True
            aucb_negative_mask = aucb_mask & (~positive_mask)

        aucb_labels = labels[aucb_mask, index]
        assert int(aucb_labels.sum()) > 0
        assert int((aucb_labels == 0).sum()) > 0
        aucb_scores = probabilities[aucb_mask, index]
        aucb = roc_auc_score(aucb_labels, aucb_scores)
        binary_metrics = calculate_binary_metrics(aucb_labels, aucb_scores)

        aucm_labels = labels[:, index]
        aucm_scores = probabilities[:, index]
        aucm = roc_auc_score(aucm_labels, aucm_scores)

        rows.append({
            "index": index,
            "name": name,
            "h5_label": H5_LABELS[index],
            "original_base": ORIGINAL_BASES[index],
            "sn": binary_metrics["sn"],
            "sp": binary_metrics["sp"],
            "acc": binary_metrics["acc"],
            "mcc": binary_metrics["mcc"],
            "aucb": float(aucb),
            "aucm": float(aucm),
            "threshold": binary_metrics["threshold"],
            "tp": binary_metrics["tp"],
            "tn": binary_metrics["tn"],
            "fp": binary_metrics["fp"],
            "fn": binary_metrics["fn"],
            "aucb_mode": aucb_mode,
            "aucb_samples": int(aucb_mask.sum()),
            "aucb_positives": int(positive_mask.sum()),
            "aucb_negatives": int(aucb_negative_mask.sum()),
        })

    aucb_values = np.array([row["aucb"] for row in rows], dtype=np.float64)
    aucm_values = np.array([row["aucm"] for row in rows], dtype=np.float64)
    sn_values = np.array([row["sn"] for row in rows], dtype=np.float64)
    sp_values = np.array([row["sp"] for row in rows], dtype=np.float64)
    acc_values = np.array([row["acc"] for row in rows], dtype=np.float64)
    mcc_values = np.array([row["mcc"] for row in rows], dtype=np.float64)
    summary = {
        "mean_sn": float(sn_values.mean()),
        "median_sn": float(np.median(sn_values)),
        "mean_sp": float(sp_values.mean()),
        "median_sp": float(np.median(sp_values)),
        "mean_acc": float(acc_values.mean()),
        "median_acc": float(np.median(acc_values)),
        "mean_mcc": float(mcc_values.mean()),
        "median_mcc": float(np.median(mcc_values)),
        "mean_aucb": float(aucb_values.mean()),
        "median_aucb": float(np.median(aucb_values)),
        "mean_aucm": float(aucm_values.mean()),
        "median_aucm": float(np.median(aucm_values)),
        "aucb_mode": aucb_mode,
    }
    return rows, summary


def evaluate_loss(model, loader, args):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for x, labels, compatibility_mask in loader:
            x = x.to(args.device)
            labels = labels.to(args.device)
            compatibility_mask = compatibility_mask.to(args.device)
            logits = model(x)
            loss = compute_loss(logits, labels, compatibility_mask, args.loss_mode)
            total_loss += loss.item() * x.shape[0]
            total_examples += x.shape[0]
    return total_loss / total_examples


TABLE4_COLUMNS = [
    ("Modification", "name"),
    ("Sn", "sn"),
    ("Sp", "sp"),
    ("Acc", "acc"),
    ("MCC", "mcc"),
    ("AUCb", "aucb"),
    ("AUCm", "aucm"),
]


def build_table4_rows(rows, summary):
    table_rows = []
    for row in rows:
        table_rows.append({
            "Modification": row["name"],
            "Sn": row["sn"],
            "Sp": row["sp"],
            "Acc": row["acc"],
            "MCC": row["mcc"],
            "AUCb": row["aucb"],
            "AUCm": row["aucm"],
        })

    table_rows.append({
        "Modification": "Mean",
        "Sn": summary["mean_sn"],
        "Sp": summary["mean_sp"],
        "Acc": summary["mean_acc"],
        "MCC": summary["mean_mcc"],
        "AUCb": summary["mean_aucb"],
        "AUCm": summary["mean_aucm"],
    })
    table_rows.append({
        "Modification": "Median",
        "Sn": summary["median_sn"],
        "Sp": summary["median_sp"],
        "Acc": summary["median_acc"],
        "MCC": summary["median_mcc"],
        "AUCb": summary["median_aucb"],
        "AUCm": summary["median_aucm"],
    })
    return table_rows


def format_table4(split_name, rows, summary):
    table_rows = build_table4_rows(rows, summary)
    headers = [column_name for column_name, _ in TABLE4_COLUMNS]
    formatted_rows = []
    for row in table_rows:
        formatted_rows.append({
            "Modification": row["Modification"],
            "Sn": f"{row['Sn']:.4f}",
            "Sp": f"{row['Sp']:.4f}",
            "Acc": f"{row['Acc']:.4f}",
            "MCC": f"{row['MCC']:.4f}",
            "AUCb": f"{row['AUCb']:.4f}",
            "AUCm": f"{row['AUCm']:.4f}",
        })

    widths = {
        header: max(len(header), max(len(row[header]) for row in formatted_rows))
        for header in headers
    }
    lines = [
        f"Table 4 style metrics split={split_name} aucb_mode={summary['aucb_mode']}",
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in formatted_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def save_table4(save_dir, split_name, rows, summary):
    table_rows = build_table4_rows(rows, summary)
    csv_path = save_dir / f"{split_name}_table4.csv"
    txt_path = save_dir / f"{split_name}_table4.txt"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[column_name for column_name, _ in TABLE4_COLUMNS])
        writer.writeheader()
        writer.writerows(table_rows)
    with txt_path.open("w") as handle:
        handle.write(format_table4(split_name, rows, summary))
        handle.write("\n")


def print_table4(split_name, rows, summary):
    print(format_table4(split_name, rows, summary), flush=True)


def save_metrics(save_dir, split_name, rows, summary):
    csv_path = save_dir / f"{split_name}_metrics.csv"
    json_path = save_dir / f"{split_name}_summary.json"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    save_table4(save_dir, split_name, rows, summary)


def cache_path(cache_dir, split_name, max_samples):
    sample_tag = "full" if max_samples is None else str(max_samples)
    return Path(cache_dir) / f"{split_name}_{sample_tag}_51nt.npz"


def load_or_create_split(data_path, split_name, max_samples, cache_dir):
    if cache_dir is None:
        return read_multirm_split(data_path, split_name, max_samples)

    path = cache_path(cache_dir, split_name, max_samples)
    if path.exists():
        loaded = np.load(path)
        return {
            "x": loaded["x"],
            "y": loaded["y"],
            "compatibility_mask": loaded["compatibility_mask"],
            "center_bases": loaded["center_bases"],
        }

    split_data = read_multirm_split(data_path, split_name, max_samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=split_data["x"],
        y=split_data["y"],
        compatibility_mask=split_data["compatibility_mask"],
        center_bases=split_data["center_bases"],
    )
    return split_data


def main():
    args = parse_args()
    set_seed(args.seed)
    selection_metric = resolve_selection_metric(args)
    save_dir = Path(args.save_dir) / args.model_type / args.loss_mode
    save_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["resolved_selection_metric"] = selection_metric
    with (save_dir / "config.json").open("w") as handle:
        json.dump(config, handle, indent=2)

    print("HDF5 structure")
    for name, shape, dtype in inspect_h5(args.data_path):
        print(f"{name}\t{shape}\t{dtype}", flush=True)

    modification_table = load_modification_table(args.modifications_path)
    chemical_features = build_chemical_feature_matrix(modification_table)

    train_data = load_or_create_split(args.data_path, "train", args.max_train_samples, args.cache_dir)
    valid_data = load_or_create_split(args.data_path, "valid", args.max_valid_samples, args.cache_dir)
    test_data = None
    if args.eval_test:
        test_data = load_or_create_split(args.data_path, "test", args.max_test_samples, args.cache_dir)

    pin_memory = args.device.startswith("cuda")

    train_loader = DataLoader(
        MultiRMSplitDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        MultiRMSplitDataset(valid_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = None
    if args.eval_test:
        test_loader = DataLoader(
            MultiRMSplitDataset(test_data),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    model = create_model(args, chemical_features).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = save_dir / "history.csv"
    best_selection_value = -1.0
    best_epoch = None
    best_valid_rows = None
    best_valid_summary = None
    with history_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "valid_loss",
                "mean_sn",
                "median_sn",
                "mean_sp",
                "median_sp",
                "mean_acc",
                "median_acc",
                "mean_mcc",
                "median_mcc",
                "mean_aucb",
                "median_aucb",
                "mean_aucm",
                "median_aucm",
                "aucb_mode",
                "selection_metric",
                "selection_value",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, args)
            valid_loss = evaluate_loss(model, valid_loader, args)
            valid_probabilities, valid_labels = collect_predictions(model, valid_loader, args.device)
            valid_rows, valid_summary = evaluate_multirm_auc(
                valid_probabilities,
                valid_labels,
                valid_data["center_bases"],
                args.valid_aucb_mode,
            )
            selection_value = valid_summary[selection_metric]
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                **valid_summary,
                "selection_metric": selection_metric,
                "selection_value": selection_value,
            })
            handle.flush()
            print(
                f"epoch={epoch} split=valid aucb_mode={args.valid_aucb_mode} "
                f"train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
                f"mean_sn={valid_summary['mean_sn']:.6f} mean_sp={valid_summary['mean_sp']:.6f} "
                f"mean_acc={valid_summary['mean_acc']:.6f} mean_mcc={valid_summary['mean_mcc']:.6f} "
                f"mean_aucb={valid_summary['mean_aucb']:.6f} mean_aucm={valid_summary['mean_aucm']:.6f} "
                f"selection_metric={selection_metric} selection_value={selection_value:.6f}",
                flush=True,
            )

            if selection_value > best_selection_value:
                best_selection_value = selection_value
                best_epoch = epoch
                best_valid_rows = valid_rows
                best_valid_summary = valid_summary
                torch.save(model.state_dict(), save_dir / "best_model.pt")
                save_metrics(save_dir, "valid", valid_rows, valid_summary)

    torch.save(model.state_dict(), save_dir / "last_model.pt")
    assert best_epoch is not None
    assert best_valid_rows is not None
    assert best_valid_summary is not None
    print(
        f"best_validation epoch={best_epoch} selection_metric={selection_metric} "
        f"selection_value={best_selection_value:.6f}",
        flush=True,
    )
    print_table4("valid", best_valid_rows, best_valid_summary)

    if args.eval_test:
        model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=args.device))
        test_probabilities, test_labels = collect_predictions(model, test_loader, args.device)
        test_rows, test_summary = evaluate_multirm_auc(
            test_probabilities,
            test_labels,
            test_data["center_bases"],
            args.test_aucb_mode,
        )
        save_metrics(save_dir, "test", test_rows, test_summary)
        print(f"split=test aucb_mode={args.test_aucb_mode}", flush=True)
        print_table4("test", test_rows, test_summary)
        print(json.dumps(test_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

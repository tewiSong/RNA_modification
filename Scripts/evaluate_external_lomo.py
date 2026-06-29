#!/usr/bin/env python3
"""Evaluate trained LOMO checkpoints on an external H5 benchmark."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from paper_multirm import (  # noqa: E402
    ChemicalMultiRMv2,
    MODIFICATION_NAMES,
    RmDataset,
    build_chemical_feature_matrix,
    collect_predictions,
    load_modification_table,
    read_split_as_kmers,
)


def parse_method_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("method entries must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def build_v2_model(config, device):
    modification_table = load_modification_table(config.get("modifications_path", "Data/modifications.csv"))
    chemical_features = build_chemical_feature_matrix(
        modification_table,
        site_weight=float(config.get("site_weight", 0.0)),
        fp_kind=config.get("fp_kind", "morgan_r2"),
        bio_weight=float(config.get("bio_weight", 0.0)),
    )
    model = ChemicalMultiRMv2(
        config.get("embedding_path", "Embeddings/embeddings_12RM.pkl"),
        chemical_features,
        tau=float(config.get("tau", 0.4)),
        chemical_encoder_type=config.get("chemical_encoder_type", "linear"),
    )
    return model.to(device)


def external_aucb(labels, scores, rng):
    positives = np.flatnonzero(labels > 0.5)
    negatives = np.flatnonzero(labels < 0.5)
    if len(positives) == 0 or len(negatives) == 0:
        return np.nan
    keep_neg = rng.choice(negatives, size=min(len(positives), len(negatives)), replace=False)
    keep = np.concatenate([positives, keep_neg])
    return float(roc_auc_score(labels[keep], scores[keep]))


def evaluate_checkpoint(method_name, root, config_path, external_data, args):
    mod_dir = config_path.parent
    with config_path.open() as handle:
        config = json.load(handle)
    command = config.get("command", "")
    if command != "train_chemical_v2_lomo":
        return {
            "method": method_name,
            "root": str(root),
            "heldout_mod": mod_dir.name,
            "seed": config.get("seed", ""),
            "status": f"skipped unsupported command {command}",
        }
    heldout_mod = config.get("heldout_mod", mod_dir.name)
    if heldout_mod not in MODIFICATION_NAMES:
        return {
            "method": method_name,
            "root": str(root),
            "heldout_mod": heldout_mod,
            "seed": config.get("seed", ""),
            "status": "skipped unknown heldout_mod",
        }
    checkpoint = mod_dir / args.checkpoint
    if not checkpoint.exists():
        return {
            "method": method_name,
            "root": str(root),
            "heldout_mod": heldout_mod,
            "seed": config.get("seed", ""),
            "status": f"missing checkpoint {args.checkpoint}",
        }

    model = build_v2_model(config, args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    loader = DataLoader(
        RmDataset(external_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    probabilities, labels = collect_predictions(model, loader, args.device)
    index = MODIFICATION_NAMES.index(heldout_mod)
    y = labels[:, index]
    score = probabilities[:, index]
    n_pos = int((y > 0.5).sum())
    n_neg = int((y < 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        aucm = np.nan
    else:
        aucm = float(roc_auc_score(y, score))
    rng = np.random.default_rng(args.seed)
    aucb = external_aucb(y, score, rng)
    return {
        "method": method_name,
        "root": str(root),
        "heldout_mod": heldout_mod,
        "seed": config.get("seed", ""),
        "checkpoint": str(checkpoint),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "external_aucm": aucm,
        "external_aucb": aucb,
        "status": "ok",
    }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = [
            "method",
            "root",
            "seed",
            "heldout_mod",
            "checkpoint",
            "n_pos",
            "n_neg",
            "external_aucm",
            "external_aucb",
            "status",
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path, rows):
    ok = [row for row in rows if row.get("status") == "ok"]
    grouped = {}
    for row in ok:
        grouped.setdefault(row["method"], []).append(row)
    summary_rows = []
    for method, method_rows in grouped.items():
        aucm = np.array([float(row["external_aucm"]) for row in method_rows], dtype=float)
        aucb = np.array([float(row["external_aucb"]) for row in method_rows], dtype=float)
        summary_rows.append(
            {
                "method": method,
                "n_rows": len(method_rows),
                "mean_external_aucm": float(np.nanmean(aucm)),
                "std_external_aucm": float(np.nanstd(aucm, ddof=1)) if len(aucm) > 1 else 0.0,
                "mean_external_aucb": float(np.nanmean(aucb)),
                "std_external_aucb": float(np.nanstd(aucb, ddof=1)) if len(aucb) > 1 else 0.0,
            }
        )
    write_csv(
        path,
        summary_rows,
        fieldnames=[
            "method",
            "n_rows",
            "mean_external_aucm",
            "std_external_aucm",
            "mean_external_aucb",
            "std_external_aucb",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external_h5", default="Data/external_rmbase/processed/external_rmbase_human.h5")
    parser.add_argument("--methods", nargs="+", type=parse_method_arg, required=True)
    parser.add_argument("--output_dir", default="Results/external_rmbase_lomo")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--embedding_path", default="Embeddings/embeddings_12RM.pkl")
    parser.add_argument("--cache_dir", default="Results/external_rmbase_lomo/cache")
    parser.add_argument("--length", type=int, default=51)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    external_data = read_split_as_kmers(
        args.external_h5,
        "test",
        args.length,
        args.embedding_path,
        args.cache_dir,
    )

    rows = []
    for method_name, root in args.methods:
        for config_path in sorted(root.glob("*/config.json")):
            row = evaluate_checkpoint(method_name, root, config_path, external_data, args)
            rows.append(row)
            print(row, flush=True)

    write_csv(output_dir / "external_lomo_metrics.csv", rows)
    write_summary(output_dir / "external_lomo_method_summary.csv", rows)


if __name__ == "__main__":
    main()

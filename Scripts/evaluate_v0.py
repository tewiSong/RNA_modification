import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from train_v0 import (
    collect_predictions,
    create_model,
    evaluate_multirm_auc,
    load_or_create_split,
    print_table4,
    save_metrics,
)
from v0_data import MultiRMSplitDataset, build_chemical_feature_matrix, load_modification_table


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained MultiRM v0 checkpoint")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aucb_mode", choices=["auto", "same_base_unmodified", "multirm_block"], default="auto")
    return parser.parse_args()


def resolve_aucb_mode(split_name, aucb_mode):
    if aucb_mode != "auto":
        return aucb_mode
    if split_name == "test":
        return "multirm_block"
    return "same_base_unmodified"


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    with (run_dir / "config.json").open() as handle:
        train_config = json.load(handle)
    model_args = SimpleNamespace(**train_config)
    model_args.device = args.device

    modification_table = load_modification_table(model_args.modifications_path)
    chemical_features = build_chemical_feature_matrix(modification_table)
    model = create_model(model_args, chemical_features).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

    max_samples_key = f"max_{args.split}_samples"
    split_data = load_or_create_split(
        model_args.data_path,
        args.split,
        getattr(model_args, max_samples_key),
        model_args.cache_dir,
    )
    loader = DataLoader(
        MultiRMSplitDataset(split_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    probabilities, labels = collect_predictions(model, loader, args.device)
    rows, summary = evaluate_multirm_auc(
        probabilities,
        labels,
        split_data["center_bases"],
        resolve_aucb_mode(args.split, args.aucb_mode),
    )
    save_metrics(run_dir, args.split, rows, summary)
    print_table4(args.split, rows, summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

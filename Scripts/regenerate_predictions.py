import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from paper_multirm import (
    ChemicalMultiRM,
    ChemicalMultiRMv1,
    ModificationIdMultiRM,
    PaperMultiRM,
    RmDataset,
    collect_predictions,
    read_split_as_kmers,
)
from v0_data import build_chemical_feature_matrix, load_modification_table


COMMAND_TO_MODEL = {
    "train_original": "PaperMultiRM",
    "train_original_lomo": "PaperMultiRM",
    "train_chemical": "ChemicalMultiRM",
    "train_chemical_lomo": "ChemicalMultiRM",
    "train_modid": "ModificationIdMultiRM",
    "train_modid_lomo": "ModificationIdMultiRM",
    "train_chemical_v1": "ChemicalMultiRMv1",
    "train_chemical_v1_lomo": "ChemicalMultiRMv1",
}


def build_model(config, device):
    model_name = COMMAND_TO_MODEL[config["command"]]
    if model_name == "PaperMultiRM":
        return PaperMultiRM(config["embedding_path"]).to(device)
    if model_name == "ModificationIdMultiRM":
        return ModificationIdMultiRM(config["embedding_path"]).to(device)
    modification_table = load_modification_table(config["modifications_path"])
    chemical_features = build_chemical_feature_matrix(modification_table)
    if model_name == "ChemicalMultiRM":
        return ChemicalMultiRM(config["embedding_path"], chemical_features).to(device)
    if model_name == "ChemicalMultiRMv1":
        return ChemicalMultiRMv1(
            config["embedding_path"], chemical_features,
            num_heads=config["num_heads"], scorer_type=config["scorer_type"],
        ).to(device)
    raise ValueError(model_name)


def regenerate_for(experiment_dir, device):
    experiment_dir = Path(experiment_dir)
    config_path = experiment_dir / "config.json"
    weights_path = experiment_dir / "best_model.pt"
    if not (config_path.exists() and weights_path.exists()):
        print(f"SKIP {experiment_dir} (missing config or weights)", flush=True)
        return
    with config_path.open() as handle:
        config = json.load(handle)

    model = build_model(config, device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    test_data = read_split_as_kmers(
        config["data_path"], "test", config["length"], config["embedding_path"], config["cache_dir"],
    )
    valid_data = read_split_as_kmers(
        config["data_path"], "valid", config["length"], config["embedding_path"], config["cache_dir"],
    )
    test_loader = DataLoader(RmDataset(test_data), batch_size=config["batch_size"], shuffle=False)
    valid_loader = DataLoader(RmDataset(valid_data), batch_size=config["batch_size"], shuffle=False)

    test_prob, test_label = collect_predictions(model, test_loader, device)
    np.savez(experiment_dir / "test_predictions.npz", prob=test_prob, label=test_label)
    valid_prob, valid_label = collect_predictions(model, valid_loader, device)
    np.savez(experiment_dir / "valid_predictions.npz", prob=valid_prob, label=valid_label)
    print(f"OK   {experiment_dir} test={test_prob.shape} valid={valid_prob.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="Results/paper_aligned")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(args.root)
    flat = [
        root / "original_from_scratch",
        root / "chemical",
        root / "modid",
        root / "chemical_v1_bilinear",
        root / "chemical_v1_lowrank",
        root / "chemical_v1_hypernetwork",
    ]
    lomo_parents = [
        root / "original_lomo",
        root / "chemical_lomo",
        root / "modid_lomo",
        root / "chemical_v1_bilinear_lomo",
    ]
    for path in flat:
        regenerate_for(path, args.device)
    for parent in lomo_parents:
        if not parent.exists():
            continue
        for heldout_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
            regenerate_for(heldout_dir, args.device)


if __name__ == "__main__":
    main()

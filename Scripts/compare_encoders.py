"""Compare MLP-encoder vs Linear-encoder LOMO results."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table

ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")


def cos_pair(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_encoder_outputs(save_dir, chem_features, encoder_type):
    config = json.load(open(Path(save_dir) / "config.json"))
    model = ChemicalMultiRMv1(
        "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
        chem_features,
        num_heads=config["num_heads"],
        scorer_type=config["scorer_type"],
        chemical_encoder_type=encoder_type,
    )
    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        out = model.chemical_encoder(model.chemical_features).numpy()
    return out


def main():
    mod_table = load_modification_table("/ibex/user/songt/MultiRM/Data/modifications.csv")
    chem_features = build_chemical_feature_matrix(mod_table)

    name_to_idx = {n: i for i, n in enumerate(MODIFICATION_NAMES)}
    neighbors = {"m6A": "m6Am", "m7G": "Gm", "Am": "m6Am", "Psi": "Um"}

    print("=== Comparison: MLP vs Linear encoder, LOMO models ===\n")
    print(f"{'Held-out':10s} {'Encoder':10s} {'cos(held,neighbor)':>20s} {'AUCb':>8s} {'AUCm':>8s} {'pos_prob_mean':>14s}")
    print("-" * 76)

    for heldout in ["m6A", "m7G", "Am", "Psi"]:
        neighbor = neighbors[heldout]
        heldout_idx = name_to_idx[heldout]
        neighbor_idx = name_to_idx[neighbor]

        for encoder_type, suffix in [("mlp", ""), ("linear", "_linenc")]:
            save_dir = ROOT / f"chemical_v1_bilinear{suffix}_lomo" / heldout
            if not (save_dir / "best_model.pt").exists():
                print(f"{heldout:10s} {encoder_type:10s} {'(no checkpoint)':>20s}")
                continue
            try:
                out = load_encoder_outputs(save_dir, chem_features, encoder_type)
                cos = cos_pair(out[heldout_idx], out[neighbor_idx])
                summary = json.load(open(save_dir / "test_heldout_summary.json"))
                aucb = summary["AUCb"]
                aucm = summary["AUCm"]
            except Exception as e:
                print(f"{heldout:10s} {encoder_type:10s} ERROR: {e}")
                continue

            # Compute pos_prob_mean from saved predictions
            pred_path = save_dir / "test_predictions.npz"
            if pred_path.exists():
                pred = np.load(pred_path)
                pos_mask = pred["label"][:, heldout_idx] == 1
                pos_mean = float(pred["prob"][pos_mask, heldout_idx].mean())
            else:
                pos_mean = float("nan")

            print(f"{heldout:10s} {encoder_type:10s} {cos:+20.3f} {aucb:8.3f} {aucm:8.3f} {pos_mean:14.4f}")

    # Also compare all-mod results
    print("\n=== All-mod comparison ===")
    for encoder_type, suffix in [("mlp", ""), ("linear", "_linenc")]:
        save_dir = ROOT / f"chemical_v1_bilinear{suffix}"
        if (save_dir / "test_summary.json").exists():
            s = json.load(open(save_dir / "test_summary.json"))
            print(f"  {encoder_type:10s}  AUCb={s['AUCb']:.3f}  AUCm={s['AUCm']:.3f}  MCC={s['MCC']:.3f}")
        else:
            print(f"  {encoder_type:10s}  (no checkpoint)")


if __name__ == "__main__":
    main()

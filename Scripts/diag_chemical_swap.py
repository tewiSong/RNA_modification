"""Diagnostic: does swapping chemical_features[heldout] change predictions?

Tests candidate alpha: model treats m6A like its nearest neighbor (m6Am).
Also tests sensitivity of chemical conditioning at all (m7G control).
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


ROOT = Path("/ibex/user/songt/MultiRM")
device = torch.device("cpu")


def load_model(ckpt_path, scorer_type):
    table = load_modification_table(ROOT / "Data/modifications.csv")
    chem = build_chemical_feature_matrix(table)
    model = ChemicalMultiRMv1(
        embedding_path=str(ROOT / "Embeddings/embeddings_12RM.pkl"),
        chemical_features=chem,
        num_task=12,
        scorer_type=scorer_type,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    return model, chem


def run_forward_subset(model, x_tensor, batch=64):
    probs = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch):
            out = model(x_tensor[start:start + batch])
            logits = torch.stack(out, dim=1)  # (B, 12)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def experiment(ckpt_path, scorer_type, heldout, neighbor):
    print(f"\n=== {scorer_type} LOMO heldout={heldout}, neighbor swap = {neighbor} ===")
    model, chem = load_model(ckpt_path, scorer_type)

    heldout_idx = MODIFICATION_NAMES.index(heldout)
    neighbor_idx = MODIFICATION_NAMES.index(neighbor)

    # Load test split + filter positives for heldout class
    test = np.load(ROOT / "Results/paper_aligned/cache/test_51bp_3mer.npz")
    x_all = test["x"]
    y_all = test["y"]
    pos_mask = y_all[:, heldout_idx] == 1
    n_pos = pos_mask.sum()
    print(f"  test positives for {heldout}: {n_pos}")
    # Use up to 50 positives
    pos_idx = np.where(pos_mask)[0][:50]
    x_pos = torch.from_numpy(x_all[pos_idx].copy())

    # Original chemical features
    chem_orig = model.chemical_features.clone()

    # Baseline forward
    prob_orig = run_forward_subset(model, x_pos)
    p_h_orig = prob_orig[:, heldout_idx]
    p_n_orig = prob_orig[:, neighbor_idx]
    print(f"  ORIGINAL: prob[heldout={heldout}]: mean={p_h_orig.mean():.4f} max={p_h_orig.max():.4f}")
    print(f"  ORIGINAL: prob[neighbor={neighbor}]: mean={p_n_orig.mean():.4f} max={p_n_orig.max():.4f}")

    # SWAP: replace chemical_features[heldout] with chemical_features[neighbor]
    model.chemical_features[heldout_idx] = chem_orig[neighbor_idx]
    prob_swap = run_forward_subset(model, x_pos)
    p_h_swap = prob_swap[:, heldout_idx]
    print(f"  SWAPPED (chem[{heldout}]=chem[{neighbor}]): prob[heldout slot]: "
          f"mean={p_h_swap.mean():.4f} max={p_h_swap.max():.4f}")

    # Restore
    model.chemical_features[heldout_idx] = chem_orig[heldout_idx]

    # ZERO test (sanity)
    model.chemical_features[heldout_idx] = torch.zeros_like(chem_orig[heldout_idx])
    prob_zero = run_forward_subset(model, x_pos)
    p_h_zero = prob_zero[:, heldout_idx]
    print(f"  ZEROED chem[{heldout}]:                    prob[heldout slot]: "
          f"mean={p_h_zero.mean():.4f} max={p_h_zero.max():.4f}")
    model.chemical_features[heldout_idx] = chem_orig[heldout_idx]

    # Encoder output distances
    with torch.no_grad():
        chem_states = model.chemical_encoder(chem_orig)  # (12, 512)
    print(f"\n  chem_encoder OUTPUT pairwise distances (selected rows):")
    for j, name in enumerate(MODIFICATION_NAMES):
        d = torch.norm(chem_states[heldout_idx] - chem_states[j]).item()
        print(f"    ||enc({heldout}) - enc({name})|| = {d:.3f}")

    return {
        "orig_heldout": p_h_orig,
        "swap_heldout": p_h_swap,
        "zero_heldout": p_h_zero,
        "orig_neighbor": p_n_orig,
    }


if __name__ == "__main__":
    # Experiment 1: m6A held out, swap with m6Am (closest paper-wise: same A base, methylated)
    experiment(
        ckpt_path=ROOT / "Results/paper_aligned/chemical_v1_bilinear_lomo/m6A/best_model.pt",
        scorer_type="bilinear",
        heldout="m6A",
        neighbor="m6Am",
    )

    # Experiment 2: m7G held out (success case), swap with Gm
    experiment(
        ckpt_path=ROOT / "Results/paper_aligned/chemical_v1_bilinear_lomo/m7G/best_model.pt",
        scorer_type="bilinear",
        heldout="m7G",
        neighbor="Gm",
    )

    # Experiment 3: hypernetwork m6A (the model with AUCb 0.71-0.74 but mean=0)
    experiment(
        ckpt_path=ROOT / "Results/paper_aligned/chemical_v1_hypernetwork_lomo/m6A/best_model.pt",
        scorer_type="hypernetwork",
        heldout="m6A",
        neighbor="m6Am",
    )

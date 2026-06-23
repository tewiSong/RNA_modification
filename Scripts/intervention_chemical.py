"""Intervention experiment: does ChemicalMultiRMv1 actually condition on chemistry?

Loads a fully-trained checkpoint, then at test time replaces the chemical_features
matrix with various perturbations (pairwise swap, full shuffle, zero) and measures
how the per-modification AUCb changes.

Key question: when we swap chemical_features[i] with chemical_features[j],
does the logit column i now reflect modification j's predictions (logits follow
chemistry) or does it still reflect modification i (logits follow index)?
"""

import sys
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path("/ibex/user/songt/MultiRM/Scripts")))

from paper_multirm import ChemicalMultiRM, ChemicalMultiRMv1  # noqa: E402
from v0_data import (  # noqa: E402
    MODIFICATION_NAMES,
    build_chemical_feature_matrix,
    load_modification_table,
)


DEVICE = torch.device("cpu")
ROOT = Path("/ibex/user/songt/MultiRM")
EMBED_PATH = ROOT / "Embeddings/embeddings_12RM.pkl"
MOD_PATH = ROOT / "Data/modifications.csv"
TEST_NPZ = ROOT / "Results/paper_aligned/cache/test_51bp_3mer.npz"
V1_CKPT = ROOT / "Results/paper_aligned/chemical_v1_bilinear/best_model.pt"
V0_CKPT = ROOT / "Results/paper_aligned/chemical/best_model.pt"
OUT_DIR = ROOT / "Results/paper_aligned/intervention"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model(kind):
    mod_table = load_modification_table(str(MOD_PATH))
    chem = build_chemical_feature_matrix(mod_table)
    if kind == "v1":
        model = ChemicalMultiRMv1(str(EMBED_PATH), chem, num_heads=8, scorer_type="bilinear")
        ckpt = V1_CKPT
    elif kind == "v0":
        model = ChemicalMultiRM(str(EMBED_PATH), chem)
        ckpt = V0_CKPT
    else:
        raise ValueError(kind)
    state = torch.load(str(ckpt), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)
    return model


def load_test():
    data = np.load(str(TEST_NPZ))
    return data["x"].copy(), data["y"].copy()


@contextmanager
def patched_chemistry(model, new_features):
    original = model.chemical_features.detach().clone()
    with torch.no_grad():
        model.chemical_features.copy_(torch.as_tensor(new_features, dtype=original.dtype, device=original.device))
    try:
        yield
    finally:
        with torch.no_grad():
            model.chemical_features.copy_(original)


def collect_probs(model, x_tensor, batch_size=256):
    probs = []
    with torch.no_grad():
        for start in range(0, x_tensor.shape[0], batch_size):
            batch = x_tensor[start:start + batch_size].to(DEVICE)
            outs = model(batch)
            logits = torch.stack(outs, dim=1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def aucb_block(labels_col, scores_col):
    """Replicates compute_table's AUCb block: positives followed by equal-count negatives."""
    pos_idx = np.flatnonzero(labels_col == 1)
    if pos_idx.size == 0:
        return float("nan")
    start = int(pos_idx[0])
    block = int(pos_idx.size * 2)
    end = start + block
    if end > labels_col.shape[0]:
        return float("nan")
    sub_labels = labels_col[start:end]
    sub_scores = scores_col[start:end]
    if sub_labels.sum() == 0 or sub_labels.sum() == sub_labels.shape[0]:
        return float("nan")
    return float(roc_auc_score(sub_labels, sub_scores))


def aucb_block_using_block_of(reference_col, scores_col, label_col_for_score):
    """Use the row-window from reference_col (positives of col i define the window),
    but score with scores_col and labels with label_col_for_score.

    Used to ask: 'in the rows where modification i has its AUCb block,
    does the swapped logit column (which is what the model produced for index i
    but with chemistry j) predict modification j's labels in that window?'

    Simpler: just compute AUCb on label_col_for_score's own block.
    For the cross metric AUCb_i_j_swap, we use label_j's own block.
    """
    return aucb_block(label_col_for_score, scores_col)


def run_intervention(model, x_tensor, labels, model_label, swap_pairs):
    n_mod = len(MODIFICATION_NAMES)
    name_to_idx = {n: i for i, n in enumerate(MODIFICATION_NAMES)}

    # Baseline
    base_probs = collect_probs(model, x_tensor)
    baseline_aucb = {i: aucb_block(labels[:, i], base_probs[:, i]) for i in range(n_mod)}

    results = {"model": model_label, "baseline_aucb": baseline_aucb, "swaps": [], "shuffle": {}, "zero": {}, "constant_mean": {}}

    print(f"\n==== {model_label} ====", flush=True)
    print("Baseline AUCb per modification:")
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"  {n:<5s}: {baseline_aucb[i]:.4f}")

    # Intervention 1: pairwise swap
    print("\nIntervention 1: pairwise chemistry swap")
    print(f"{'pair':<14s} {'AUCb_i_normal':>14s} {'AUCb_i_swap':>13s} {'AUCb_j_using_col_i':>20s} {'AUCb_j_normal':>14s}")
    chem = model.chemical_features.detach().cpu().numpy().copy()
    for (name_i, name_j) in swap_pairs:
        i = name_to_idx[name_i]
        j = name_to_idx[name_j]
        swapped = chem.copy()
        swapped[[i, j]] = swapped[[j, i]]
        with patched_chemistry(model, swapped):
            swap_probs = collect_probs(model, x_tensor)
        # Logit column i now uses chemistry that originally lived at row j.
        aucb_i_swap = aucb_block(labels[:, i], swap_probs[:, i])  # column i predicts label i, but col i was computed with chem_j
        aucb_j_using_col_i = aucb_block(labels[:, j], swap_probs[:, i])  # does column i now predict label j?
        aucb_j_normal = baseline_aucb[j]
        results["swaps"].append({
            "i": name_i, "j": name_j,
            "AUCb_i_normal": baseline_aucb[i],
            "AUCb_i_swap": aucb_i_swap,
            "AUCb_j_using_col_i_swap": aucb_j_using_col_i,
            "AUCb_j_normal": aucb_j_normal,
        })
        print(f"{name_i:>5s}<->{name_j:<5s} {baseline_aucb[i]:>14.4f} {aucb_i_swap:>13.4f} {aucb_j_using_col_i:>20.4f} {aucb_j_normal:>14.4f}")

    # Intervention 2: full shuffle (seed 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n_mod)
    while np.any(perm == np.arange(n_mod)):
        perm = rng.permutation(n_mod)
    shuffled = chem[perm]
    with patched_chemistry(model, shuffled):
        shuf_probs = collect_probs(model, x_tensor)
    shuffle_aucb_orig_align = {i: aucb_block(labels[:, i], shuf_probs[:, i]) for i in range(n_mod)}
    shuffle_aucb_perm_align = {i: aucb_block(labels[:, perm[i]], shuf_probs[:, i]) for i in range(n_mod)}
    results["shuffle"] = {
        "perm": perm.tolist(),
        "aucb_orig_label_alignment": shuffle_aucb_orig_align,
        "aucb_permuted_label_alignment": shuffle_aucb_perm_align,
    }
    print("\nIntervention 2: full chemistry shuffle (perm={})".format([MODIFICATION_NAMES[k] for k in perm]))
    print(f"{'mod_i':<6s} {'baseline':>9s} {'shuf_origlbl':>13s} {'shuf_permlbl(j=perm[i])':>25s}")
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"{n:<6s} {baseline_aucb[i]:>9.4f} {shuffle_aucb_orig_align[i]:>13.4f} {shuffle_aucb_perm_align[i]:>25.4f}")

    # Intervention 3: zero out chemistry
    zero = np.zeros_like(chem)
    with patched_chemistry(model, zero):
        zero_probs = collect_probs(model, x_tensor)
    zero_aucb = {i: aucb_block(labels[:, i], zero_probs[:, i]) for i in range(n_mod)}
    results["zero"] = zero_aucb
    print("\nIntervention 3: chemistry = 0")
    print(f"{'mod_i':<6s} {'baseline':>9s} {'zeroed':>9s}")
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"{n:<6s} {baseline_aucb[i]:>9.4f} {zero_aucb[i]:>9.4f}")

    # Intervention 3b: constant = column mean (so the encoder still gets a "real-scale" input)
    const = np.tile(chem.mean(axis=0, keepdims=True), (n_mod, 1))
    with patched_chemistry(model, const):
        const_probs = collect_probs(model, x_tensor)
    const_aucb = {i: aucb_block(labels[:, i], const_probs[:, i]) for i in range(n_mod)}
    results["constant_mean"] = const_aucb
    print("\nIntervention 3b: chemistry = column-mean (identical for all 12 rows)")
    print(f"{'mod_i':<6s} {'baseline':>9s} {'const_mean':>11s}")
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"{n:<6s} {baseline_aucb[i]:>9.4f} {const_aucb[i]:>11.4f}")

    return results


def main():
    x, y = load_test()
    print(f"Test set: x shape {x.shape}, y shape {y.shape}", flush=True)
    x_tensor = torch.from_numpy(x).long()

    swap_pairs = [
        ("m6A", "Am"),
        ("m6A", "m6Am"),
        ("m6A", "m7G"),
        ("Psi", "m7G"),
        ("m1A", "m6A"),
    ]

    all_results = {}
    for kind, label in [("v1", "ChemicalMultiRMv1 (bilinear, all-mod training)"),
                        ("v0", "ChemicalMultiRM v0 (all-mod training)")]:
        model = build_model(kind)
        results = run_intervention(model, x_tensor, y, label, swap_pairs)
        all_results[kind] = results

    out_path = OUT_DIR / "intervention_results.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved results to {out_path}", flush=True)


if __name__ == "__main__":
    main()

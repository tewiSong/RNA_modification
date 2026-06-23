"""
Decisive test: inject full-train model's well-separated mod_{m6A} into the
LOMO m6A model. If col[m6A] then fires on m6A test positives → encoder
collapse IS the bottleneck. If not → the deeper issue is that the scorer
+ RNA features cannot distinguish m6A sites from m6Am sites at all.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1, RmDataset, read_split_as_kmers
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")
M6A = MODIFICATION_NAMES.index("m6A")
M6AM = MODIFICATION_NAMES.index("m6Am")


def load_model(save_dir, chem_features, encoder_type="mlp"):
    config = json.load(open(Path(save_dir) / "config.json"))
    model = ChemicalMultiRMv1(
        "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
        chem_features,
        num_heads=config["num_heads"],
        scorer_type=config["scorer_type"],
        chemical_encoder_type=config.get("chemical_encoder_type", encoder_type),
    )
    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def predict_with_mod_override(model, loader, mod_override=None):
    """If mod_override is given, replace model's chemical_states with it during forward."""
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            rna_emb = model.embed(x)
            rna_out, _ = model.NaiveBiLSTM(rna_emb)
            if mod_override is None:
                chemical_states = model.chemical_encoder(model.chemical_features)
            else:
                chemical_states = mod_override.to(DEVICE)
            context_states = model.attention(rna_out, chemical_states)
            center_state = rna_out[:, rna_out.shape[1] // 2, :]
            logits = model.scorer(center_state, context_states, chemical_states)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def report(label, prob, mod_idx, tag):
    pos_mask = label[:, mod_idx] == 1
    pos_indices = np.flatnonzero(pos_mask)
    start = int(pos_indices[0])
    end = start + pos_indices.shape[0] * 2
    aucb = roc_auc_score(label[start:end, mod_idx], prob[start:end, mod_idx])
    aucm = roc_auc_score(label[:, mod_idx], prob[:, mod_idx])
    pos_mean = prob[pos_mask, mod_idx].mean()
    pos_max = prob[pos_mask, mod_idx].max()
    print(f"    {tag:55s}  pos_mean={pos_mean:.4f}  pos_max={pos_max:.4f}  AUCb={aucb:.3f}  AUCm={aucm:.3f}")


def main():
    mod_table = load_modification_table("/ibex/user/songt/MultiRM/Data/modifications.csv")
    chem_features = build_chemical_feature_matrix(mod_table)

    test_data = read_split_as_kmers(
        "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5", "test", 51,
        "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
        "/ibex/user/songt/MultiRM/Results/paper_aligned/cache",
    )
    label = test_data["y"]
    loader = DataLoader(RmDataset(test_data), batch_size=128, shuffle=False)

    print("=== Setup: get mod vectors from each model ===")
    full_mlp = load_model(ROOT / "chemical_v1_bilinear", chem_features)
    full_lin = load_model(ROOT / "chemical_v1_bilinear_linenc", chem_features)
    lomo_mlp = load_model(ROOT / "chemical_v1_bilinear_lomo" / "m6A", chem_features)
    lomo_lin = load_model(ROOT / "chemical_v1_bilinear_linenc_lomo" / "m6A", chem_features)

    with torch.no_grad():
        mod_full_mlp = full_mlp.chemical_encoder(full_mlp.chemical_features).cpu()
        mod_full_lin = full_lin.chemical_encoder(full_lin.chemical_features).cpu()
        mod_lomo_mlp = lomo_mlp.chemical_encoder(lomo_mlp.chemical_features).cpu()

    def cos(a, b): return float(a @ b / (a.norm() * b.norm() + 1e-12))
    print(f"  full_mlp:   cos(mod_m6A, mod_m6Am) = {cos(mod_full_mlp[M6A], mod_full_mlp[M6AM]):+.3f}")
    print(f"  full_lin:   cos(mod_m6A, mod_m6Am) = {cos(mod_full_lin[M6A], mod_full_lin[M6AM]):+.3f}")
    print(f"  lomo_mlp:   cos(mod_m6A, mod_m6Am) = {cos(mod_lomo_mlp[M6A], mod_lomo_mlp[M6AM]):+.3f}")

    print("\n=== Run LOMO m6A model with various mod-vector overrides ===")
    print("(Same scorer/attention/BiLSTM trained without m6A positives, different mod vectors)\n")

    # Baseline: normal mod from lomo's collapsed encoder
    prob = predict_with_mod_override(lomo_mlp, loader, None)
    report(label, prob, M6A, "[baseline] lomo mod_m6A (mlp, collapsed)")

    # Inject full-train mlp's mod_m6A (well-separated)
    override = mod_lomo_mlp.clone()
    override[M6A] = mod_full_mlp[M6A]
    prob = predict_with_mod_override(lomo_mlp, loader, override)
    report(label, prob, M6A, "[inject] full_mlp mod_m6A (cos 0.095 from m6Am)")

    # Inject full-train linear's mod_m6A
    override = mod_lomo_mlp.clone()
    override[M6A] = mod_full_lin[M6A]
    prob = predict_with_mod_override(lomo_mlp, loader, override)
    report(label, prob, M6A, "[inject] full_lin mod_m6A")

    # Inject manually orthogonalized mod_m6A: subtract m6Am component
    orth = mod_lomo_mlp[M6A] - (mod_lomo_mlp[M6A] @ mod_lomo_mlp[M6AM]) / (mod_lomo_mlp[M6AM] @ mod_lomo_mlp[M6AM]) * mod_lomo_mlp[M6AM]
    orth = orth / orth.norm() * mod_lomo_mlp[M6A].norm()  # restore magnitude
    override = mod_lomo_mlp.clone()
    override[M6A] = orth
    prob = predict_with_mod_override(lomo_mlp, loader, override)
    report(label, prob, M6A, "[inject] orthogonalized mod_m6A (cos≈0 from m6Am)")

    # Inject pure random vector with same magnitude (3 trials for robustness)
    print("\n  --- Random mod_m6A controls (3 seeds) ---")
    for seed in [0, 1, 42]:
        torch.manual_seed(seed)
        rand_mod = torch.randn_like(mod_lomo_mlp[M6A])
        rand_mod = rand_mod / rand_mod.norm() * mod_lomo_mlp[M6A].norm()
        override = mod_lomo_mlp.clone()
        override[M6A] = rand_mod
        prob = predict_with_mod_override(lomo_mlp, loader, override)
        report(label, prob, M6A, f"[inject] random mod_m6A seed={seed}")

    # Check: does orth ALSO accidentally predict m6Am better/worse? (control)
    print("\n  --- What does orth_m6A do for col[m6Am] (sanity) ---")
    orth = mod_lomo_mlp[M6A] - (mod_lomo_mlp[M6A] @ mod_lomo_mlp[M6AM]) / (mod_lomo_mlp[M6AM] @ mod_lomo_mlp[M6AM]) * mod_lomo_mlp[M6AM]
    orth = orth / orth.norm() * mod_lomo_mlp[M6A].norm()
    override = mod_lomo_mlp.clone()
    override[M6A] = orth
    prob = predict_with_mod_override(lomo_mlp, loader, override)
    report(label, prob, M6AM, "[inject] orth mod_m6A, but READ col[m6Am label]")
    # And the col[m6A] using orth scoring m6Am labels
    pos_mask_m6am = label[:, M6AM] == 1
    if pos_mask_m6am.sum() > 0:
        # Score col[m6A using orth] against m6Am label
        aucm_cross = roc_auc_score(label[:, M6AM], prob[:, M6A])
        print(f"    cross check: AUCm(label=m6Am, score=col[m6A] using orth mod) = {aucm_cross:.3f}")

    print("\n=== Same overrides on linear-encoder LOMO m6A model ===")
    with torch.no_grad():
        mod_lomo_lin = lomo_lin.chemical_encoder(lomo_lin.chemical_features).cpu()
    print(f"  lomo_lin:   cos(mod_m6A, mod_m6Am) = {cos(mod_lomo_lin[M6A], mod_lomo_lin[M6AM]):+.3f}")

    prob = predict_with_mod_override(lomo_lin, loader, None)
    report(label, prob, M6A, "[baseline] lomo_lin mod_m6A (also collapsed)")

    # Inject orth on linear-encoder lomo model
    orth = mod_lomo_lin[M6A] - (mod_lomo_lin[M6A] @ mod_lomo_lin[M6AM]) / (mod_lomo_lin[M6AM] @ mod_lomo_lin[M6AM]) * mod_lomo_lin[M6AM]
    orth = orth / orth.norm() * mod_lomo_lin[M6A].norm()
    override = mod_lomo_lin.clone()
    override[M6A] = orth
    prob = predict_with_mod_override(lomo_lin, loader, override)
    report(label, prob, M6A, "[inject] orth mod_m6A on lomo_lin")

    print("\nDone.")


if __name__ == "__main__":
    main()

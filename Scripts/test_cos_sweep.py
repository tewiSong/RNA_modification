"""
For the LOMO m6A model, synthesize mod_m6A at controlled cosine angles
to mod_m6Am, then measure m6A AUCm. This quantifies: how separated must
mod_m6A be from mod_m6Am for the model's latent m6A signal to come out?
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


def load_model(save_dir, chem_features):
    config = json.load(open(Path(save_dir) / "config.json"))
    model = ChemicalMultiRMv1(
        "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
        chem_features,
        num_heads=config["num_heads"],
        scorer_type=config["scorer_type"],
        chemical_encoder_type=config.get("chemical_encoder_type", "mlp"),
    )
    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def predict_with_mod_override(model, loader, mod_override):
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            rna_emb = model.embed(x)
            rna_out, _ = model.NaiveBiLSTM(rna_emb)
            chemical_states = mod_override.to(DEVICE)
            context_states = model.attention(rna_out, chemical_states)
            center_state = rna_out[:, rna_out.shape[1] // 2, :]
            logits = model.scorer(center_state, context_states, chemical_states)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def aucm_for(prob, label, k):
    return float(roc_auc_score(label[:, k], prob[:, k]))


def aucb_for(prob, label, k):
    pos = np.flatnonzero(label[:, k] == 1)
    start, end = int(pos[0]), int(pos[0]) + pos.shape[0] * 2
    return float(roc_auc_score(label[start:end, k], prob[start:end, k]))


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

    lomo = load_model(ROOT / "chemical_v1_bilinear_lomo" / "m6A", chem_features)
    with torch.no_grad():
        base = lomo.chemical_encoder(lomo.chemical_features).cpu()

    mod_m6Am = base[M6AM]
    mod_m6A_orig = base[M6A]
    mag = mod_m6A_orig.norm()

    # Build orthonormal basis: u1 = m6Am direction, u2 = orth(m6A wrt m6Am)
    u1 = mod_m6Am / mod_m6Am.norm()
    orth = mod_m6A_orig - (mod_m6A_orig @ u1) * u1
    u2 = orth / orth.norm()

    print("=== Sweep cos(synthetic mod_m6A, mod_m6Am) ===")
    print(f"  Original cos in lomo encoder: {float(mod_m6A_orig @ u1 / mag):.3f}")
    print(f"  Original AUCm on m6A label: (see baseline below)")
    print()
    print(f"{'cos':>6s}  {'AUCm(m6A)':>11s}  {'AUCb(m6A)':>11s}  {'AUCm(m6Am sanity)':>20s}")
    print("-" * 60)

    cosines = [0.99, 0.95, 0.9, 0.85, 0.7, 0.5, 0.3, 0.1, 0.0, -0.3, -0.7]
    # For each target cos, build mod_m6A = c * u1 + sqrt(1-c^2) * u2, then scale to original magnitude
    for c in cosines:
        s = float(np.sqrt(max(0.0, 1.0 - c * c)))
        synth = (c * u1 + s * u2) * mag  # has norm = mag, dot with u1 = c * mag, so cos with mod_m6Am direction is c
        override = base.clone()
        override[M6A] = synth
        prob = predict_with_mod_override(lomo, loader, override)
        a_m6a = aucm_for(prob, label, M6A)
        b_m6a = aucb_for(prob, label, M6A)
        # Sanity: scoring col[m6A] against m6Am label - should NOT track if synth is genuinely m6A-discriminating
        a_m6am_cross = float(roc_auc_score(label[:, M6AM], prob[:, M6A]))
        print(f"  {c:+5.2f}  {a_m6a:11.3f}  {b_m6a:11.3f}  {a_m6am_cross:20.3f}")

    # Also: random directions baseline (10 seeds)
    print("\n--- Random direction baselines (10 seeds, same magnitude) ---")
    print(f"{'seed':>6s}  {'AUCm(m6A)':>11s}  {'AUCb(m6A)':>11s}  {'AUCm(m6Am cross)':>20s}")
    for seed in range(10):
        torch.manual_seed(seed)
        rand = torch.randn(base.shape[1])
        rand = rand / rand.norm() * mag
        override = base.clone()
        override[M6A] = rand
        prob = predict_with_mod_override(lomo, loader, override)
        a_m6a = aucm_for(prob, label, M6A)
        b_m6a = aucb_for(prob, label, M6A)
        a_m6am_cross = float(roc_auc_score(label[:, M6AM], prob[:, M6A]))
        c_with_m6am = float((rand / mag) @ u1)
        print(f"  {seed:6d}  {a_m6a:11.3f}  {b_m6a:11.3f}  {a_m6am_cross:20.3f}  (cos with m6Am: {c_with_m6am:+.3f})")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""
Intervention experiment: validate the "garbage column" hypothesis.

If the hypothesis is right:
  - Test 1: Loading m6A LOMO model and replacing chemical_features[m6A] with
    chemical_features[m6Am] should NOT raise m6A predictions, because the
    m6A column itself has been trained to output ~0 regardless of input
    chemistry. The "garbage column" is the scorer's m6A output column.
  - Test 2: Same swap on m7G LOMO model: replacing m7G chem with Gm chem
    should produce smaller changes than for m6A (m7G column isn't a garbage
    column).
  - Test 3: On the all-mod (non-LOMO) model, swap m6A <-> m6Am chemistry. If
    chemistry actually conditions predictions, the m6A column should now
    behave like m6Am column.
  - Test 4: Zero out all chemistry. If model still distinguishes modifications,
    chemistry isn't really being used (modid-style behavior).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1, RmDataset, read_split_as_kmers
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODS_PATH = "/ibex/user/songt/MultiRM/Data/modifications.csv"
EMB_PATH = "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl"
DATA_PATH = "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5"
CACHE_DIR = "/ibex/user/songt/MultiRM/Results/paper_aligned/cache"


def tanimoto(mod_table, i, j):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_i = gen.GetFingerprint(Chem.MolFromSmiles(mod_table.iloc[i]["canonical_smiles"]))
    fp_j = gen.GetFingerprint(Chem.MolFromSmiles(mod_table.iloc[j]["canonical_smiles"]))
    return float(DataStructs.TanimotoSimilarity(fp_i, fp_j))


def aucb_block(label, prob, mod_idx):
    pos = np.flatnonzero(label[:, mod_idx] == 1)
    start = int(pos[0])
    end = start + pos.shape[0] * 2
    if label[start:end, mod_idx].sum() == 0 or (1 - label[start:end, mod_idx]).sum() == 0:
        return float("nan")
    return float(roc_auc_score(label[start:end, mod_idx], prob[start:end, mod_idx]))


def aucm(label, prob, mod_idx):
    return float(roc_auc_score(label[:, mod_idx], prob[:, mod_idx]))


def report(prob, label, mod_idx, tag):
    pos_mask = label[:, mod_idx] == 1
    n_pos = int(pos_mask.sum())
    pos_mean = float(prob[pos_mask, mod_idx].mean())
    pos_max = float(prob[pos_mask, mod_idx].max())
    aucb = aucb_block(label, prob, mod_idx)
    aucm_val = aucm(label, prob, mod_idx)
    print(f"    {tag:45s}  n_pos={n_pos:4d}  pos_prob mean={pos_mean:.4f} max={pos_max:.4f}  AUCb={aucb:.3f}  AUCm={aucm_val:.3f}")


def predict(model, test_loader):
    probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            outputs = model(x.to(DEVICE))
            logits = torch.stack(outputs, dim=1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def load_v1_model(save_dir, chem_features):
    config = json.load(open(Path(save_dir) / "config.json"))
    model = ChemicalMultiRMv1(EMB_PATH, chem_features, num_heads=config["num_heads"], scorer_type=config["scorer_type"])
    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def main():
    mod_table = load_modification_table(MODS_PATH)
    chem_features = build_chemical_feature_matrix(mod_table)
    chem_features_t = torch.as_tensor(chem_features, dtype=torch.float32)

    test_data = read_split_as_kmers(DATA_PATH, "test", 51, EMB_PATH, CACHE_DIR)
    label = test_data["y"]
    loader = DataLoader(RmDataset(test_data), batch_size=128, shuffle=False)

    name_to_idx = {n: i for i, n in enumerate(MODIFICATION_NAMES)}
    M6A, M6AM, M7G, GM = name_to_idx["m6A"], name_to_idx["m6Am"], name_to_idx["m7G"], name_to_idx["Gm"]

    print("=== Tanimoto distances (verifying subagent #3's numbers) ===")
    pairs = [("m6A", "m6Am"), ("m6A", "m1A"), ("m6A", "Am"), ("m6A", "I"), ("m6A", "m7G"),
             ("m7G", "Gm"), ("m7G", "m6A"), ("Psi", "Um"), ("Am", "m6Am")]
    for a, b in pairs:
        t = tanimoto(mod_table, name_to_idx[a], name_to_idx[b])
        print(f"  Tanimoto({a:5s}, {b:5s}) = {t:.3f}")

    # ---------- Test 1: m6A LOMO model intervention ----------
    print("\n=== Test 1: m6A LOMO (bilinear) — swap m6A chemistry with neighbors ===")
    m6a_lomo = load_v1_model("/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/m6A", chem_features)
    original = m6a_lomo.chemical_features.clone()

    prob_normal = predict(m6a_lomo, loader)
    report(prob_normal, label, M6A, "normal (m6A chemistry)")

    for swap_name in ["m6Am", "m1A", "Am", "I", "m7G", "Psi"]:
        j = name_to_idx[swap_name]
        m6a_lomo.chemical_features[M6A] = chem_features_t[j].to(DEVICE)
        prob_swap = predict(m6a_lomo, loader)
        report(prob_swap, label, M6A, f"swap chem m6A->{swap_name}, read col m6A")
        m6a_lomo.chemical_features.copy_(original)

    print("\n  Also: set m6A chemistry to ZEROS (no chemistry signal)")
    m6a_lomo.chemical_features[M6A].zero_()
    prob_zero = predict(m6a_lomo, loader)
    report(prob_zero, label, M6A, "swap chem m6A->zeros, read col m6A")
    m6a_lomo.chemical_features.copy_(original)

    print("\n  Also: set m6A chemistry to RANDOM values (~N(0,1))")
    torch.manual_seed(0)
    m6a_lomo.chemical_features[M6A] = torch.randn_like(chem_features_t[M6A]).to(DEVICE)
    prob_rand = predict(m6a_lomo, loader)
    report(prob_rand, label, M6A, "swap chem m6A->random, read col m6A")
    m6a_lomo.chemical_features.copy_(original)

    # ---------- Test 2: m7G LOMO model intervention (control) ----------
    print("\n=== Test 2: m7G LOMO (bilinear) — swap m7G chemistry with neighbors (CONTROL) ===")
    m7g_lomo = load_v1_model("/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/m7G", chem_features)
    original7 = m7g_lomo.chemical_features.clone()

    prob_normal7 = predict(m7g_lomo, loader)
    report(prob_normal7, label, M7G, "normal (m7G chemistry)")

    for swap_name in ["Gm", "m6A", "Psi", "I"]:
        j = name_to_idx[swap_name]
        m7g_lomo.chemical_features[M7G] = chem_features_t[j].to(DEVICE)
        prob_swap = predict(m7g_lomo, loader)
        report(prob_swap, label, M7G, f"swap chem m7G->{swap_name}, read col m7G")
        m7g_lomo.chemical_features.copy_(original7)

    # ---------- Test 3: full-train model — does chemistry actually condition? ----------
    print("\n=== Test 3: full-train (all-mod) bilinear — chemistry conditioning strength ===")
    full = load_v1_model("/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear", chem_features)
    original_full = full.chemical_features.clone()

    prob_full_normal = predict(full, loader)
    print("\n  NORMAL chemistry, per-column metrics:")
    for mod_idx in [M6A, M6AM, M7G, GM]:
        report(prob_full_normal, label, mod_idx, f"col[{MODIFICATION_NAMES[mod_idx]}]")

    print("\n  After swap m6A <-> m6Am chemistry (both directions):")
    full.chemical_features[M6A] = chem_features_t[M6AM].to(DEVICE)
    full.chemical_features[M6AM] = chem_features_t[M6A].to(DEVICE)
    prob_swap = predict(full, loader)
    full.chemical_features.copy_(original_full)
    for mod_idx in [M6A, M6AM]:
        report(prob_swap, label, mod_idx, f"col[{MODIFICATION_NAMES[mod_idx]}] (chem now from {'m6Am' if mod_idx==M6A else 'm6A'})")

    # Cross-label: does m6A col now predict m6Am ground truth?
    print("\n  Cross-label: scoring col[m6A] against label[m6Am] (should be high if chemistry conditioning is strong):")
    aucb_cross = float(roc_auc_score(label[:, M6AM], prob_swap[:, M6A]))
    aucb_baseline = float(roc_auc_score(label[:, M6AM], prob_full_normal[:, M6AM]))
    print(f"    AUCm(label=m6Am, score=col[m6A] using m6Am chem) = {aucb_cross:.3f}  vs  baseline AUCm(label=m6Am, score=col[m6Am]) = {aucb_baseline:.3f}")

    # ---------- Test 4: full-train model with chemistry zeroed out ----------
    print("\n=== Test 4: full-train model with ALL chemistry zeroed — is chemistry used at all? ===")
    full.chemical_features.zero_()
    prob_zero = predict(full, loader)
    full.chemical_features.copy_(original_full)
    print("  Per-column AUCb with zeroed chemistry vs normal:")
    for mod_idx in range(12):
        a_zero = aucb_block(label, prob_zero, mod_idx)
        a_normal = aucb_block(label, prob_full_normal, mod_idx)
        print(f"    col[{MODIFICATION_NAMES[mod_idx]:5s}]  AUCb_zero={a_zero:.3f}  AUCb_normal={a_normal:.3f}  delta={a_normal - a_zero:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

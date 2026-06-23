"""
Verify the root-cause hypothesis: does the chemical encoder destroy
input-space chemistry similarity at its output?

Specifically:
1. Compute Tanimoto / cosine similarity matrix on RAW Morgan FP (input)
2. Compute cosine similarity matrix on encoder OUTPUTS for each trained model
   (full-train, m6A LOMO, m7G LOMO)
3. Compare. If output similarities are much smaller / negative for chemically
   close pairs, the encoder is doing contrastive separation → root cause.
4. Specifically check (m6A, m6Am) input vs output similarity across models.
5. Also check whether the encoder learns to make CO-OCCURRING modifications
   even more distinct (e.g. if m6A LOMO model has not seen m6A positives,
   does its encoder still distinguish m6A from m6Am?).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODS_PATH = "/ibex/user/songt/MultiRM/Data/modifications.csv"
EMB_PATH = "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl"


def cos_matrix(vectors):
    norm = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norm
    return unit @ unit.T


def tanimoto_matrix(mod_table):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(row["canonical_smiles"])) for _, row in mod_table.iterrows()]
    K = len(fps)
    mat = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            mat[i, j] = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return mat


def load_model(save_dir, chem_features):
    config = json.load(open(Path(save_dir) / "config.json"))
    model = ChemicalMultiRMv1(EMB_PATH, chem_features, num_heads=config["num_heads"], scorer_type=config["scorer_type"])
    model.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def get_encoder_outputs(model):
    with torch.no_grad():
        out = model.chemical_encoder(model.chemical_features).cpu().numpy()
    return out


def report_pair(name, i, j, tani_input, cos_input, cos_out):
    a, b = MODIFICATION_NAMES[i], MODIFICATION_NAMES[j]
    print(f"  {name:50s}  Tani_FP({a},{b})={tani_input:.3f}  cos_FP_full={cos_input:.3f}  cos_encoder_out={cos_out:+.3f}")


def print_matrix(label, mat, labels):
    print(f"\n  {label}:")
    header = "        " + " ".join(f"{l:>6s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = " ".join(f"{mat[i, j]:+6.3f}" for j in range(len(labels)))
        print(f"  {row_label:>6s} {row}")


def main():
    mod_table = load_modification_table(MODS_PATH)
    chem_features = build_chemical_feature_matrix(mod_table)

    print("=== Reference similarities at INPUT ===")
    tani = tanimoto_matrix(mod_table)
    cos_input = cos_matrix(chem_features)
    print_matrix("Tanimoto (Morgan FP)", tani, MODIFICATION_NAMES)
    print_matrix("Cosine of full chemistry features (FP+desc+base)", cos_input, MODIFICATION_NAMES)

    print("\n\n=== Encoder OUTPUT cosine similarity for each trained model ===")
    models = {
        "full_train (chemical_v1_bilinear, all 12 seen)": "/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear",
        "LOMO m6A (chemical_v1_bilinear_lomo)": "/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/m6A",
        "LOMO m7G (chemical_v1_bilinear_lomo)": "/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/m7G",
        "LOMO Am  (chemical_v1_bilinear_lomo)": "/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/Am",
        "LOMO Psi (chemical_v1_bilinear_lomo)": "/ibex/user/songt/MultiRM/Results/paper_aligned/chemical_v1_bilinear_lomo/Psi",
    }
    for name, save_dir in models.items():
        if not (Path(save_dir) / "best_model.pt").exists():
            print(f"  SKIP {name}: no checkpoint")
            continue
        print(f"\n----- {name} -----")
        model = load_model(save_dir, chem_features)
        enc_out = get_encoder_outputs(model)
        cos_out = cos_matrix(enc_out)
        print_matrix("Encoder OUTPUT cosine", cos_out, MODIFICATION_NAMES)

    print("\n\n=== Hypothesis test: does encoder shrink high-similarity pairs? ===")
    print("If hypothesis correct: encoder OUTPUT cos for (m6A, m6Am) << input cos.")
    print()
    print(f"  Input cos (full features) m6A-m6Am: {cos_input[7, 8]:.3f}")
    print(f"  Tanimoto m6A-m6Am: {tani[7, 8]:.3f}")
    for name, save_dir in models.items():
        if not (Path(save_dir) / "best_model.pt").exists():
            continue
        model = load_model(save_dir, chem_features)
        enc_out = get_encoder_outputs(model)
        cos_out = cos_matrix(enc_out)
        c_m6A_m6Am = cos_out[7, 8]
        c_m6A_m1A = cos_out[7, 4]
        c_m7G_Gm = cos_out[9, 2]
        c_m6A_m7G = cos_out[7, 9]
        print(f"  {name[:55]:55s}  cos(m6A,m6Am)={c_m6A_m6Am:+.3f}  cos(m6A,m1A)={c_m6A_m1A:+.3f}  cos(m7G,Gm)={c_m7G_Gm:+.3f}  cos(m6A,m7G)={c_m6A_m7G:+.3f}")

    print("\n=== Also: pairwise sim correlation input vs output ===")
    print("Spearman/Pearson correlation between input-FP cosine and encoder-output cosine.")
    print("If hypothesis correct: correlation should be LOW or NEGATIVE for similar pairs.")
    from scipy import stats
    upper = np.triu_indices(12, k=1)
    input_sims = cos_input[upper]
    tani_sims = tani[upper]
    for name, save_dir in models.items():
        if not (Path(save_dir) / "best_model.pt").exists():
            continue
        model = load_model(save_dir, chem_features)
        enc_out = get_encoder_outputs(model)
        cos_out = cos_matrix(enc_out)
        out_sims = cos_out[upper]
        sp = stats.spearmanr(input_sims, out_sims)
        pr = stats.pearsonr(input_sims, out_sims)
        sp_tani = stats.spearmanr(tani_sims, out_sims)
        print(f"  {name[:55]:55s}  Spearman(cos_input,cos_out)={sp.statistic:+.3f}  Pearson={pr.statistic:+.3f}  Spearman(Tani,cos_out)={sp_tani.statistic:+.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

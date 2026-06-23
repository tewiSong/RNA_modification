"""Generate the final 12-mod LOMO summary + scatter plot for Tani vs AUCm."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1
from v0_data import build_chemical_feature_matrix, load_modification_table, MODIFICATION_NAMES

ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")


def main():
    mod_table = load_modification_table("/ibex/user/songt/MultiRM/Data/modifications.csv")
    chem = build_chemical_feature_matrix(mod_table)
    n2i = {n: i for i, n in enumerate(MODIFICATION_NAMES)}

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(r["canonical_smiles"])) for _, r in mod_table.iterrows()]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print("=== Full 12-mod LOMO scan (chemical_v1_bilinear, weighted_bce + AdamW + early stop) ===\n")
    print(f"{'held':6s}  {'nearest_twin (Tani)':>23s}  {'enc_cos':>8s}  {'AUCb':>6s}  {'AUCm':>6s}  {'verdict':<10s}")
    print("-" * 75)

    rows = []
    for h in MODIFICATION_NAMES:
        hi = n2i[h]
        tanis = sorted([(MODIFICATION_NAMES[j], float(DataStructs.TanimotoSimilarity(fps[hi], fps[j])))
                        for j in range(12) if j != hi], key=lambda x: -x[1])
        nb_name, nb_tani = tanis[0]
        nbi = n2i[nb_name]

        d = ROOT / "chemical_v1_bilinear_lomo" / h
        if not (d / "test_heldout_summary.json").exists():
            print(f"{h:6s}  {nb_name+' ('+f'{nb_tani:.2f})':>23s}  (no checkpoint)")
            continue
        cfg = json.load(open(d / "config.json"))
        m = ChemicalMultiRMv1(
            "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
            chem, num_heads=cfg["num_heads"], scorer_type=cfg["scorer_type"],
            chemical_encoder_type=cfg.get("chemical_encoder_type", "mlp"),
        )
        m.load_state_dict(torch.load(d / "best_model.pt", map_location="cpu"))
        m.eval()
        with torch.no_grad():
            out = m.chemical_encoder(m.chemical_features).numpy()
        c = cos(out[hi], out[nbi])
        s = json.load(open(d / "test_heldout_summary.json"))
        verdict = "PASS" if s["AUCm"] > 0.5 else "FAIL"
        rows.append((h, nb_name, nb_tani, c, s["AUCb"], s["AUCm"], verdict))
        print(f"{h:6s}  {nb_name+' ('+f'{nb_tani:.2f})':>23s}  {c:+8.3f}  {s['AUCb']:6.3f}  {s['AUCm']:6.3f}  {verdict:<10s}")

    # Site_weight + encoder sweep on m6A
    print("\n=== m6A LOMO with site features (B experiments) ===\n")
    print(f"{'site_w':>8s}  {'encoder':>14s}  {'AUCb':>6s}  {'AUCm':>6s}")
    print("-" * 50)
    for sw in [5, 12]:
        for enc in ["mlp", "frozen_linear"]:
            d = ROOT / f"chemical_v1_bilinear_sw{sw}_{enc}_lomo" / "m6A"
            if (d / "test_heldout_summary.json").exists():
                s = json.load(open(d / "test_heldout_summary.json"))
                print(f"{sw:8d}  {enc:>14s}  {s['AUCb']:6.3f}  {s['AUCm']:6.3f}")
            else:
                print(f"{sw:8d}  {enc:>14s}  (no result)")

    # Also report baseline for comparison
    bd = ROOT / "chemical_v1_bilinear_lomo" / "m6A"
    if (bd / "test_heldout_summary.json").exists():
        s = json.load(open(bd / "test_heldout_summary.json"))
        print(f"   (sw=0, mlp baseline)        {s['AUCb']:6.3f}  {s['AUCm']:6.3f}")

    # Generate scatter plot data
    print("\n=== Scatter data (Tani vs AUCm) ===")
    print("Tani,AUCm,Modification,Verdict")
    for h, nb, t, c, b, a, v in rows:
        print(f"{t:.3f},{a:.3f},{h},{v}")


if __name__ == "__main__":
    main()

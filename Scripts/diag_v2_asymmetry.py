"""Diagnose why v2 sharp-attn (tau=0.4) rescues m6A LOMO but harms Am LOMO.

Four experiments:
  E1: attention map analysis — where does col[heldout] attend? cf col[m6Am]?
  E2: mod-vector swap intervention — replace mod_heldout, watch AUCm.
  E3: RNA motif analysis on test positives (m6A vs Am vs m6Am 3-mer + IC).
  E4: encoder cos(mod_heldout, mod_m6Am): v1 mlp vs v2 linear.

CPU only. Loads npz test set; ~1200 forward passes per model.
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from paper_multirm import ChemicalMultiRMv1, ChemicalMultiRMv2, RmDataset, read_split_as_kmers
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table

DEVICE = "cpu"
ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")
EMB = "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl"
DATA = "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5"
CACHE = "/ibex/user/songt/MultiRM/Results/paper_aligned/cache"
MODCSV = "/ibex/user/songt/MultiRM/Data/modifications.csv"

IDX = {n: i for i, n in enumerate(MODIFICATION_NAMES)}


def load_v2(save_dir, chem_features):
    cfg = json.load(open(Path(save_dir) / "config.json"))
    m = ChemicalMultiRMv2(EMB, chem_features, num_task=12,
                         tau=cfg["tau"],
                         chemical_encoder_type=cfg["chemical_encoder_type"])
    m.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    m.eval().to(DEVICE)
    return m, cfg


def load_v1(save_dir, chem_features):
    cfg = json.load(open(Path(save_dir) / "config.json"))
    m = ChemicalMultiRMv1(EMB, chem_features,
                         num_heads=cfg["num_heads"],
                         scorer_type=cfg["scorer_type"],
                         chemical_encoder_type=cfg.get("chemical_encoder_type", "mlp"))
    m.load_state_dict(torch.load(Path(save_dir) / "best_model.pt", map_location=DEVICE))
    m.eval().to(DEVICE)
    return m, cfg


def v2_forward_full(model, loader):
    """Return (probs[N,K], attn[N,K,L]) for v2 sharp-attn."""
    probs, attns = [], []
    with torch.no_grad():
        chem = model.chemical_encoder(model.chemical_features)  # (K,D)
        Q = model.scorer.Wq(chem)
        for x, _ in loader:
            x = x.to(DEVICE)
            emb = model.embed(x)
            rna, _ = model.NaiveBiLSTM(emb)  # (B,L,D)
            Kmat = model.scorer.Wk(rna)
            Vmat = model.scorer.Wv(rna)
            scale = model.scorer.tau * (Kmat.shape[-1] ** 0.5)
            logits = torch.einsum("kd,bld->bkl", Q, Kmat) / scale
            attn = logits.softmax(dim=-1)  # (B,K,L)
            r = torch.einsum("bkl,bld->bkd", attn, Vmat)
            score = model.scorer.mlp(r).squeeze(-1)  # (B,K)
            probs.append(torch.sigmoid(score).cpu().numpy())
            attns.append(attn.cpu().numpy())
    return np.concatenate(probs, 0), np.concatenate(attns, 0)


def v2_forward_with_mod_override(model, loader, mod_override):
    """Run v2 forward replacing chemistry encoder output with mod_override (K,D)."""
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            emb = model.embed(x)
            rna, _ = model.NaiveBiLSTM(emb)
            Q = model.scorer.Wq(mod_override.to(DEVICE))
            Kmat = model.scorer.Wk(rna)
            Vmat = model.scorer.Wv(rna)
            scale = model.scorer.tau * (Kmat.shape[-1] ** 0.5)
            logits = torch.einsum("kd,bld->bkl", Q, Kmat) / scale
            attn = logits.softmax(dim=-1)
            r = torch.einsum("bkl,bld->bkd", attn, Vmat)
            score = model.scorer.mlp(r).squeeze(-1)
            probs.append(torch.sigmoid(score).cpu().numpy())
    return np.concatenate(probs, 0)


def aucm_for(label, prob, mod_idx):
    return roc_auc_score(label[:, mod_idx], prob[:, mod_idx])


def cos(a, b):
    a = a.flatten(); b = b.flatten()
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


# ------------------------- E3 helper: per-position k-mers ------------------------
def get_test_center_seqs():
    """Recover the original 51bp sequences for the test split from the h5 file."""
    import pandas as pd
    frame = pd.read_hdf(DATA, "test_in_nucleo")
    arr = frame.iloc[:, 500-25:500+26].to_numpy(dtype=str)
    seqs = np.array(["".join(row).replace("T", "U") for row in arr])
    return seqs


def top_center_kmers(seqs, pos_mask, half_window=2):
    """Center base ± half_window k-mer at index 25, returns Counter of strings."""
    center = 25
    sub = [s[center-half_window:center+half_window+1] for s, m in zip(seqs, pos_mask) if m]
    return Counter(sub)


def pwm_ic(seqs, pos_mask, window=10):
    """Information content per position around center (25-w .. 25+w)."""
    center = 25
    keep = [s for s, m in zip(seqs, pos_mask) if m]
    if not keep:
        return None
    sub = np.array([list(s[center-window:center+window+1]) for s in keep])
    bases = ["A", "C", "G", "U"]
    L = sub.shape[1]
    ic = np.zeros(L)
    for pos in range(L):
        cnt = Counter(sub[:, pos])
        total = sum(cnt.values())
        h = 0.0
        for b in bases:
            p = cnt.get(b, 0) / total
            if p > 0:
                h -= p * np.log2(p)
        ic[pos] = 2.0 - h
    return ic


def main():
    print("=" * 88)
    print("v2 sharp-attn LOMO asymmetry diagnostic: m6A rescued, Am damaged")
    print("=" * 88)

    table = load_modification_table(MODCSV)
    chem_features = build_chemical_feature_matrix(table)

    test_data = read_split_as_kmers(DATA, "test", 51, EMB, CACHE)
    label = test_data["y"]
    loader = DataLoader(RmDataset(test_data), batch_size=128, shuffle=False)

    # Load all four models we need
    v2_m6a, _ = load_v2(ROOT / "chemical_v2_tau0.4_linear_morgan_r2_lomo" / "m6A", chem_features)
    v2_am, _ = load_v2(ROOT / "chemical_v2_tau0.4_linear_morgan_r2_lomo" / "Am", chem_features)
    v1_m6a, _ = load_v1(ROOT / "chemical_v1_bilinear_lomo" / "m6A", chem_features)
    v1_am, _ = load_v1(ROOT / "chemical_v1_bilinear_lomo" / "Am", chem_features)

    # =========================================================================
    # E4: encoder cos(mod_heldout, mod_m6Am) — v1 mlp vs v2 linear
    # =========================================================================
    print("\n[E4] Encoder cos(mod_heldout, mod_m6Am): v1 mlp vs v2 linear")
    with torch.no_grad():
        v1_m6a_mods = v1_m6a.chemical_encoder(v1_m6a.chemical_features).cpu()
        v1_am_mods = v1_am.chemical_encoder(v1_am.chemical_features).cpu()
        v2_m6a_mods = v2_m6a.chemical_encoder(v2_m6a.chemical_features).cpu()
        v2_am_mods = v2_am.chemical_encoder(v2_am.chemical_features).cpu()

    iM6A, iAm, iM6Am = IDX["m6A"], IDX["Am"], IDX["m6Am"]
    print(f"  v1 mlp LOMO m6A:  cos(mod_m6A,  mod_m6Am) = {cos(v1_m6a_mods[iM6A],  v1_m6a_mods[iM6Am]):+.3f}")
    print(f"  v1 mlp LOMO Am:   cos(mod_Am,   mod_m6Am) = {cos(v1_am_mods[iAm],   v1_am_mods[iM6Am]):+.3f}")
    print(f"  v2 lin LOMO m6A:  cos(mod_m6A,  mod_m6Am) = {cos(v2_m6a_mods[iM6A],  v2_m6a_mods[iM6Am]):+.3f}")
    print(f"  v2 lin LOMO Am:   cos(mod_Am,   mod_m6Am) = {cos(v2_am_mods[iAm],   v2_am_mods[iM6Am]):+.3f}")
    # Also cross-check: mod_Am vs other mods in v2 Am LOMO
    others = [n for n in MODIFICATION_NAMES if n != "Am"]
    cosines = sorted(
        [(n, cos(v2_am_mods[iAm], v2_am_mods[IDX[n]])) for n in others],
        key=lambda x: -x[1])
    print(f"\n  v2 lin LOMO Am: top-5 neighbours of mod_Am in encoder space:")
    for n, c in cosines[:5]:
        print(f"     {n:>5s}: cos = {c:+.3f}")
    print(f"\n  v2 lin LOMO m6A: top-5 neighbours of mod_m6A in encoder space:")
    others = [n for n in MODIFICATION_NAMES if n != "m6A"]
    cosines = sorted(
        [(n, cos(v2_m6a_mods[iM6A], v2_m6a_mods[IDX[n]])) for n in others],
        key=lambda x: -x[1])
    for n, c in cosines[:5]:
        print(f"     {n:>5s}: cos = {c:+.3f}")

    # =========================================================================
    # E1: Attention patterns — where do col[m6A], col[Am], col[m6Am] attend?
    # =========================================================================
    print("\n[E1] Attention pattern on v2 LOMO models")
    prob_m6a, attn_m6a = v2_forward_full(v2_m6a, loader)   # (1200,12,49)
    prob_am, attn_am = v2_forward_full(v2_am, loader)

    print(f"  Sanity AUCm: v2 LOMO m6A col[m6A]  = {aucm_for(label, prob_m6a, iM6A):.3f}  (expect ~0.584)")
    print(f"  Sanity AUCm: v2 LOMO Am  col[Am]   = {aucm_for(label, prob_am, iAm):.3f}   (expect ~0.361)")

    L = attn_m6a.shape[-1]  # 49
    center = L // 2  # 24

    def attn_profile(attn, pos_mask, col_idx):
        """Mean attention distribution over L for the col_idx column,
        averaged across samples in pos_mask."""
        a = attn[pos_mask][:, col_idx, :]  # (P, L)
        return a.mean(0), a.argmax(1)

    # v2 LOMO m6A: how does col[m6A] attend on m6A positives? col[m6Am] on m6A positives?
    pos_m6a = label[:, iM6A] == 1
    pos_am = label[:, iAm] == 1
    pos_m6am = label[:, iM6Am] == 1

    print(f"\n  v2 LOMO m6A model, attention on m6A POSITIVES (n={pos_m6a.sum()}):")
    for col_name in ["m6A", "m6Am", "m1A"]:
        ci = IDX[col_name]
        mean_a, argmax = attn_profile(attn_m6a, pos_m6a, ci)
        # peak position, mass within ±3, ±5 of center
        peak = int(mean_a.argmax())
        mass3 = float(mean_a[max(0, center-3): center+4].sum())
        mass5 = float(mean_a[max(0, center-5): center+6].sum())
        amax_at_center = float((np.abs(argmax - center) <= 3).mean())
        print(f"    col[{col_name:>5s}]: peak@pos {peak:2d}  ±3 mass={mass3:.3f}  ±5 mass={mass5:.3f}  "
              f"frac argmax in ±3 of center={amax_at_center:.2f}")

    # Compare attention distribution: col[m6A] vs col[m6Am] on the SAME m6A positives
    a_m6a, _ = attn_profile(attn_m6a, pos_m6a, iM6A)
    a_m6am_on_m6a_pos, _ = attn_profile(attn_m6a, pos_m6a, iM6Am)
    # cosine similarity & TVD between the two attention distributions
    cs = float(np.dot(a_m6a, a_m6am_on_m6a_pos) / (np.linalg.norm(a_m6a) * np.linalg.norm(a_m6am_on_m6a_pos) + 1e-12))
    tvd = float(0.5 * np.abs(a_m6a - a_m6am_on_m6a_pos).sum())
    print(f"    (col[m6A] vs col[m6Am]) attention profile cos={cs:.3f}  TVD={tvd:.3f}")

    print(f"\n  v2 LOMO Am model, attention on Am POSITIVES (n={pos_am.sum()}):")
    for col_name in ["Am", "m6Am", "Gm"]:
        ci = IDX[col_name]
        mean_a, argmax = attn_profile(attn_am, pos_am, ci)
        peak = int(mean_a.argmax())
        mass3 = float(mean_a[max(0, center-3): center+4].sum())
        mass5 = float(mean_a[max(0, center-5): center+6].sum())
        amax_at_center = float((np.abs(argmax - center) <= 3).mean())
        print(f"    col[{col_name:>5s}]: peak@pos {peak:2d}  ±3 mass={mass3:.3f}  ±5 mass={mass5:.3f}  "
              f"frac argmax in ±3 of center={amax_at_center:.2f}")
    a_am, _ = attn_profile(attn_am, pos_am, iAm)
    a_m6am_on_am_pos, _ = attn_profile(attn_am, pos_am, iM6Am)
    cs = float(np.dot(a_am, a_m6am_on_am_pos) / (np.linalg.norm(a_am) * np.linalg.norm(a_m6am_on_am_pos) + 1e-12))
    tvd = float(0.5 * np.abs(a_am - a_m6am_on_am_pos).sum())
    print(f"    (col[Am] vs col[m6Am]) attention profile cos={cs:.3f}  TVD={tvd:.3f}")

    # Print full mean attention vectors (positions 20-30 around center) for visual
    print(f"\n  v2 m6A LOMO mean attention on m6A POS, pos 18-30:")
    print(f"    col[m6A]:  " + " ".join(f"{x:.3f}" for x in a_m6a[18:31]))
    print(f"    col[m6Am]: " + " ".join(f"{x:.3f}" for x in a_m6am_on_m6a_pos[18:31]))
    print(f"  v2 Am  LOMO mean attention on Am  POS, pos 18-30:")
    print(f"    col[Am]:   " + " ".join(f"{x:.3f}" for x in a_am[18:31]))
    print(f"    col[m6Am]: " + " ".join(f"{x:.3f}" for x in a_m6am_on_am_pos[18:31]))

    # =========================================================================
    # E2: Mod-vector swap intervention — does Am behave at all?
    # =========================================================================
    print("\n[E2] Mod-vector swap intervention on v2 LOMO models")
    print("     We measure AUCm of col[heldout] when we REPLACE mod_heldout with other vectors.")
    print("     The fact that score_k = MLP(softmax(Wq mod_k . Wk rna_l)/tau . Wv rna_l)")
    print("     means changing only mod_k changes the routing for col[k] only.\n")

    def swap_test(model, mods, heldout_idx, heldout_name):
        donors = ["self"] + [n for n in MODIFICATION_NAMES if n != heldout_name][:11] + ["random_1", "random_2"]
        results = []
        for donor in donors:
            override = mods.clone()
            if donor == "self":
                pass
            elif donor.startswith("random"):
                seed = int(donor.split("_")[1])
                g = torch.Generator().manual_seed(seed)
                v = torch.randn(mods.shape[1], generator=g)
                v = v / v.norm() * mods[heldout_idx].norm()
                override[heldout_idx] = v
            else:
                # use donor's mod vector (after encoding)
                override[heldout_idx] = mods[IDX[donor]]
            prob = v2_forward_with_mod_override(model, loader, override)
            au = aucm_for(label, prob, heldout_idx)
            # Also collect mean prob on heldout positives
            pm = float(prob[label[:, heldout_idx] == 1, heldout_idx].mean())
            results.append((donor, au, pm))
        return results

    print(f"  v2 LOMO m6A: swap mod_m6A with other donors, read AUCm of col[m6A]")
    print(f"  {'donor':<12s}  AUCm   pos-mean")
    for donor, au, pm in swap_test(v2_m6a, v2_m6a_mods, iM6A, "m6A"):
        marker = "  <-- baseline" if donor == "self" else ""
        print(f"    {donor:<10s}  {au:.3f}   {pm:.4f}{marker}")

    print(f"\n  v2 LOMO Am: swap mod_Am with other donors, read AUCm of col[Am]")
    print(f"  {'donor':<12s}  AUCm   pos-mean")
    for donor, au, pm in swap_test(v2_am, v2_am_mods, iAm, "Am"):
        marker = "  <-- baseline" if donor == "self" else ""
        print(f"    {donor:<10s}  {au:.3f}   {pm:.4f}{marker}")

    # =========================================================================
    # E3: motif analysis on RNA test positives
    # =========================================================================
    print("\n[E3] RNA sequence motif around center for test positives")
    seqs = get_test_center_seqs()
    print(f"  loaded {len(seqs)} test sequences, length {len(seqs[0])}")

    for mod in ["m6A", "Am", "m6Am"]:
        ci = IDX[mod]
        mask = label[:, ci] == 1
        print(f"\n  {mod} positives (n={mask.sum()}):")
        center_base_counts = Counter([s[25] for s, m in zip(seqs, mask) if m])
        print(f"    center base distribution: {dict(center_base_counts)}")
        # Center 5-mer (pos 23..27)
        kc = top_center_kmers(seqs, mask, half_window=2)
        print(f"    top center 5-mers (pos 23-27, top 8):")
        for kmer, n in kc.most_common(8):
            print(f"      {kmer}  {n}")
        # IC over ±5
        ic = pwm_ic(seqs, mask, window=5)
        if ic is not None:
            print(f"    info-content positions 20..30 (max=2 bits/base):")
            print(f"      " + " ".join(f"{v:.2f}" for v in ic))
            print(f"      total IC = {ic.sum():.2f} bits")
        # DRACH check (DRACH = [AGT][AG]AC[ACT], center A)
        # positions: center -2..+2 -> 23..27
        drach_hits = 0
        for s, m in zip(seqs, mask):
            if not m:
                continue
            sub = s[23:28]
            if (sub[0] in "AGU" and sub[1] in "AG" and sub[2] == "A"
                    and sub[3] == "C" and sub[4] in "ACU"):
                drach_hits += 1
        print(f"    DRACH-like fraction at center 5-mer: {drach_hits}/{int(mask.sum())} = "
              f"{drach_hits/max(1,mask.sum()):.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()

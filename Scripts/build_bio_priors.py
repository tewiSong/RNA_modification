"""Build biology priors for 12 RNA modifications.

Output: Data/bio_priors.pkl with shape (12, D_bio) feature matrix where
each row corresponds to a modification in the canonical MODIFICATION_NAMES order.

Three feature blocks per modification:
  1. Writer enzyme indicator (23 dim).
  2. Region distribution over [5cap_TSS, cap_adjacent, 5UTR_other, CDS, 3UTR] (5 dim).
  3. 11-nt sequence motif PWM centred on the modification site (11 * 4 = 44 dim,
     row-normalised per position).

All values are hand-curated from peer-reviewed literature. No public database
is queried at runtime since the priors do not change per training run.

References for sourced values:
  m6A   - Dominissini 2012, Linder 2015 (DRACH consensus, 3UTR/CDS enrichment),
          Liu 2014 (METTL3-METTL14), Warda 2017 (METTL16).
  m6Am  - Sendinc 2019, Boulias 2019 (PCIF1, TSS+1 exclusive, BCA flank).
  Am    - Werner 2011 (CMTR1, CMTR2 cap), Ringeard 2019 (FTSJ3 internal).
  Cm/Gm/Um - Krogh 2016, Birkedal 2015 (Fibrillarin/NOP1 snoRNA-guided + CMTR).
  m1A   - Dominissini 2016, Safra 2017 (TRMT6/61A, 5UTR enrichment).
  m5C   - Yang 2017 (NSUN2 mRNA), Goll 2006 (DNMT2 tRNA).
  m5U   - Powell 2020 (TRMT2A tRNA, m5U mRNA writers less characterised).
  m7G   - Pandolfini 2019, Zhang 2019 (METTL1-WDR4 internal m7G).
  Psi   - Carlile 2014, Schwartz 2014 (PUS family, UNUAR motif for PUS7).
  I     - Bass 2002 (ADAR1/2, dsRNA substrate).
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from v0_data import MODIFICATION_NAMES


# ---------- (1) Writer enzymes ----------

WRITERS = [
    "METTL3", "METTL14", "METTL16",   # m6A family
    "PCIF1",                           # m6Am
    "CMTR1", "CMTR2",                  # cap 2'-O methyl
    "FTSJ3",                           # internal 2'-O methyl
    "NOP1",                            # snoRNA-guided 2'-O methyl (Fibrillarin)
    "TRMT6", "TRMT61A",                # m1A mRNA
    "TRMT10C",                         # m1A mt
    "NSUN2", "DNMT2",                  # m5C
    "TRMT2A",                          # m5U
    "METTL1", "WDR4",                  # m7G internal
    "RNGTT",                           # cap m7G
    "PUS1", "PUS7", "PUS10", "DKC1",   # pseudouridylation
    "ADAR1", "ADAR2",                  # A-to-I
]
WRITER_DIM = len(WRITERS)


def writer_vector(enzyme_weights):
    """Build a sparse writer indicator vector from {enzyme_name: weight} dict."""
    vec = np.zeros(WRITER_DIM, dtype=np.float32)
    for name, weight in enzyme_weights.items():
        if name not in WRITERS:
            raise ValueError(f"unknown writer {name}")
        vec[WRITERS.index(name)] = float(weight)
    return vec


WRITER_PER_MOD = {
    "Am":   {"CMTR1": 0.5, "FTSJ3": 0.5},
    "Cm":   {"NOP1": 0.6, "CMTR1": 0.2, "CMTR2": 0.2},
    "Gm":   {"NOP1": 0.6, "FTSJ3": 0.4},
    "Um":   {"NOP1": 0.6, "CMTR1": 0.2, "CMTR2": 0.2},
    "m1A":  {"TRMT6": 0.5, "TRMT61A": 0.5},
    "m5C":  {"NSUN2": 0.8, "DNMT2": 0.2},
    "m5U":  {"TRMT2A": 1.0},
    "m6A":  {"METTL3": 0.5, "METTL14": 0.4, "METTL16": 0.1},
    "m6Am": {"PCIF1": 1.0},
    "m7G":  {"METTL1": 0.5, "WDR4": 0.5},
    "Psi":  {"PUS1": 0.25, "PUS7": 0.25, "PUS10": 0.25, "DKC1": 0.25},
    "I":    {"ADAR1": 0.7, "ADAR2": 0.3},
}


# ---------- (2) Region distribution ----------
# Bins: cap_TSS (position 1, immediately after m7G cap),
#       cap_adjacent (positions 2..10),
#       5UTR_other (positions 11..end_of_5UTR),
#       CDS,
#       3UTR.

REGION_BINS = ["cap_TSS", "cap_adjacent", "5UTR_other", "CDS", "3UTR"]
REGION_DIM = len(REGION_BINS)


REGION_PER_MOD = {
    # m6A: Dominissini 2012, Meyer 2012 - 3UTR + CDS heavy, near stop codon.
    "m6A":  np.array([0.00, 0.01, 0.04, 0.55, 0.40], dtype=np.float32),
    # m6Am: Sendinc 2019 - exclusively TSS+1.
    "m6Am": np.array([0.95, 0.03, 0.02, 0.00, 0.00], dtype=np.float32),
    # Am: mixture of CMTR1 cap-1 (~30%) and FTSJ3 internal (~70%, spread).
    "Am":   np.array([0.05, 0.25, 0.10, 0.35, 0.25], dtype=np.float32),
    # Cm: snoRNA-guided internal + cap.
    "Cm":   np.array([0.05, 0.15, 0.10, 0.40, 0.30], dtype=np.float32),
    # Gm: similar pattern, mostly internal.
    "Gm":   np.array([0.02, 0.10, 0.08, 0.50, 0.30], dtype=np.float32),
    # Um: similar to other 2'-O-methyl, mostly internal.
    "Um":   np.array([0.03, 0.12, 0.10, 0.45, 0.30], dtype=np.float32),
    # m1A: Safra 2017 - enriched in 5UTR and first 100 nt of CDS.
    "m1A":  np.array([0.05, 0.10, 0.45, 0.30, 0.10], dtype=np.float32),
    # m5C: Yang 2017 - broad, slight 3UTR enrichment.
    "m5C":  np.array([0.00, 0.05, 0.15, 0.40, 0.40], dtype=np.float32),
    # m5U: less characterised in mRNA; tRNA T-loop preference - assume diffuse.
    "m5U":  np.array([0.00, 0.05, 0.15, 0.45, 0.35], dtype=np.float32),
    # m7G (internal): Pandolfini 2019 - CDS-heavy.
    "m7G":  np.array([0.00, 0.02, 0.08, 0.60, 0.30], dtype=np.float32),
    # Psi: Carlile 2014 - throughout mRNA, CDS-enriched.
    "Psi":  np.array([0.00, 0.05, 0.15, 0.45, 0.35], dtype=np.float32),
    # I (A-to-I): mostly in 3UTR and introns (in mature mRNA: 3UTR).
    "I":    np.array([0.00, 0.05, 0.10, 0.20, 0.65], dtype=np.float32),
}


# ---------- (3) Sequence motif PWM ----------
# 11-nt window centred on the modification site (position 5 = the modified
# nucleotide itself). Each position is a 4-dim base probability over (A,C,G,U).
# A flat prior (uniform 0.25 per base) indicates no strong consensus.

BASES_ORDER = ["A", "C", "G", "U"]
WINDOW_LEN = 11
CENTER_POS = 5


def flat_pwm():
    return np.full((WINDOW_LEN, 4), 0.25, dtype=np.float32)


def base_onehot(b):
    out = np.zeros(4, dtype=np.float32)
    out[BASES_ORDER.index(b)] = 1.0
    return out


def smoothed(probs, eps=0.05):
    """Smooth a 4-dim base preference toward uniform by eps."""
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / max(probs.sum(), 1e-6)
    return (1.0 - eps) * probs + eps * 0.25


def make_pwm(center_base, flanks):
    """Build an 11-nt PWM. center_base sets position 5. flanks is a dict of
    {offset_from_center: {base: prob}} for non-flat positions; missing offsets
    are uniform.
    """
    pwm = flat_pwm()
    pwm[CENTER_POS] = base_onehot(center_base)
    for offset, base_probs in flanks.items():
        idx = CENTER_POS + offset
        probs = np.zeros(4, dtype=np.float32)
        for base, p in base_probs.items():
            probs[BASES_ORDER.index(base)] = p
        pwm[idx] = smoothed(probs)
    return pwm


MOTIF_PER_MOD = {
    # m6A: DRACH = [A/G/U][A/G]_A_C_[A/C/U]. Centre is A.
    # Position -2: D = {A:0.33, G:0.33, U:0.34}.
    # Position -1: R = {A:0.5, G:0.5}.
    # Position +1: C dominant.
    # Position +2: H = {A:0.33, C:0.33, U:0.34}.
    "m6A":  make_pwm("A", {
        -2: {"A": 0.33, "G": 0.33, "U": 0.34},
        -1: {"A": 0.50, "G": 0.50},
        +1: {"C": 0.85, "A": 0.10, "U": 0.05},
        +2: {"A": 0.33, "C": 0.33, "U": 0.34},
    }),
    # m6Am: BCA at TSS+1. B = {C,G,U}. Centre is A. Strong cap-context.
    # Position -1 is the cap; encode as flat (cap is m7G not a standard base).
    # Position +1: B = {C:0.33, G:0.33, U:0.34}.
    # Position +2: weak preference toward C.
    "m6Am": make_pwm("A", {
        +1: {"C": 0.40, "G": 0.30, "U": 0.30},
        +2: {"C": 0.50, "A": 0.20, "G": 0.15, "U": 0.15},
    }),
    # Am (internal + cap-1): no strong consensus from FTSJ3 internal sites;
    # cap-1 Am has only "first transcribed A" identity. Flat motif except centre.
    "Am":   make_pwm("A", {}),
    # Cm: snoRNA-guided, sequence-dependent on snoRNA antisense. Broad. Centre C.
    "Cm":   make_pwm("C", {}),
    # Gm: similar - broad. Centre G.
    "Gm":   make_pwm("G", {}),
    # Um: broad. Centre U.
    "Um":   make_pwm("U", {}),
    # m1A: Safra 2017 reports GUUC motif from tRNA-like context; mRNA m1A has
    # weaker consensus. Centre A with mild upstream G.
    "m1A":  make_pwm("A", {
        -1: {"U": 0.50, "C": 0.20, "A": 0.20, "G": 0.10},
        -2: {"G": 0.40, "U": 0.30, "A": 0.20, "C": 0.10},
    }),
    # m5C: NSUN2 mRNA motif - mild context (AGCAGAGCC reported). Centre C.
    "m5C":  make_pwm("C", {
        -1: {"G": 0.40, "A": 0.30, "C": 0.20, "U": 0.10},
        +1: {"G": 0.35, "A": 0.25, "C": 0.20, "U": 0.20},
    }),
    # m5U: weak mRNA consensus. Centre U.
    "m5U":  make_pwm("U", {}),
    # m7G (internal): METTL1-WDR4 prefers structured contexts; sequence-level
    # consensus less specific. Centre G.
    "m7G":  make_pwm("G", {}),
    # Psi: PUS7 motif UNUAR. Centre U.
    "Psi":  make_pwm("U", {
        -2: {"U": 0.50, "A": 0.20, "C": 0.15, "G": 0.15},
        +1: {"A": 0.50, "U": 0.20, "C": 0.15, "G": 0.15},
        +2: {"A": 0.40, "G": 0.40, "C": 0.10, "U": 0.10},
    }),
    # I (A-to-I): ADAR prefers A in dsRNA, weak sequence consensus with -1 U/A
    # depletion bias. Centre is A (substrate before deamination).
    "I":    make_pwm("A", {
        -1: {"C": 0.40, "A": 0.25, "U": 0.20, "G": 0.15},
        +1: {"G": 0.40, "A": 0.25, "U": 0.20, "C": 0.15},
    }),
}


# ---------- Build the full matrix ----------
# Each block is encoded as DEVIATION FROM UNIFORM and L2-normalised. Without
# this, flat motif PWMs and overlapping region distributions push pairwise
# cosines to >0.6 across the board because the uniform component dominates
# the dot product. By centring on uniform and L2-normalising each block,
# only the discriminative deviations contribute to similarity.

def normalise_block(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x


def build():
    """Exclude the centre position from the motif block. The centre base is
    fully determined by the modification chemistry (already encoded in
    chem-side base one-hot), and including it gives flat-motif modifications
    (Am, Cm, Gm, Um, m5U, m7G) a degenerate one-hot motif vector that drives
    spurious cosine similarity with any other same-centre-base modification.
    """
    rows = []
    flank_positions = [p for p in range(WINDOW_LEN) if p != CENTER_POS]
    for mod in MODIFICATION_NAMES:
        w = writer_vector(WRITER_PER_MOD[mod])
        w = normalise_block(w)

        r = REGION_PER_MOD[mod] - (1.0 / REGION_DIM)
        r = normalise_block(r)

        m_full = MOTIF_PER_MOD[mod] - 0.25
        m_flanks = m_full[flank_positions].reshape(-1)  # (10 * 4) = 40 dim
        m = normalise_block(m_flanks)

        row = np.concatenate([w, r, m])
        rows.append(row)
    return np.stack(rows).astype(np.float32)


def cosine_matrix(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norm
    return Xn @ Xn.T


def main():
    bio = build()
    assert bio.shape == (12, WRITER_DIM + REGION_DIM + 40), bio.shape
    print(f"bio_priors shape: {bio.shape}  (writer={WRITER_DIM}, region={REGION_DIM}, motif_flanks=40)")

    # Raw PWMs for per-sample bio matching path. Stored separately from the
    # L2-normalised feature_matrix so the matching path can compute exact
    # log-likelihood ratios.
    raw_pwms = np.stack([MOTIF_PER_MOD[m] for m in MODIFICATION_NAMES], axis=0)
    assert raw_pwms.shape == (12, WINDOW_LEN, 4), raw_pwms.shape

    # Chemistry Tanimoto similarity matrix (Morgan FP r=2 fpSize=2048).
    # Fixed, computed from SMILES only. Used as cross-mod weight in the
    # biomatch path; LOMO-safe because no training is involved.
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    import pandas as pd
    mod_table = pd.read_csv("/ibex/user/songt/MultiRM/Data/modifications.csv")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(row["canonical_smiles"]))
           for _, row in mod_table.iterrows()]
    K = len(fps)
    tanimoto = np.zeros((K, K), dtype=np.float32)
    for i in range(K):
        for j in range(K):
            tanimoto[i, j] = float(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    out_path = Path("/ibex/user/songt/MultiRM/Data/bio_priors.pkl")
    with out_path.open("wb") as h:
        pickle.dump({
            "feature_matrix": bio,
            "modification_names": MODIFICATION_NAMES,
            "writer_names": WRITERS,
            "region_bins": REGION_BINS,
            "writer_dim": WRITER_DIM,
            "region_dim": REGION_DIM,
            "motif_dim": 40,
            "motif_window": WINDOW_LEN,
            "motif_center_pos": CENTER_POS,
            "motif_excludes_center": True,
            "raw_pwms": raw_pwms.astype(np.float32),
            "bases_order": BASES_ORDER,
            "tanimoto_matrix": tanimoto,
        }, h)
    print(f"wrote {out_path}")

    # Show pairwise cosine matrix
    C = cosine_matrix(bio)
    print("\nbio prior cosine matrix:")
    header = "        " + " ".join(f"{n:>6s}" for n in MODIFICATION_NAMES)
    print(header)
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"  {n:>6s} " + " ".join(f"{C[i,j]:+6.3f}" for j in range(12)))


if __name__ == "__main__":
    main()

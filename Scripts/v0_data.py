import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from torch.utils.data import Dataset


MODIFICATION_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Psi", "I"]
H5_LABELS = ["hAm", "hCm", "hGm", "hTm", "hm1A", "hm5C", "hm5U", "hm6A", "hm6Am", "hm7G", "hPsi", "Atol"]
ORIGINAL_BASES = ["A", "C", "G", "U", "A", "C", "U", "A", "A", "G", "U", "A"]
BASES = ["A", "C", "G", "T"]
CENTER_INDEX_1001 = 500
WINDOW_LENGTH = 51
CROP_START = 475
CROP_END = 526
CENTER_INDEX_51 = 25


def inspect_h5(path):
    rows = []
    with h5py.File(path, "r") as handle:
        def visitor(name, obj):
            if hasattr(obj, "shape"):
                rows.append((name, tuple(obj.shape), str(obj.dtype)))
            else:
                rows.append((name, "group", ""))

        handle.visititems(visitor)
    return rows


def canonical_base(base):
    if base == "U":
        return "T"
    return base


def modification_base_classes():
    return np.array([canonical_base(base) for base in ORIGINAL_BASES])


def one_hot_encode_nt_array(nt_array):
    encoded = np.zeros((nt_array.shape[0], nt_array.shape[1], len(BASES)), dtype=np.float32)
    encoded[:, :, 0] = nt_array == "A"
    encoded[:, :, 1] = nt_array == "C"
    encoded[:, :, 2] = nt_array == "G"
    encoded[:, :, 3] = (nt_array == "T") | (nt_array == "U")
    return encoded


def build_compatibility_mask(center_bases):
    center_classes = np.array([canonical_base(base) for base in center_bases])
    mod_base_classes = modification_base_classes()
    return center_classes[:, None] == mod_base_classes[None, :]


def read_multirm_split(data_path, split_name, max_samples=None):
    input_key = f"{split_name}_in_nucleo"
    output_key = f"{split_name}_out"
    input_frame = pd.read_hdf(data_path, input_key, start=0, stop=max_samples)
    output_frame = pd.read_hdf(data_path, output_key, start=0, stop=max_samples)

    assert list(output_frame.columns) == H5_LABELS
    assert input_frame.shape[1] == 1001

    cropped = input_frame.iloc[:, CROP_START:CROP_END].to_numpy(dtype=str)
    labels = output_frame.to_numpy(dtype=np.float32).copy()

    assert cropped.shape[1] == WINDOW_LENGTH
    center_bases = cropped[:, CENTER_INDEX_51].astype(str)
    one_hot = one_hot_encode_nt_array(cropped)
    compatibility_mask = build_compatibility_mask(center_bases)

    invalid_positive_count = int(((labels > 0.5) & (~compatibility_mask)).sum())
    assert invalid_positive_count == 0

    return {
        "x": one_hot,
        "y": labels,
        "compatibility_mask": compatibility_mask.astype(np.float32),
        "center_bases": np.array([canonical_base(base) for base in center_bases]),
    }


class MultiRMSplitDataset(Dataset):
    def __init__(self, split_data):
        self.x = torch.from_numpy(split_data["x"])
        self.y = torch.from_numpy(split_data["y"])
        self.compatibility_mask = torch.from_numpy(split_data["compatibility_mask"])
        self.center_bases = split_data["center_bases"]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.compatibility_mask[index]


def load_modification_table(path):
    table = pd.read_csv(path)
    assert list(table["name"]) == MODIFICATION_NAMES
    assert list(table["h5_label"]) == H5_LABELS
    assert list(table["original_base"]) == ORIGINAL_BASES

    for _, row in table.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        assert mol is not None
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        assert canonical_smiles == row["canonical_smiles"]

    return table


SITE_LABELS = [
    "methyl_N1_base",
    "methyl_N6_base",
    "methyl_N7_base",
    "methyl_C5_base",
    "methyl_2Oribose",
    "C5_glycosidic_isomer",
    "deamination_base",
]
SITES_PER_MOD = {
    "Am":   ["methyl_2Oribose"],
    "Cm":   ["methyl_2Oribose"],
    "Gm":   ["methyl_2Oribose"],
    "Um":   ["methyl_2Oribose"],
    "m1A":  ["methyl_N1_base"],
    "m5C":  ["methyl_C5_base"],
    "m5U":  ["methyl_C5_base"],
    "m6A":  ["methyl_N6_base"],
    "m6Am": ["methyl_N6_base", "methyl_2Oribose"],
    "m7G":  ["methyl_N7_base"],
    "Psi":  ["C5_glycosidic_isomer"],
    "I":    ["deamination_base"],
}


def site_features_for(mod_name):
    sites = SITES_PER_MOD[mod_name]
    return np.array([1.0 if label in sites else 0.0 for label in SITE_LABELS], dtype=np.float32)


def _make_fp_generator(fp_kind, n_bits):
    if fp_kind in ("morgan_r2", "morgan_r2_count"):
        return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    if fp_kind in ("morgan_r4", "morgan_r4_count"):
        return rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=n_bits)
    if fp_kind == "atom_pair":
        return rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    raise ValueError(f"unknown fp_kind={fp_kind}")


def _load_bio_prior_block(modification_table):
    """Load the per-modification biology prior block from Data/bio_priors.pkl.

    Returns a (K, D_bio) float32 matrix aligned to modification_table["name"].
    The bio prior file is expected to have already been L2-normalised per
    block (writer / region / motif) before concatenation; this function does
    not re-normalise.
    """
    import os
    import pickle
    path = os.path.join(os.path.dirname(__file__), "..", "Data", "bio_priors.pkl")
    pack = pickle.load(open(path, "rb"))
    pack_names = list(pack["modification_names"])
    table_names = list(modification_table["name"])
    if pack_names != table_names:
        raise ValueError(f"bio_priors order mismatch: {pack_names} vs {table_names}")
    return pack["feature_matrix"].astype(np.float32)


def load_per_sample_metadata(split_name, metadata_dir="/ibex/user/songt/MultiRM/Data/metadata"):
    """Load Path-A per-sample metadata: mature-mRNA region one-hot + log cap distance.

    Definition of 'mapped' here is exon-mapped (cap_distance >= 0). Samples whose
    aligned centre lands in an intron / intergenic region are NOT mature-mRNA
    positions; they carry no cap distance or mature-mRNA region, so they get
    mapped=0 and contribute nothing to the Path-A match.

    region_onehot is over five mature-mRNA bins only: cap_TSS, cap_adjacent,
    5UTR_other, CDS, 3UTR. Intron is NOT a sixth bin; intron samples have
    mapped=0 and all-zero region_onehot. noncoding-exon samples fold to
    5UTR_other (closest mature-mRNA position; alternative is a separate
    noncoding bin, but bio_priors does not provide one).

    Returns dict with:
      'region_onehot' (N, 5): one-hot for mature-mRNA-position samples;
          all-zero where mapped=0.
      'log_cap_distance' (N,): log1p(cap_distance) where mapped=1; 0 otherwise.
      'mapped' (N,): 1.0 if the centre is in an exon (cap_distance>=0); else 0.
    """
    import os
    import numpy as np
    path = os.path.join(metadata_dir, f"{split_name}_metadata.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Per-sample metadata not built yet: {path}")
    data = np.load(path, allow_pickle=True)
    region_labels = list(data["region_labels"])
    region_idx = data["region_idx"]
    cap_distance = data["cap_distance"]
    N = len(region_idx)
    mapped = (cap_distance >= 0).astype(np.float32)
    region_onehot = np.zeros((N, 5), dtype=np.float32)
    log_cap = np.zeros(N, dtype=np.float32)
    for i in range(N):
        if mapped[i] < 0.5:
            continue
        cd = int(cap_distance[i])
        if 0 <= cd <= 2:
            region_onehot[i, 0] = 1.0  # cap_TSS
        elif 3 <= cd <= 10:
            region_onehot[i, 1] = 1.0  # cap_adjacent
        else:
            r = region_labels[region_idx[i]]
            if r == "CDS":
                region_onehot[i, 3] = 1.0
            elif r == "3UTR":
                region_onehot[i, 4] = 1.0
            elif r == "5UTR":
                region_onehot[i, 2] = 1.0  # 5UTR_other
            elif r == "noncoding":
                region_onehot[i, 2] = 1.0  # fold noncoding exon into 5UTR_other
            # intron / intergenic / unknown cannot reach this branch:
            # mapped=1 implies cap_distance>=0 which only happens when the
            # mature-mRNA position is defined inside an exon.
        log_cap[i] = np.log1p(float(cd))
    return {
        "region_onehot": region_onehot,
        "log_cap_distance": log_cap,
        "mapped": mapped,
    }


def build_chemical_feature_matrix(
    modification_table,
    n_bits=2048,
    site_weight=0.0,
    fp_kind="morgan_r2",
    bio_weight=0.0,
):
    """Build per-modification chemistry feature matrix.

    fp_kind options:
      - "morgan_r2"        : default; Morgan r=2 binary (Tani(m6A,m6Am)=0.78).
      - "morgan_r4_count"  : Morgan r=4 count FP (Tani(m6A,m6Am)=0.62; best
                              symmetric FP we measured, still > 0.5).
      - "atom_pair"        : atom-pair binary FP (Tani(m6A,m6Am)=0.78).
      - "discriminative"   : orthogonal mod one-hot (12 dims) replacing the
                              fingerprint block. cos(m6A,m6Am)=0 by
                              construction; sacrifices chemistry-similarity
                              priors used for LOMO transfer.
    """
    use_count = fp_kind.endswith("_count")
    use_discriminative = fp_kind == "discriminative"
    if not use_discriminative:
        fp_gen = _make_fp_generator(fp_kind, n_bits)

    fingerprint_features = []
    descriptor_features = []
    base_features = []
    site_block = []
    for idx, (_, row) in enumerate(modification_table.iterrows()):
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        assert mol is not None

        if use_discriminative:
            fp_arr = np.zeros((len(modification_table),), dtype=np.float32)
            fp_arr[idx] = 1.0
        elif use_count:
            cfp = fp_gen.GetCountFingerprint(mol)
            fp_arr = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(cfp, fp_arr)
        else:
            bfp = fp_gen.GetFingerprint(mol)
            fp_arr = np.zeros((n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(bfp, fp_arr)
        fingerprint_features.append(fp_arr)

        formal_charge = Chem.GetFormalCharge(mol)
        descriptor_features.append([
            Descriptors.MolWt(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            formal_charge,
        ])

        base = row["original_base"]
        base_features.append([
            1.0 if base == "A" else 0.0,
            1.0 if base == "C" else 0.0,
            1.0 if base == "G" else 0.0,
            1.0 if base == "U" else 0.0,
        ])

        site_block.append(site_features_for(row["name"]))

    fingerprints = np.stack(fingerprint_features).astype(np.float32)
    descriptors = np.array(descriptor_features, dtype=np.float32)
    descriptor_mean = descriptors.mean(axis=0, keepdims=True)
    descriptor_std = descriptors.std(axis=0, keepdims=True)
    descriptors = (descriptors - descriptor_mean) / descriptor_std
    bases = np.array(base_features, dtype=np.float32)
    blocks = [fingerprints, descriptors, bases]
    if site_weight > 0.0:
        sites = np.stack(site_block) * float(site_weight)
        blocks.append(sites)
    if bio_weight > 0.0:
        # Per-row L2-normalise the chemistry block before adding bio so the two
        # blocks contribute on comparable scales. This matches the gate
        # computation in gate_s1_5.py and produces the same cosine geometry.
        chem_only = np.concatenate(blocks, axis=1).astype(np.float32)
        chem_norm = np.linalg.norm(chem_only, axis=1, keepdims=True) + 1e-12
        chem_n = chem_only / chem_norm
        bio = _load_bio_prior_block(modification_table) * float(bio_weight)
        return np.concatenate([chem_n, bio], axis=1).astype(np.float32)
    return np.concatenate(blocks, axis=1).astype(np.float32)

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


def build_chemical_feature_matrix(modification_table, n_bits=2048):
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fingerprint_features = []
    descriptor_features = []
    base_features = []
    for _, row in modification_table.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        assert mol is not None

        fingerprint = morgan_generator.GetFingerprint(mol)
        fingerprint_array = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        fingerprint_features.append(fingerprint_array)

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

    fingerprints = np.stack(fingerprint_features).astype(np.float32)
    descriptors = np.array(descriptor_features, dtype=np.float32)
    descriptor_mean = descriptors.mean(axis=0, keepdims=True)
    descriptor_std = descriptors.std(axis=0, keepdims=True)
    descriptors = (descriptors - descriptor_mean) / descriptor_std
    bases = np.array(base_features, dtype=np.float32)
    return np.concatenate([fingerprints, descriptors, bases], axis=1).astype(np.float32)

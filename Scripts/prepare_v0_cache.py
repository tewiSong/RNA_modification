import argparse
import json
from pathlib import Path

import numpy as np

from v0_data import inspect_h5, read_multirm_split


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare cached 51 nt MultiRM v0 splits")
    parser.add_argument("--data_path", default="Data/MultiRM_data.h5")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_valid_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def cache_path(cache_dir, split_name, max_samples):
    sample_tag = "full" if max_samples is None else str(max_samples)
    return Path(cache_dir) / f"{split_name}_{sample_tag}_51nt.npz"


def save_split(cache_dir, split_name, split_data, max_samples):
    output_path = cache_path(cache_dir, split_name, max_samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=split_data["x"],
        y=split_data["y"],
        compatibility_mask=split_data["compatibility_mask"],
        center_bases=split_data["center_bases"],
    )
    return output_path


def main():
    args = parse_args()
    print("HDF5 structure")
    for name, shape, dtype in inspect_h5(args.data_path):
        print(f"{name}\t{shape}\t{dtype}", flush=True)

    split_specs = [
        ("train", args.max_train_samples),
        ("valid", args.max_valid_samples),
        ("test", args.max_test_samples),
    ]
    metadata = {}
    for split_name, max_samples in split_specs:
        split_data = read_multirm_split(args.data_path, split_name, max_samples)
        output_path = save_split(args.cache_dir, split_name, split_data, max_samples)
        metadata[split_name] = {
            "path": str(output_path),
            "samples": int(split_data["x"].shape[0]),
            "window_length": int(split_data["x"].shape[1]),
            "channels": int(split_data["x"].shape[2]),
        }
        print(f"saved {split_name}: {output_path}", flush=True)

    metadata_path = Path(args.cache_dir) / "metadata.json"
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()

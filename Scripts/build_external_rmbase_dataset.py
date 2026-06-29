#!/usr/bin/env python3
"""Build an RMBase-derived external benchmark for MultiRM-style evaluation.

This script has two modes:

1. Metadata mode, without --reference_fasta:
   parse RMBase tar.gz files and write positive-site metadata plus a summary.

2. H5 mode, with --reference_fasta:
   fetch centered genomic windows, sample matched negatives, and write an H5
   containing test_in_nucleo/test_out keys compatible with the current code.

The H5 mode requires an uncompressed hg38 FASTA. Gzipped FASTA is deliberately
not supported for random access; decompress first if needed.
"""

import argparse
import csv
import io
import json
import random
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MODIFICATION_NAMES = ["Am", "Cm", "Gm", "Um", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Psi", "I"]
H5_LABELS = ["hAm", "hCm", "hGm", "hTm", "hm1A", "hm5C", "hm5U", "hm6A", "hm6Am", "hm7G", "hPsi", "Atol"]
ORIGINAL_BASES = ["A", "C", "G", "U", "A", "C", "U", "A", "A", "G", "U", "A"]
BASE_TO_DNA = {"A": "A", "C": "C", "G": "G", "U": "T", "T": "T"}
WINDOW_LENGTH = 1001
CENTER_INDEX = WINDOW_LENGTH // 2
RMBASE_COLUMNS = [
    "chrom",
    "start",
    "end",
    "mod_id",
    "score",
    "strand",
    "mod_type",
    "support_num",
    "support_list",
    "support_list_sub",
    "pub_list",
    "cell_list",
    "seq_type_list",
    "gene_id",
    "transcript_id",
    "gene_name",
    "gene_type",
    "region",
    "seq",
]


@dataclass(frozen=True)
class RmbaseSelection:
    modification: str
    source_mod_type: str
    accepted_mod_types: tuple


RMBASE_SELECTIONS = [
    RmbaseSelection("Am", "Nm", ("Am",)),
    RmbaseSelection("Cm", "Nm", ("Cm",)),
    RmbaseSelection("Gm", "Nm", ("Gm",)),
    RmbaseSelection("Um", "Nm", ("Um", "Tm")),
    RmbaseSelection("m1A", "m1A", ("m1A",)),
    RmbaseSelection("m5C", "m5C", ("m5C",)),
    RmbaseSelection("m5U", "otherMod", ("m5U", "m5U_site")),
    RmbaseSelection("m6A", "m6A", ("m6A",)),
    RmbaseSelection("m6Am", "otherMod", ("m6Am", "m6Am_site")),
    RmbaseSelection("m7G", "m7G", ("m7G",)),
    RmbaseSelection("Psi", "Pseudo", ("Pseudo", "Psi", "pseudouridine")),
    RmbaseSelection("I", "RNA-editing", ("I", "A-I", "A-to-I", "RNA-editing")),
]


def normalize_base(base):
    return BASE_TO_DNA.get(str(base).upper(), str(base).upper())


def reverse_complement(sequence):
    table = str.maketrans("ACGTUacgtu", "TGCAAtgcaa")
    return sequence.translate(table)[::-1].upper().replace("U", "T")


def clean_sequence(sequence):
    return str(sequence).upper().replace("U", "T")


def parse_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def rmbase_tar_path(raw_dir, assembly, source_mod_type):
    return Path(raw_dir) / f"{assembly}.{source_mod_type}.tar.gz"


def read_rmbase_tar(path):
    rows = []
    with tarfile.open(path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                continue
            text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace")
            reader = csv.reader(text, delimiter="\t")
            for raw in reader:
                if not raw or raw[0].startswith("#"):
                    continue
                if raw[0].lower() in {"chrom", "chr"}:
                    continue
                padded = raw + [""] * max(0, len(RMBASE_COLUMNS) - len(raw))
                row = dict(zip(RMBASE_COLUMNS, padded[: len(RMBASE_COLUMNS)]))
                row["raw_column_count"] = len(raw)
                rows.append(row)
    return rows


def select_positive_sites(raw_dir, assembly):
    selected = []
    missing_files = []
    source_cache = {}

    for selection in RMBASE_SELECTIONS:
        path = rmbase_tar_path(raw_dir, assembly, selection.source_mod_type)
        if not path.exists():
            missing_files.append(str(path))
            continue
        if path not in source_cache:
            source_cache[path] = read_rmbase_tar(path)
        accepted = {x.lower() for x in selection.accepted_mod_types}
        for row in source_cache[path]:
            mod_type = str(row.get("mod_type", "")).strip()
            if mod_type.lower() not in accepted and selection.source_mod_type.lower() != mod_type.lower():
                continue
            if selection.source_mod_type in {"Nm", "otherMod", "RNA-editing"} and mod_type.lower() not in accepted:
                continue
            start = parse_int(row.get("start"))
            end = parse_int(row.get("end"))
            if start is None or end is None:
                continue
            center0 = start if end <= start + 1 else (start + end) // 2
            sequence_41 = clean_sequence(row.get("seq", ""))
            selected.append(
                {
                    "modification": selection.modification,
                    "source_mod_type": selection.source_mod_type,
                    "chrom": row.get("chrom", ""),
                    "start": start,
                    "end": end,
                    "center0_assumed": center0,
                    "strand": row.get("strand", ""),
                    "mod_id": row.get("mod_id", ""),
                    "score": row.get("score", ""),
                    "support_num": row.get("support_num", ""),
                    "support_list": row.get("support_list", ""),
                    "support_list_sub": row.get("support_list_sub", ""),
                    "pub_list": row.get("pub_list", ""),
                    "cell_list": row.get("cell_list", ""),
                    "seq_type_list": row.get("seq_type_list", ""),
                    "gene_id": row.get("gene_id", ""),
                    "transcript_id": row.get("transcript_id", ""),
                    "gene_name": row.get("gene_name", ""),
                    "gene_type": row.get("gene_type", ""),
                    "region": row.get("region", ""),
                    "seq_41": sequence_41,
                    "raw_column_count": row.get("raw_column_count", ""),
                }
            )
    return pd.DataFrame(selected), missing_files


class IndexedFasta:
    def __init__(self, fasta_path):
        self.fasta_path = Path(fasta_path)
        if self.fasta_path.suffix == ".gz":
            raise ValueError("Use an uncompressed FASTA for random access; .gz is not supported.")
        self.index_path = self.fasta_path.with_suffix(self.fasta_path.suffix + ".fai")
        if not self.index_path.exists():
            self._build_index()
        self.records = self._read_index()
        self.handle = self.fasta_path.open("rb")

    def _build_index(self):
        records = []
        with self.fasta_path.open("rb") as handle:
            name = None
            length = 0
            seq_offset = None
            line_bases = None
            line_width = None
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.startswith(b">"):
                    if name is not None:
                        records.append((name, length, seq_offset, line_bases, line_width))
                    name = line[1:].decode("utf-8", errors="replace").strip().split()[0]
                    length = 0
                    seq_offset = None
                    line_bases = None
                    line_width = None
                    continue
                stripped = line.rstrip(b"\r\n")
                if seq_offset is None:
                    seq_offset = offset
                    line_bases = len(stripped)
                    line_width = len(line)
                length += len(stripped)
            if name is not None:
                records.append((name, length, seq_offset, line_bases, line_width))
        with self.index_path.open("w") as out:
            for record in records:
                out.write("\t".join(map(str, record)) + "\n")

    def _read_index(self):
        records = {}
        with self.index_path.open() as handle:
            for line in handle:
                name, length, offset, line_bases, line_width = line.rstrip("\n").split("\t")[:5]
                records[name] = {
                    "length": int(length),
                    "offset": int(offset),
                    "line_bases": int(line_bases),
                    "line_width": int(line_width),
                }
        return records

    def normalize_chrom(self, chrom):
        if chrom in self.records:
            return chrom
        if chrom.startswith("chr") and chrom[3:] in self.records:
            return chrom[3:]
        with_chr = "chr" + chrom
        if with_chr in self.records:
            return with_chr
        return chrom

    def fetch(self, chrom, start0, end0):
        chrom = self.normalize_chrom(str(chrom))
        if chrom not in self.records:
            raise KeyError(f"chromosome not in FASTA: {chrom}")
        rec = self.records[chrom]
        if start0 < 0 or end0 > rec["length"] or end0 <= start0:
            raise ValueError(f"invalid interval: {chrom}:{start0}-{end0}")
        chunks = []
        pos = start0
        while pos < end0:
            line_index = pos // rec["line_bases"]
            in_line = pos % rec["line_bases"]
            n = min(end0 - pos, rec["line_bases"] - in_line)
            byte_offset = rec["offset"] + line_index * rec["line_width"] + in_line
            self.handle.seek(byte_offset)
            chunks.append(self.handle.read(n))
            pos += n
        return b"".join(chunks).decode("ascii").upper()


def build_positive_examples(sites, fasta):
    examples = []
    rejects = Counter()
    grouped = defaultdict(set)
    metadata_by_key = {}

    for _, row in sites.iterrows():
        mod = row["modification"]
        mod_index = MODIFICATION_NAMES.index(mod)
        center = int(row["center0_assumed"])
        start = center - CENTER_INDEX
        end = center + CENTER_INDEX + 1
        try:
            seq = fasta.fetch(row["chrom"], start, end)
        except Exception as exc:
            rejects[f"fetch_failed:{type(exc).__name__}"] += 1
            continue
        seq = clean_sequence(seq)
        if len(seq) != WINDOW_LENGTH or any(base not in "ACGT" for base in seq):
            rejects["bad_sequence"] += 1
            continue
        if str(row.get("strand", "+")) == "-":
            seq = reverse_complement(seq)
        expected_base = normalize_base(ORIGINAL_BASES[mod_index])
        if seq[CENTER_INDEX] != expected_base:
            rejects[f"center_base_mismatch:{mod}"] += 1
            continue
        key = (row["chrom"], center, row.get("strand", "."), seq)
        grouped[key].add(mod)
        metadata_by_key.setdefault(key, row.to_dict())

    for key, mods in grouped.items():
        chrom, center, strand, seq = key
        labels = np.zeros(len(MODIFICATION_NAMES), dtype=np.float32)
        for mod in mods:
            labels[MODIFICATION_NAMES.index(mod)] = 1.0
        meta = metadata_by_key[key].copy()
        meta.update(
            {
                "example_type": "positive",
                "chrom": chrom,
                "center0_assumed": center,
                "strand": strand,
                "sequence": seq,
                "positive_modifications": ";".join(sorted(mods)),
            }
        )
        examples.append((seq, labels, meta))
    return examples, rejects


def sample_negative_examples(fasta, positive_examples, negative_ratio, seed):
    rng = random.Random(seed)
    positives_by_center = {(m["chrom"], int(m["center0_assumed"])) for _, _, m in positive_examples}
    chroms = [(name, rec["length"]) for name, rec in fasta.records.items() if rec["length"] > WINDOW_LENGTH + 2]
    total_len = sum(length for _, length in chroms)
    cumulative = []
    acc = 0
    for name, length in chroms:
        acc += length
        cumulative.append((acc, name, length))

    base_targets = Counter()
    for _, labels, _ in positive_examples:
        for idx, value in enumerate(labels):
            if value > 0.5:
                base_targets[normalize_base(ORIGINAL_BASES[idx])] += negative_ratio

    negatives = []
    rejects = Counter()
    for target_base, target_count in base_targets.items():
        attempts = 0
        while sum(1 for _, _, m in negatives if m["center_base"] == target_base) < target_count:
            attempts += 1
            if attempts > target_count * 1000 + 10000:
                rejects[f"negative_sampling_exhausted:{target_base}"] += 1
                break
            pick = rng.randrange(total_len)
            chrom = None
            chrom_len = None
            for cutoff, name, length in cumulative:
                if pick < cutoff:
                    chrom = name
                    chrom_len = length
                    break
            center = rng.randrange(CENTER_INDEX, chrom_len - CENTER_INDEX)
            if (chrom, center) in positives_by_center:
                rejects["negative_overlaps_positive"] += 1
                continue
            try:
                seq = clean_sequence(fasta.fetch(chrom, center - CENTER_INDEX, center + CENTER_INDEX + 1))
            except Exception:
                rejects["negative_fetch_failed"] += 1
                continue
            if len(seq) != WINDOW_LENGTH or any(base not in "ACGT" for base in seq):
                rejects["negative_bad_sequence"] += 1
                continue
            if seq[CENTER_INDEX] != target_base:
                rejects["negative_center_base_mismatch"] += 1
                continue
            labels = np.zeros(len(MODIFICATION_NAMES), dtype=np.float32)
            meta = {
                "example_type": "negative",
                "chrom": chrom,
                "center0_assumed": center,
                "strand": "+",
                "sequence": seq,
                "positive_modifications": "",
                "center_base": target_base,
            }
            negatives.append((seq, labels, meta))
    return negatives, rejects


def write_h5(examples, output_h5):
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    sequences = [seq for seq, _, _ in examples]
    labels = np.stack([labels for _, labels, _ in examples], axis=0)
    input_frame = pd.DataFrame([list(seq) for seq in sequences])
    output_frame = pd.DataFrame(labels, columns=H5_LABELS)
    input_frame.to_hdf(output_h5, key="test_in_nucleo", mode="w", format="table")
    output_frame.to_hdf(output_h5, key="test_out", mode="a", format="table")


def write_metadata(examples, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row_id, (_, labels, meta) in enumerate(examples):
        out = dict(meta)
        out["row_id"] = row_id
        for mod, label, value in zip(MODIFICATION_NAMES, H5_LABELS, labels):
            out[label] = int(value > 0.5)
            out[f"{mod}_label"] = int(value > 0.5)
        out.pop("sequence", None)
        rows.append(out)
    pd.DataFrame(rows).to_csv(output_csv, index=False)


def write_summary(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="Data/external_rmbase/raw/rmbase_v3")
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument("--reference_fasta", default=None)
    parser.add_argument("--output_h5", default="Data/external_rmbase/processed/external_rmbase_human.h5")
    parser.add_argument("--metadata_csv", default="Data/external_rmbase/processed/external_rmbase_human_metadata.csv")
    parser.add_argument("--positive_sites_csv", default="Data/external_rmbase/processed/external_rmbase_positive_sites.csv")
    parser.add_argument("--summary_json", default="Data/external_rmbase/processed/external_rmbase_build_summary.json")
    parser.add_argument("--negative_ratio", type=int, default=1)
    parser.add_argument("--max_positive_per_mod", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    sites, missing_files = select_positive_sites(args.raw_dir, args.assembly)
    Path(args.positive_sites_csv).parent.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.positive_sites_csv, index=False)
    sites_for_h5 = sites
    if args.max_positive_per_mod > 0 and len(sites) > 0:
        sampled = []
        for _, group in sites.groupby("modification", sort=False):
            if len(group) > args.max_positive_per_mod:
                sampled.append(group.sample(n=args.max_positive_per_mod, random_state=args.seed))
            else:
                sampled.append(group)
        sites_for_h5 = pd.concat(sampled, axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    summary = {
        "assembly": args.assembly,
        "raw_dir": args.raw_dir,
        "positive_site_rows": int(len(sites)),
        "positive_site_rows_for_h5": int(len(sites_for_h5)),
        "max_positive_per_mod": args.max_positive_per_mod,
        "positive_site_counts_by_modification": sites["modification"].value_counts().to_dict() if len(sites) else {},
        "positive_site_counts_for_h5_by_modification": sites_for_h5["modification"].value_counts().to_dict() if len(sites_for_h5) else {},
        "missing_files": missing_files,
        "h5_written": False,
    }

    if args.reference_fasta is None:
        summary["mode"] = "metadata_only"
        summary["reason_h5_not_written"] = "reference_fasta was not supplied"
        write_summary(args.summary_json, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    fasta = IndexedFasta(args.reference_fasta)
    positives, positive_rejects = build_positive_examples(sites_for_h5, fasta)
    negatives, negative_rejects = sample_negative_examples(fasta, positives, args.negative_ratio, args.seed)
    examples = positives + negatives
    write_h5(examples, args.output_h5)
    write_metadata(examples, args.metadata_csv)

    summary.update(
        {
            "mode": "h5",
            "reference_fasta": args.reference_fasta,
            "positive_examples": len(positives),
            "negative_examples": len(negatives),
            "positive_rejects": dict(positive_rejects),
            "negative_rejects": dict(negative_rejects),
            "output_h5": args.output_h5,
            "metadata_csv": args.metadata_csv,
            "h5_written": True,
        }
    )
    write_summary(args.summary_json, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

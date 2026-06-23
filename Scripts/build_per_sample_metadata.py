"""Path A: align 1001-nt windows to GRCh38, look up GENCODE annotations.

Avoids gffutils (which is too slow on human v44). Instead:
1. Parse the GENCODE GFF3 once into per-chromosome IntervalTrees (transcripts
   only, with their type and bounds), pickled for reuse.
2. For each transcript, also store its exon / CDS / UTR feature intervals in a
   nested IntervalTree, retrieved by transcript_id.
3. Align 1001-nt windows with mappy minimap2 bindings.
4. For each mapped window centre, query the chromosome IntervalTree to find
   overlapping transcripts, then walk into the per-transcript feature tree to
   classify the region (5'UTR / CDS / 3'UTR / intron / noncoding).

Output: per-split metadata.npz with columns: chrom, pos_center, strand,
region_idx, cap_distance, polyA_distance, transcript_id.
"""
import argparse
import gzip
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")


REGION_LABELS = ["unknown", "intergenic", "intron", "5UTR", "CDS", "3UTR", "noncoding"]


def label_to_idx(label):
    return REGION_LABELS.index(label)


# ---------- GFF parsing ----------

def parse_attributes(attr_field):
    """GFF3 attribute string -> dict."""
    out = {}
    for kv in attr_field.split(";"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def parse_gencode_gff(gff_path):
    """Single-pass parser. Returns:
    transcripts: dict[chrom] -> list of (start, end, strand, transcript_id, transcript_type)
    features: dict[transcript_id] -> list of (start, end, feature_type)
        feature_type ∈ {'CDS','five_prime_UTR','three_prime_UTR','exon'}
    """
    print(f"Parsing {gff_path}...", flush=True)
    t0 = time.time()
    transcripts = defaultdict(list)
    features = defaultdict(list)
    feature_kinds = {"CDS", "five_prime_UTR", "three_prime_UTR", "exon"}
    n_lines = 0
    n_tx = 0
    n_feat = 0
    opener = gzip.open if str(gff_path).endswith(".gz") else open
    with opener(gff_path, "rt") as h:
        for line in h:
            if line.startswith("#"):
                continue
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            chrom, _src, ftype, start, end, _score, strand, _frame, attrs = parts
            if ftype == "transcript":
                a = parse_attributes(attrs)
                tid = a.get("transcript_id")
                ttype = a.get("transcript_type", "unknown")
                if tid:
                    transcripts[chrom].append((int(start) - 1, int(end), strand, tid, ttype))
                    n_tx += 1
            elif ftype in feature_kinds:
                a = parse_attributes(attrs)
                tid = a.get("transcript_id")
                if tid:
                    features[tid].append((int(start) - 1, int(end), ftype))
                    n_feat += 1
            if n_lines % 500000 == 0:
                print(f"  parsed {n_lines} lines; {n_tx} transcripts; {n_feat} features", flush=True)
    print(f"  done parsing in {time.time()-t0:.1f}s; {n_tx} transcripts, {n_feat} features", flush=True)
    return dict(transcripts), dict(features)


def build_transcript_trees(transcripts):
    """Build per-chromosome IntervalTree of transcript intervals."""
    from intervaltree import IntervalTree, Interval
    print("Building per-chromosome IntervalTrees...", flush=True)
    t0 = time.time()
    trees = {}
    for chrom, lst in transcripts.items():
        intervals = [Interval(s, e, (s, e, strand, tid, ttype)) for (s, e, strand, tid, ttype) in lst]
        trees[chrom] = IntervalTree(intervals)
    print(f"  built {len(trees)} chromosome trees in {time.time()-t0:.1f}s", flush=True)
    return trees


# ---------- minimap2 alignment ----------

def load_split_sequences(data_path, split_name):
    df = pd.read_hdf(data_path, f"{split_name}_in_nucleo")
    arr = df.to_numpy(dtype=str)
    return ["".join(arr[i].tolist()).replace("U", "T") for i in range(arr.shape[0])]


def query_centre_to_ref_pos(hit, centre_in_query):
    """Walk minimap2 hit's CIGAR to find the genomic position that corresponds
    to query position `centre_in_query`. Handles splice-aware CIGAR with intron
    skips (op=3, 'N'). Returns None if the centre falls in a query-only span
    (insertion / soft-clip) and therefore has no ref position.

    mappy operation codes (see minimap2 docs):
      0 = M (match/mismatch; both advance)
      1 = I (insertion to ref; query advances)
      2 = D (deletion from ref; ref advances)
      3 = N (intron skip; ref advances)
      4 = S (soft clip; query advances)
      7 = = (exact match; both advance)
      8 = X (mismatch; both advance)
    """
    if not (hit.q_st <= centre_in_query < hit.q_en):
        return None
    cigar = hit.cigar  # list of (length, op)
    if hit.strand == 1:
        q_pos = hit.q_st
        r_pos = hit.r_st
        for length, op in cigar:
            if op in (0, 7, 8):
                if q_pos + length > centre_in_query:
                    return r_pos + (centre_in_query - q_pos)
                q_pos += length
                r_pos += length
            elif op in (1, 4):
                if q_pos + length > centre_in_query:
                    return None  # centre is inside an insertion to ref
                q_pos += length
            elif op in (2, 3):
                r_pos += length
            else:
                return None
        return None
    # Negative strand: mappy reports q_st/q_en in original (forward) query
    # coordinates; the CIGAR walks the *reverse-complemented* query against the
    # forward reference. So query offset from the end (q_en - 1 - centre) is
    # consumed left-to-right in the CIGAR; ref position counts up from r_st.
    q_consumed_from_end = (hit.q_en - 1) - centre_in_query  # >=0
    q_walk = 0
    r_pos = hit.r_st
    for length, op in cigar:
        if op in (0, 7, 8):
            if q_walk + length > q_consumed_from_end:
                return r_pos + (q_consumed_from_end - q_walk)
            q_walk += length
            r_pos += length
        elif op in (1, 4):
            if q_walk + length > q_consumed_from_end:
                return None
            q_walk += length
        elif op in (2, 3):
            r_pos += length
        else:
            return None
    return None


def align_windows(aligner, sequences, log_every=10000):
    centre_in_query = 500
    N = len(sequences)
    chrom = np.full(N, "", dtype=object)
    pos_center = np.full(N, -1, dtype=np.int64)
    strand = np.zeros(N, dtype=np.int8)
    mapq = np.zeros(N, dtype=np.int16)
    t0 = time.time()
    mapped = 0
    for i, seq in enumerate(sequences):
        try:
            hits = list(aligner.map(seq))
        except Exception:
            hits = []
        if not hits:
            continue
        best = hits[0]
        ref_centre = query_centre_to_ref_pos(best, centre_in_query)
        if ref_centre is None:
            continue
        chrom[i] = best.ctg
        pos_center[i] = ref_centre
        strand[i] = best.strand
        mapq[i] = best.mapq
        mapped += 1
        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  aligned {i+1}/{N} ({mapped} mapped) rate {rate:.0f}/s ETA {eta/60:.1f} min", flush=True)
    print(f"  done: {mapped}/{N} mapped ({100*mapped/N:.1f}%)", flush=True)
    return chrom, pos_center, strand, mapq


# ---------- annotation ----------

def mature_position(p, exons_sorted, t_strand):
    """Given genomic position p and a list of (start, end) exons sorted in
    transcription order (5' to 3' on the mRNA), return (mature_offset, total_mature_len, exon_idx).
    mature_offset = number of mature-mRNA nt 5' of p (0 means p is the very first base of the mRNA).
    Returns (-1, total_mature_len, -1) if p is not inside any exon.
    """
    total = sum(e - s for (s, e) in exons_sorted)
    cum = 0
    for idx, (s, e) in enumerate(exons_sorted):
        if s <= p < e:
            if t_strand == "+":
                offset_in_exon = p - s
            else:
                offset_in_exon = (e - 1) - p
            return cum + offset_in_exon, total, idx
        cum += (e - s)
    return -1, total, -1


def classify_position(p, transcripts_at_pos, features_by_tid):
    """Pick best overlapping transcript and classify p in mature-mRNA terms.

    Returns: (region_idx, cap_distance_mature, polya_distance_mature, transcript_id)
    where cap_distance_mature is the number of mature-mRNA nt 5' of p, walking
    the matched transcript's exons in transcription order.

    Selection rule: among all transcripts whose genomic span overlaps p, keep
    only those whose EXONS contain p (i.e. transcripts for which p is a
    mature-mRNA position, not an intron). Without this filter the longest
    overlapping transcript can have p in an intron even though splice-aware
    alignment placed the centre in an exon of a different (typically shorter)
    overlapping transcript.
    """
    if not transcripts_at_pos:
        return label_to_idx("intergenic"), -1, -1, ""
    transcripts_with_p_in_exon = []
    for t in transcripts_at_pos:
        _, _, _, tid, _ = t
        feats = features_by_tid.get(tid, [])
        if any(s <= p < e for (s, e, ft) in feats if ft == "exon"):
            transcripts_with_p_in_exon.append(t)
    if not transcripts_with_p_in_exon:
        return label_to_idx("intron"), -1, -1, ""
    pc = [t for t in transcripts_with_p_in_exon if t[4] == "protein_coding"]
    candidates = pc if pc else transcripts_with_p_in_exon
    candidates = sorted(candidates, key=lambda x: -(x[1] - x[0]))
    best = candidates[0]
    _, _, t_strand, tid, ttype = best

    feats = features_by_tid.get(tid, [])
    exons = [(s, e) for (s, e, ft) in feats if ft == "exon"]
    if t_strand == "+":
        exons_sorted = sorted(exons, key=lambda x: x[0])
    else:
        exons_sorted = sorted(exons, key=lambda x: -x[0])
    cap_dist, mature_len, exon_idx = mature_position(p, exons_sorted, t_strand)
    if cap_dist < 0:
        # The aligned centre fell outside every annotated exon; classify
        # conservatively as intron (rare under splice-aware alignment).
        return label_to_idx("intron"), -1, -1, tid
    polya_dist = mature_len - cap_dist - 1

    if ttype != "protein_coding":
        return label_to_idx("noncoding"), cap_dist, polya_dist, tid

    in_cds = any(s <= p < e for s, e, ft in feats if ft == "CDS")
    in_5utr = any(s <= p < e for s, e, ft in feats if ft == "five_prime_UTR")
    in_3utr = any(s <= p < e for s, e, ft in feats if ft == "three_prime_UTR")
    if in_cds:
        region = "CDS"
    elif in_5utr:
        region = "5UTR"
    elif in_3utr:
        region = "3UTR"
    else:
        region = "CDS"
    return label_to_idx(region), cap_dist, polya_dist, tid


def annotate_positions(trees, features_by_tid, chrom_arr, pos_arr, strand_arr):
    N = len(chrom_arr)
    region_idx = np.zeros(N, dtype=np.int8)
    cap_dist = np.full(N, -1, dtype=np.int32)
    polya_dist = np.full(N, -1, dtype=np.int32)
    transcript_id = np.full(N, "", dtype=object)
    t0 = time.time()
    for i in range(N):
        if pos_arr[i] < 0 or not chrom_arr[i]:
            continue
        c = chrom_arr[i]
        p = int(pos_arr[i])
        tree = trees.get(c)
        if tree is None:
            region_idx[i] = label_to_idx("intergenic")
            continue
        overlaps = [iv.data for iv in tree[p]]
        r_idx, cd, pd, tid = classify_position(p, overlaps, features_by_tid)
        region_idx[i] = r_idx
        cap_dist[i] = cd
        polya_dist[i] = pd
        transcript_id[i] = tid
        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - i - 1) / rate
            print(f"  annotated {i+1}/{N} rate {rate:.0f}/s ETA {eta/60:.1f} min", flush=True)
    return region_idx, cap_dist, polya_dist, transcript_id


# ---------- driver ----------

def cache_trees(gff_path, cache_path):
    if Path(cache_path).exists():
        print(f"Loading cached annotations from {cache_path}...", flush=True)
        with open(cache_path, "rb") as h:
            data = pickle.load(h)
        return data["trees"], data["features"]
    transcripts, features = parse_gencode_gff(gff_path)
    trees = build_transcript_trees(transcripts)
    print(f"Saving cache to {cache_path}...", flush=True)
    with open(cache_path, "wb") as h:
        pickle.dump({"trees": trees, "features": features}, h, protocol=4)
    return trees, features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="/ibex/user/songt/MultiRM/Data/MultiRM_data.h5")
    parser.add_argument("--reference", default="/ibex/user/songt/MultiRM/Data/reference/GRCh38.primary_assembly.genome.fa.gz")
    parser.add_argument("--gff", default="/ibex/user/songt/MultiRM/Data/reference/gencode.v44.annotation.gff3.gz")
    parser.add_argument("--cache_path", default="/ibex/user/songt/MultiRM/Data/reference/gencode.v44.trees.pkl")
    parser.add_argument("--out_dir", default="/ibex/user/songt/MultiRM/Data/metadata")
    parser.add_argument("--splits", nargs="+", default=["test", "valid", "train"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    trees, features = cache_trees(args.gff, args.cache_path)

    import mappy
    print(f"\nBuilding minimap2 index from {args.reference}...", flush=True)
    t0 = time.time()
    # preset='splice' enables intron-aware alignment for mature mRNA -> genome.
    # MultiRM 1001-nt windows are from mature mRNA, so genomic alignment must
    # be splice-aware; otherwise the centre lands inside introns of the spanned
    # transcript and region annotation collapses to "intron" for almost every
    # window.
    aligner = mappy.Aligner(args.reference, preset="splice", best_n=1)
    if not aligner:
        raise RuntimeError("Failed to build minimap2 index.")
    print(f"  ready in {time.time()-t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"\n=== split={split} ===", flush=True)
        seqs = load_split_sequences(args.h5, split)
        if args.limit:
            seqs = seqs[:args.limit]
        print(f"  {len(seqs)} sequences", flush=True)
        chrom, pos_center, strand, mapq = align_windows(aligner, seqs)
        region_idx, cap_dist, polya_dist, tids = annotate_positions(trees, features, chrom, pos_center, strand)
        out_path = out_dir / f"{split}_metadata.npz"
        np.savez_compressed(
            out_path,
            chrom=np.array(chrom, dtype=object),
            pos_center=pos_center,
            strand=strand,
            mapq=mapq,
            region_idx=region_idx,
            cap_distance=cap_dist,
            polyA_distance=polya_dist,
            transcript_id=np.array(tids, dtype=object),
            region_labels=np.array(REGION_LABELS, dtype=object),
        )
        n_mapped = int((pos_center >= 0).sum())
        n_total = pos_center.shape[0]
        rc = {REGION_LABELS[i]: int((region_idx == i).sum()) for i in range(len(REGION_LABELS))}
        print(f"  mapped {n_mapped}/{n_total} ({100*n_mapped/n_total:.1f}%); regions={rc}", flush=True)
        print(f"  wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-gffdb
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

set -euo pipefail
cd /ibex/user/songt/MultiRM

# Clean up stale partial db
rm -f Data/reference/gencode.v44.db

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

PYTHONUNBUFFERED=1 python <<'PYEOF'
import time
import gffutils
t0 = time.time()
print("Creating gffutils DB from GENCODE v44 GFF3...", flush=True)
db = gffutils.create_db(
    "/ibex/user/songt/MultiRM/Data/reference/gencode.v44.annotation.gff3.gz",
    "/ibex/user/songt/MultiRM/Data/reference/gencode.v44.db",
    force=True,
    merge_strategy="merge",
    keep_order=True,
    disable_infer_genes=True,
    disable_infer_transcripts=True,
)
print(f"DB created in {time.time()-t0:.1f}s", flush=True)
# Sanity check
print("\nFeature counts:", flush=True)
for ft in ["gene", "transcript", "exon", "CDS", "five_prime_UTR", "three_prime_UTR"]:
    n = db.count_features_of_type(ft)
    print(f"  {ft}: {n}", flush=True)
PYEOF

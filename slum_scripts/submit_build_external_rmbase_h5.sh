#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-rmbase-h5
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=120G

set -euo pipefail

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

REF_DIR=Data/reference/hg38
REF_GZ=${REF_DIR}/hg38.fa.gz
REF_FA=${REF_DIR}/hg38.fa

mkdir -p "${REF_DIR}" Data/external_rmbase/processed

if [[ ! -s "${REF_FA}" ]]; then
  if [[ ! -s "${REF_GZ}" ]]; then
    curl -L --fail --retry 5 --retry-delay 10 --connect-timeout 30 \
      --speed-time 180 --speed-limit 1024 -C - \
      -o "${REF_GZ}" \
      https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
  fi
  gunzip -c "${REF_GZ}" > "${REF_FA}.tmp"
  mv "${REF_FA}.tmp" "${REF_FA}"
fi

python Scripts/build_external_rmbase_dataset.py \
  --raw_dir Data/external_rmbase/raw/rmbase_v3 \
  --reference_fasta "${REF_FA}" \
  --max_positive_per_mod 5000 \
  --negative_ratio 1 \
  --output_h5 Data/external_rmbase/processed/external_rmbase_human.h5 \
  --metadata_csv Data/external_rmbase/processed/external_rmbase_human_metadata.csv \
  --positive_sites_csv Data/external_rmbase/processed/external_rmbase_positive_sites.csv \
  --summary_json Data/external_rmbase/processed/external_rmbase_build_summary.json

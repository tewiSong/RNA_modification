#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-rmbase-download
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=8G

set -euo pipefail

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/download_rmbase_external.py \
  --insecure \
  --timeout 600 \
  --out_dir Data/external_rmbase/raw/rmbase_v3

python Scripts/build_external_rmbase_dataset.py \
  --raw_dir Data/external_rmbase/raw/rmbase_v3 \
  --positive_sites_csv Data/external_rmbase/processed/external_rmbase_positive_sites.csv \
  --summary_json Data/external_rmbase/processed/external_rmbase_build_summary.json

#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-metadata
#SBATCH -o /ibex/user/songt/MultiRM/slurm_logs/%x.%j.out
#SBATCH -e /ibex/user/songt/MultiRM/slurm_logs/%x.%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail
cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

PYTHONUNBUFFERED=1 python Scripts/build_per_sample_metadata.py \
  --h5 Data/MultiRM_data.h5 \
  --reference Data/reference/GRCh38.primary_assembly.genome.fa.gz \
  --gff Data/reference/gencode.v44.annotation.gff3.gz \
  --cache_path Data/reference/gencode.v44.trees.pkl \
  --out_dir Data/metadata \
  --splits test valid train

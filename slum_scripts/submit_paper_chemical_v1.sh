#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-paper-chemical-v1
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch slum_scripts/submit_paper_chemical_v1.sh <bilinear|lowrank|hypernetwork>"
  exit 2
fi

SCORER_TYPE="$1"

echo "scorer_type=${SCORER_TYPE}"

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/check_cuda_v0.py

python Scripts/paper_multirm.py train_chemical_v1 \
  --data_path Data/MultiRM_data.h5 \
  --embedding_path Embeddings/embeddings_12RM.pkl \
  --modifications_path Data/modifications.csv \
  --save_dir "Results/paper_aligned/chemical_v1_${SCORER_TYPE}" \
  --cache_dir Results/paper_aligned/cache \
  --scorer_type "${SCORER_TYPE}" \
  --num_heads 8 \
  --length 51 \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.0001 \
  --lr_decay 0.8 \
  --lr_patience 5 \
  --loss_strategy paper_ohem_uw \
  --device cuda

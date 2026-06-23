#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-modid-lomo
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: sbatch slum_scripts/submit_paper_modid_lomo.sh <Am|Cm|Gm|Um|m1A|m5C|m5U|m6A|m6Am|m7G|Psi|I>"
  exit 2
fi

HELDOUT_MOD="$1"

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/check_cuda_v0.py

python Scripts/paper_multirm.py train_modid_lomo \
  --data_path Data/MultiRM_data.h5 \
  --embedding_path Embeddings/embeddings_12RM.pkl \
  --save_dir Results/paper_aligned/modid_lomo \
  --cache_dir Results/paper_aligned/cache \
  --heldout_mod "${HELDOUT_MOD}" \
  --length 51 \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.0001 \
  --lr_decay 0.8 \
  --lr_patience 5 \
  --loss_strategy weighted_bce \
  --early_stop_patience 15 \
  --weight_decay 1e-4 \
  --device cuda

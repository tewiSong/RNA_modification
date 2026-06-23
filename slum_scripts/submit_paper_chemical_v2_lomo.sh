#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-v2-lomo
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: sbatch slum_scripts/submit_paper_chemical_v2_lomo.sh <tau> <heldout_mod> [encoder=linear] [fp_kind=morgan_r2] [bio_weight=0] [seed=1]"
  exit 2
fi

TAU="$1"
HELDOUT_MOD="$2"
ENC="${3:-linear}"
FP="${4:-morgan_r2}"
BIOW="${5:-0}"
SEED="${6:-1}"
SAVE_SUFFIX=""
if [[ "${SEED}" != "1" ]]; then
  SAVE_SUFFIX="_seed${SEED}"
fi

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/check_cuda_v0.py

python Scripts/paper_multirm.py train_chemical_v2_lomo \
  --data_path Data/MultiRM_data.h5 \
  --embedding_path Embeddings/embeddings_12RM.pkl \
  --modifications_path Data/modifications.csv \
  --save_dir "Results/paper_aligned/chemical_v3_tau${TAU}_${ENC}_${FP}_bio${BIOW}${SAVE_SUFFIX}_lomo" \
  --cache_dir Results/paper_aligned/cache \
  --tau "${TAU}" \
  --chemical_encoder_type "${ENC}" \
  --fp_kind "${FP}" \
  --bio_weight "${BIOW}" \
  --seed "${SEED}" \
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

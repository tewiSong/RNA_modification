#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-v2-softlabel-filtered
#SBATCH -o /ibex/user/songt/MultiRM/slurm_logs/%x.%j.out
#SBATCH -e /ibex/user/songt/MultiRM/slurm_logs/%x.%j.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

if [[ "$#" -lt 6 ]]; then
  echo "Usage: sbatch $0 <tau> <heldout_mod> <gamma> <loss_mode> <aux_weight> <tani_min> [pwm_gate=1] [same_base=0] [encoder=linear] [seed=1]"
  exit 2
fi

TAU="$1"
HELDOUT_MOD="$2"
GAMMA="$3"
LOSS_MODE="$4"
AUX_WEIGHT="$5"
TANI_MIN="$6"
PWM_GATE="${7:-1}"
SAME_BASE="${8:-0}"
ENC="${9:-linear}"
SEED="${10:-1}"

SAVE_SUFFIX=""
if [[ "${SEED}" != "1" ]]; then
  SAVE_SUFFIX="_seed${SEED}"
fi
SAVE_SUFFIX="${SAVE_SUFFIX}_g${GAMMA}_${LOSS_MODE}_aw${AUX_WEIGHT}_tmin${TANI_MIN}"
if [[ "${PWM_GATE}" == "1" || "${PWM_GATE}" == "true" ]]; then
  SAVE_SUFFIX="${SAVE_SUFFIX}_pwm"
fi
if [[ "${SAME_BASE}" == "1" || "${SAME_BASE}" == "true" ]]; then
  SAVE_SUFFIX="${SAVE_SUFFIX}_samebase"
fi

cd /ibex/user/songt/MultiRM
source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna
python Scripts/check_cuda_v0.py

EXTRA_ARGS=()
if [[ "${PWM_GATE}" == "1" || "${PWM_GATE}" == "true" ]]; then
  EXTRA_ARGS+=(--soft_label_pwm_gate)
fi
if [[ "${SAME_BASE}" == "1" || "${SAME_BASE}" == "true" ]]; then
  EXTRA_ARGS+=(--soft_label_same_base_only)
fi

python Scripts/paper_multirm.py train_chemical_v2_lomo \
  --data_path Data/MultiRM_data.h5 \
  --embedding_path Embeddings/embeddings_12RM.pkl \
  --modifications_path Data/modifications.csv \
  --save_dir "Results/paper_aligned/chemical_v2_softlabel_filtered_tau${TAU}_${ENC}${SAVE_SUFFIX}_lomo" \
  --cache_dir Results/paper_aligned/cache \
  --tau "${TAU}" \
  --chemical_encoder_type "${ENC}" \
  --fp_kind morgan_r2 \
  --bio_weight 0.0 \
  --soft_label_gamma "${GAMMA}" \
  --soft_label_loss_mode "${LOSS_MODE}" \
  --soft_label_aux_weight "${AUX_WEIGHT}" \
  --soft_label_tani_min "${TANI_MIN}" \
  "${EXTRA_ARGS[@]}" \
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

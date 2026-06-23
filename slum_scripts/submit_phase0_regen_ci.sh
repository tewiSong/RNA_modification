#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-phase0-regen-ci
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/check_cuda_v0.py

python Scripts/regenerate_predictions.py --root Results/paper_aligned --device cuda

ROOT=Results/paper_aligned

for EXP in original_from_scratch chemical modid chemical_v1_bilinear chemical_v1_lowrank chemical_v1_hypernetwork; do
  if [[ -f "${ROOT}/${EXP}/test_predictions.npz" ]]; then
    python Scripts/bootstrap_ci.py \
      --predictions_path "${ROOT}/${EXP}/test_predictions.npz" \
      --out_csv "${ROOT}/${EXP}/test_ci.csv" \
      --mod_indices all
  fi
done

declare -A LOMO_INDEX=( [Am]=0 [Cm]=1 [Gm]=2 [Um]=3 [m1A]=4 [m5C]=5 [m5U]=6 [m6A]=7 [m6Am]=8 [m7G]=9 [Psi]=10 [I]=11 )

for PARENT in original_lomo chemical_lomo modid_lomo chemical_v1_bilinear_lomo; do
  for MOD_DIR in "${ROOT}/${PARENT}"/*/; do
    [[ -d "${MOD_DIR}" ]] || continue
    MOD=$(basename "${MOD_DIR}")
    IDX=${LOMO_INDEX[${MOD}]:-}
    [[ -n "${IDX}" ]] || { echo "skip ${MOD_DIR}: unknown mod ${MOD}"; continue; }
    if [[ -f "${MOD_DIR}/test_predictions.npz" ]]; then
      python Scripts/bootstrap_ci.py \
        --predictions_path "${MOD_DIR}/test_predictions.npz" \
        --out_csv "${MOD_DIR}/test_ci.csv" \
        --mod_indices "${IDX}"
    fi
  done
done

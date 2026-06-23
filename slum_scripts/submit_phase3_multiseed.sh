#!/bin/bash
# Phase 3 multi-seed launcher. Runs seeds 2 and 3 (seed 1 = Phase 2 baseline).
# Submits new SLURM jobs that write to save_dir with a seed suffix to avoid clobbering.
# Total: 2 seeds * (3 mods * 6 models + 1 mod * 3 missing v1 m7G runs) = 42 jobs.
# Recommendation: run AFTER Phase 2 results land so we know the new defaults look healthy.

set -euo pipefail

cd /ibex/user/songt/MultiRM

# Held-out modifications (m7G from Phase 2.2, the rest from Phase 2.3)
MODS=(m7G m6A Am Psi)
SEEDS=(2 3)
SCORERS=(bilinear lowrank hypernetwork)

# Re-submit each LOMO config with --seed and a per-seed save_dir.
# We inline the python call (don't reuse the existing slurm scripts) so we can pass --seed and override save_dir.

submit_one() {
  local job_name=$1
  local command=$2
  local extra_args=$3
  local save_subdir=$4
  local heldout=$5
  local seed=$6
  sbatch \
    --job-name="${job_name}" \
    --output="${job_name}.%j.out" \
    --error="${job_name}.%j.err" \
    --time=2:00:00 \
    --nodes=1 \
    --partition=batch \
    --gres=gpu:1 \
    --mem=120G \
    --constraint="v100|a100" \
    --wrap="set -euo pipefail; source /home/songt/anaconda3/etc/profile.d/conda.sh; conda activate /ibex/user/songt/conda_envs/rna; cd /ibex/user/songt/MultiRM; python Scripts/check_cuda_v0.py; python Scripts/paper_multirm.py ${command} --data_path Data/MultiRM_data.h5 --embedding_path Embeddings/embeddings_12RM.pkl --cache_dir Results/paper_aligned/cache --heldout_mod ${heldout} --length 51 --epochs 50 --batch_size 128 --lr 0.0001 --lr_decay 0.8 --lr_patience 5 --loss_strategy weighted_bce --early_stop_patience 15 --weight_decay 1e-4 --device cuda --seed ${seed} --save_dir Results/paper_aligned/${save_subdir}_seed${seed} ${extra_args}"
}

for SEED in "${SEEDS[@]}"; do
  for MOD in "${MODS[@]}"; do
    submit_one "ms-orig-lomo-${MOD}-s${SEED}"      train_original_lomo  ""                              original_lomo                    "${MOD}" "${SEED}"
    submit_one "ms-chem-lomo-${MOD}-s${SEED}"      train_chemical_lomo  "--modifications_path Data/modifications.csv" chemical_lomo            "${MOD}" "${SEED}"
    submit_one "ms-modid-lomo-${MOD}-s${SEED}"     train_modid_lomo     ""                              modid_lomo                       "${MOD}" "${SEED}"
    for SC in "${SCORERS[@]}"; do
      submit_one "ms-v1${SC}-lomo-${MOD}-s${SEED}" train_chemical_v1_lomo "--modifications_path Data/modifications.csv --num_heads 8 --scorer_type ${SC}" "chemical_v1_${SC}_lomo" "${MOD}" "${SEED}"
    done
  done
done

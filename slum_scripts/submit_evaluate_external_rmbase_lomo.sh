#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-rmbase-eval
#SBATCH -o slurm_logs/%x.%j.out
#SBATCH -e slurm_logs/%x.%j.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --constraint=v100|a100

set -euo pipefail

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

python Scripts/evaluate_external_lomo.py \
  --external_h5 Data/external_rmbase/processed/external_rmbase_human.h5 \
  --output_dir Results/external_rmbase_lomo \
  --device cuda \
  --methods \
    baseline=Results/paper_aligned/chemical_v3_tau0.4_linear_morgan_r2_bio0_lomo \
    baseline=Results/paper_aligned/chemical_v3_tau0.4_linear_morgan_r2_bio0_seed2_lomo \
    baseline=Results/paper_aligned/chemical_v3_tau0.4_linear_morgan_r2_bio0_seed3_lomo \
    unrestricted=Results/paper_aligned/chemical_v2_softlabel_tau0.4_linear_g0.2_lomo \
    unrestricted=Results/paper_aligned/chemical_v2_softlabel_tau0.4_linear_seed2_g0.2_lomo \
    unrestricted=Results/paper_aligned/chemical_v2_softlabel_tau0.4_linear_seed3_g0.2_lomo \
    tanimoto_filtered=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_g0.2_joint_prob_aw1.0_tmin0.45_lomo \
    tanimoto_filtered=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_seed2_g0.2_joint_prob_aw1.0_tmin0.45_lomo \
    tanimoto_filtered=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_seed3_g0.2_joint_prob_aw1.0_tmin0.45_lomo \
    proposed=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo \
    proposed=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_seed2_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo \
    proposed=Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear_seed3_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo

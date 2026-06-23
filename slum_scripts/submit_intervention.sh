#!/bin/bash --login
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J multirm-intervention
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --constraint=v100|a100

set -euo pipefail

cd /ibex/user/songt/MultiRM

source /home/songt/anaconda3/etc/profile.d/conda.sh
conda activate /ibex/user/songt/conda_envs/rna

PYTHONUNBUFFERED=1 python -u Scripts/intervention_chemistry.py

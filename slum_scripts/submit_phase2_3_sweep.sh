#!/bin/bash
# Submit the Phase 2.3 broader LOMO sweep.
# Hold for completion of smoke test first; this just enqueues SLURM jobs.
#
# Coverage: 5 model variants × 3 held-out modifications = 15 jobs.
# (v1 lowrank LOMO m7G and hypernetwork LOMO m7G are submitted separately as Phase 2.2.)

set -euo pipefail

cd /ibex/user/songt/MultiRM

MODS=(m6A Am Psi)

for MOD in "${MODS[@]}"; do
  sbatch slum_scripts/submit_paper_original_lomo.sh "${MOD}"
  sbatch slum_scripts/submit_paper_chemical_lomo.sh "${MOD}"
  sbatch slum_scripts/submit_paper_modid_lomo.sh "${MOD}"
  sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh bilinear "${MOD}"
  sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh lowrank "${MOD}"
  sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh hypernetwork "${MOD}"
done

# MultiRM paper-aligned experiments

Conda env:

```bash
conda activate /ibex/user/songt/conda_envs/rna
```

## Full 12-modification test

Control: original MultiRM, trained from scratch, no pretrained weights.

```bash
sbatch slum_scripts/submit_paper_original.sh
```

Treatment: MultiRM + chemical representation, trained from scratch, no pretrained weights.

```bash
sbatch slum_scripts/submit_paper_chemical.sh
```

These two jobs are the main control group: same data, same 51 bp input, same paper-style loss, same Table 4 metrics; only the modification representation mechanism changes.

## Leave-one-modification-out test

Control: original MultiRM LOMO baseline.

```bash
sbatch slum_scripts/submit_paper_original_lomo.sh m7G
```

Treatment: chemical-representation LOMO model.

```bash
sbatch slum_scripts/submit_paper_chemical_lomo.sh m7G
```

Use one of: `Am Cm Gm Um m1A m5C m5U m6A m6Am m7G Psi I`.

These two LOMO jobs are a second control group: the same modification is held out in both jobs; the original model and chemical model are compared on that held-out modification.

## Chemical v1: multi-head cross-attention + structured scorers

v1 replaces the additive attention with standard multi-head cross-attention and tests three scoring functions: bilinear, low-rank tensor, and hypernetwork.

Full 12-modification training (one job per scorer):

```bash
sbatch slum_scripts/submit_paper_chemical_v1.sh bilinear
sbatch slum_scripts/submit_paper_chemical_v1.sh lowrank
sbatch slum_scripts/submit_paper_chemical_v1.sh hypernetwork
```

Results are written to `Results/paper_aligned/chemical_v1_<scorer>/`.

LOMO training (scorer type as first arg, held-out modification as second):

```bash
sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh bilinear m7G
sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh lowrank m7G
sbatch slum_scripts/submit_paper_chemical_v1_lomo.sh hypernetwork m7G
```

Results are written to `Results/paper_aligned/chemical_v1_<scorer>_lomo/<heldout_mod>/`.

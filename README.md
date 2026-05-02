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

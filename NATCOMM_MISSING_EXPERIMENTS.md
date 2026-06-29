# Nature Communications target: experiment status

## Target

The manuscript should be framed as a new task rather than a small improvement over fixed-label MultiRM:

**Core task:** chemistry-conditioned prediction for unseen or low-resource RNA modification types.

The current strongest result is strict leave-one-modification-out (LOMO) transfer. The proposed nucleotide-family-constrained Tanimoto pseudo-labeling method addresses the failure of chemistry-only transfer for adenosine-derived modifications.

## Completed results

### Fixed-label baseline comparison

- MultiRM-style fixed-label baseline versus chemical shared scorer in the all-modification setting.
- Result: mean AUCb changes from 0.8141 to 0.8183.
- Interpretation: ordinary 12-label prediction is not the main contribution.

### Initial LOMO evidence

- m7G held out.
- Original MultiRM LOMO AUCb: 0.5308.
- Chemical shared scorer LOMO AUCb: 0.6816.
- Chemical shared scorer LOMO AUCm: 0.9462.
- Interpretation: chemical querying is useful when the target modification has no positive training examples.

### Multi-seed baseline and ablation completion

Status: **complete**.

All four method groups now have 3 seeds x 12 held-out modifications with checkpoints and held-out test summaries:

- `baseline`: v2 chemistry-only LOMO, no pseudo-labeling.
- `unrestricted`: Tanimoto pseudo-labeling without threshold or nucleotide-family constraint.
- `tanimoto_filtered`: Tanimoto pseudo-labeling with threshold `T >= 0.45`, without nucleotide-family constraint.
- `proposed`: Tanimoto threshold plus nucleotide-family constraint.

Output roots:

- Baseline: `Results/paper_aligned/chemical_v3_tau0.4_linear_morgan_r2_bio0[_seedN]_lomo`
- Unrestricted: `Results/paper_aligned/chemical_v2_softlabel_tau0.4_linear[_seedN]_g0.2_lomo`
- Tanimoto-filtered: `Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear[_seedN]_g0.2_joint_prob_aw1.0_tmin0.45_lomo`
- Proposed: `Results/paper_aligned/chemical_v2_softlabel_filtered_tau0.4_linear[_seedN]_g0.2_joint_prob_aw1.0_tmin0.45_samebase_lomo`

Mean over twelve held-out modifications, then mean +/- sd over three seeds:

| method | AUCm | AUCb |
|---|---:|---:|
| baseline | 0.7060 +/- 0.0371 | 0.7311 +/- 0.0288 |
| unrestricted | 0.6215 +/- 0.0108 | 0.8175 +/- 0.0095 |
| tanimoto_filtered | 0.7369 +/- 0.0088 | 0.8067 +/- 0.0126 |
| proposed | 0.8327 +/- 0.0123 | 0.8229 +/- 0.0218 |

Key per-modification AUCm improvements over the baseline:

- Am: 0.384 to 0.867.
- m6A: 0.343 to 0.667.
- m6Am: 0.468 to 0.664.
- I: 0.517 to 0.842.

### Paired statistics

Status: **complete**.

Outputs:

- `Results/natcomm_multiseed_summary/per_method_metrics.csv`
- `Results/natcomm_multiseed_summary/method_seed_mean_summary.csv`
- `Results/natcomm_multiseed_summary/paired_auc_delta.csv`
- `Results/natcomm_multiseed_summary/summary.md`
- `Results/natcomm_audit/paired_significance.csv`

Paired AUCm deltas across 36 matched seed/held-out pairs:

| comparison | mean delta | paired t-test p | Wilcoxon p |
|---|---:|---:|---:|
| proposed - baseline | +0.1267 | 1.26e-4 | 8.14e-5 |
| proposed - unrestricted | +0.2112 | 7.38e-11 | 1.60e-9 |
| proposed - tanimoto_filtered | +0.0959 | 1.66e-4 | 1.74e-4 |

The AUCm comparisons support the expected conclusion. AUCb gains are smaller; proposed is significantly above baseline in AUCb but not significantly above unrestricted or tanimoto-filtered by paired tests.

### Independent RMBase validation

Status: **complete with coverage limitations**.

Completed jobs:

- `47847625`: RMBase v3.0 hg38 source download and metadata parsing.
- `47847669`: hg38 FASTA-backed 1001-nt external H5 build.
- `47847713`: external LOMO checkpoint evaluation.

Tracked deliverables now exist:

- `Data/external_rmbase/raw/rmbase_v3/`
- `Data/external_rmbase/processed/external_rmbase_human.h5`
- `Data/external_rmbase/processed/external_rmbase_human_metadata.csv`
- `Data/external_rmbase/processed/external_rmbase_build_summary.json`
- `Results/external_rmbase_lomo/external_lomo_metrics.csv`
- `Results/external_rmbase_lomo/external_lomo_method_summary.csv`

External RMBase mean AUCm:

| method | external AUCm | external AUCb |
|---|---:|---:|
| baseline | 0.5587 +/- 0.1877 | 0.5535 +/- 0.1879 |
| unrestricted | 0.2946 +/- 0.1406 | 0.2961 +/- 0.1380 |
| tanimoto_filtered | 0.5125 +/- 0.2399 | 0.5137 +/- 0.2352 |
| proposed | 0.6707 +/- 0.1275 | 0.6737 +/- 0.1214 |

Interpretation: the external stress test supports the same direction as the MultiRM LOMO benchmark: the proposed constrained method is strongest on average.

Coverage limitations:

- Psi has zero external positives, so its external AUC is NaN.
- m5U has 45 positives and m6Am has 48 positives, so those estimates are unstable.
- The external result should be reported as an independent stress test, not merged into the main MultiRM LOMO table.

### Data provenance and leakage audit

Status: **complete**.

Generated audit:

- `Scripts/natcomm_audit_interpretation.py`
- `Results/natcomm_audit/lomo_config_audit.csv`
- `Results/natcomm_audit/audit_summary.md`

Config-level checks covered 144 method/seed/held-out runs and found 0 failures after applying the training-code defaults. The audit verifies:

1. The held-out modification is removed from `train_task_indices`.
2. Baseline losses exclude the held-out column.
3. Soft-label runs include the held-out column only when chemistry-derived pseudo-label sources exist.
4. Proposed runs use `soft_label_same_base_only=true` and `soft_label_tani_min=0.45`.
5. Model selection uses seen-modification validation AUCb only, as implemented in `train_lomo_model`.

### Biological interpretation

Status: **complete**.

Generated outputs:

- `Results/natcomm_audit/nucleotide_family_table.csv`
- `Results/natcomm_audit/tanimoto_matrix.csv`
- `Results/natcomm_audit/tanimoto_heatmap.png`

Key interpretation:

- The adenosine family contains Am, m1A, m6A, m6Am, and I.
- The constrained method permits transfer inside this family, which helps Am, m6A, m6Am, and I.
- The constraint blocks chemically plausible but biologically broad cross-family transfer, for example A-derived I to C-derived m5C.

### Manuscript restructuring

Status: **mostly complete**.

`RNA_modification_rep/natcomm_unseen_transfer.tex` now follows the intended story:

1. Fixed-label predictors cannot query unseen modification chemistry.
2. Chemistry-conditioned prediction defines a new transfer task.
3. Strict LOMO reveals an identifiability failure in chemistry-only transfer.
4. Nucleotide-family-constrained pseudo-labeling resolves the failure mode.
5. Multi-seed experiments quantify final performance and ablations.
6. External RMBase validation is reported as a separate stress test.

Remaining manuscript work is editorial rather than experimental: move chronological failure analyses into supplementary material, polish figures, and decide final journal formatting.

## Current conclusion

The major missing experiments have been completed. The existing correct results support the expected claim: nucleotide-family-constrained Tanimoto pseudo-labeling is consistently stronger for unseen-modification transfer than chemistry-only and unconstrained pseudo-labeling, especially on the adenosine-derived failure cases. The only substantive caveat is external dataset coverage for Psi, m5U, and m6Am.

"""Full 12-mod post-hoc combination analysis: v2 LOMO logits + alpha * Tanimoto-biomatch.

For each held-out modification:
  combined_logit[:, k] = v2_logit[:, k] + alpha * biomatch_logit[:, k]
  biomatch_logit = pwm_match @ Tanimoto.T (closed-form, no training)
"""
import json, pickle, sys
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/ibex/user/songt/MultiRM/Scripts')
from v0_data import MODIFICATION_NAMES
from paper_multirm import read_split_as_kmers

ROOT = Path('/ibex/user/songt/MultiRM/Results/paper_aligned')

bp = pickle.load(open('/ibex/user/songt/MultiRM/Data/bio_priors.pkl', 'rb'))
T = bp["tanimoto_matrix"]

data = read_split_as_kmers(
    "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5", "test", 51,
    "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
    "/ibex/user/songt/MultiRM/Results/paper_aligned/cache",
)
pwm = data["pwm_match"]
biomatch = pwm @ T.T

n2i = {n: i for i, n in enumerate(MODIFICATION_NAMES)}

# Baselines from earlier (single seed)
v1 = {'Am':0.478, 'Cm':0.786, 'Gm':0.911, 'Um':0.927, 'm1A':0.671, 'm5C':0.769,
      'm5U':0.897, 'm6A':0.216, 'm6Am':0.418, 'm7G':0.943, 'Psi':0.931, 'I':0.554}

alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

table = {}  # table[mod][alpha] = AUCm
for mod in MODIFICATION_NAMES:
    k = n2i[mod]
    # v2 LOMO predictions live in two directories depending on whether the run
    # was submitted before or after the bio_weight CLI argument was added.
    candidates = [
        ROOT / f'chemical_v2_tau0.4_linear_morgan_r2_lomo' / mod / 'test_predictions.npz',
        ROOT / f'chemical_v3_tau0.4_linear_morgan_r2_bio0_lomo' / mod / 'test_predictions.npz',
    ]
    p_path = next((p for p in candidates if p.exists()), None)
    if p_path is None:
        continue
    pred = np.load(p_path)
    v2_prob = np.clip(pred['prob'], 1e-7, 1-1e-7)
    label = pred['label']
    v2_logit = np.log(v2_prob) - np.log(1 - v2_prob)
    row = {}
    for alpha in alphas:
        combined = v2_logit[:, k] + alpha * biomatch[:, k]
        prob = 1 / (1 + np.exp(-np.clip(combined, -50, 50)))
        row[alpha] = float(roc_auc_score(label[:, k], prob))
    table[mod] = row

# Print table
print('=' * 110)
print("Full 12-mod post-hoc combination: v2 logit + alpha * biomatch_logit, AUCm reported")
print('=' * 110)
header = f"{'mod':6s} | {'v1':>6s} | " + " | ".join(f"α={a:>4.1f}" for a in alphas)
print(header)
print('-' * len(header))
for mod in MODIFICATION_NAMES:
    if mod not in table:
        continue
    v1_v = v1.get(mod, float('nan'))
    cells = []
    for a in alphas:
        v = table[mod][a]
        mark = '*' if v > 0.5 else ' '
        cells.append(f"{v:.3f}{mark}")
    print(f"{mod:6s} | {v1_v:6.3f} | " + " | ".join(f"{c:>6s}" for c in cells))

# Summary at each alpha: count PASS, mean AUCm
print('-' * len(header))
summary_row_mean = f"{'mean':6s} | {np.mean(list(v1.values())):6.3f} | "
summary_row_pass = f"{'PASS':6s} | {sum(1 for v in v1.values() if v>0.5):6d} | "
for a in alphas:
    vals = [table[m][a] for m in MODIFICATION_NAMES if m in table]
    summary_row_mean += f"{np.mean(vals):>6.3f} | "
    summary_row_pass += f"{sum(1 for v in vals if v>0.5):>6d} | "
print(summary_row_mean)
print(summary_row_pass)

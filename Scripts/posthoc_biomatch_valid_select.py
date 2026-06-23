"""Pick the post-hoc biomatch weight alpha on VALIDATION, then report TEST.

For each LOMO heldout=K we have a v2 checkpoint with
  valid_predictions.npz  (probabilities on the heldout-K-filtered valid set)
  test_predictions.npz   (probabilities on the full test set incl. K-positives).

Selection rule (no test leakage):
  - For each alpha and each LOMO run K, compute valid mean-AUCm over the 11
    seen modifications (the run's filtered valid set has labels for them).
  - Aggregate across the 12 LOMO runs by taking the global mean of seen-mod
    valid AUCm at each alpha.
  - alpha* = argmax of that aggregate.

The selection criterion uses only the seen modifications because each LOMO run's
saved valid_predictions has the held-out modification's positives removed (so
its valid AUC is undefined). This makes alpha* a 'does not break the seen mods'
choice: it's the largest alpha that does not hurt average seen-mod AUCm.

After picking alpha*, we report TEST results (full test set, all 12 heldout
choices, AUCm per mod plus mean and PASS count). No further tuning happens
between selection and reporting.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from v0_data import MODIFICATION_NAMES
from paper_multirm import read_split_as_kmers

ROOT = Path("/ibex/user/songt/MultiRM/Results/paper_aligned")

bp = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
T = bp["tanimoto_matrix"]

valid_pack = read_split_as_kmers(
    "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5", "valid", 51,
    "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
    "/ibex/user/songt/MultiRM/Results/paper_aligned/cache",
)
test_pack = read_split_as_kmers(
    "/ibex/user/songt/MultiRM/Data/MultiRM_data.h5", "test", 51,
    "/ibex/user/songt/MultiRM/Embeddings/embeddings_12RM.pkl",
    "/ibex/user/songt/MultiRM/Results/paper_aligned/cache",
)
biomatch_valid_full = valid_pack["pwm_match"] @ T.T  # (N_valid, 12)
biomatch_test = test_pack["pwm_match"] @ T.T

valid_y_full = valid_pack["y"]

n2i = {n: i for i, n in enumerate(MODIFICATION_NAMES)}

ALPHAS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]


def find_pred_dir(mod):
    for cand in [
        ROOT / "chemical_v2_tau0.4_linear_morgan_r2_lomo" / mod,
        ROOT / "chemical_v3_tau0.4_linear_morgan_r2_bio0_lomo" / mod,
    ]:
        if (cand / "valid_predictions.npz").exists() and (cand / "test_predictions.npz").exists():
            return cand
    return None


def load_pred(path):
    z = np.load(path)
    return np.clip(z["prob"], 1e-7, 1 - 1e-7), z["label"]


def logit(p):
    return np.log(p) - np.log(1.0 - p)


# Build, for each LOMO heldout K, the alignment mask between the LOMO-filtered
# valid set (saved in valid_predictions.npz) and the full valid set (so we can
# index biomatch_valid_full correctly).
#
# In remove_positive_rows the filter is y[:, K] < 0.5 applied to valid_y_full.
# The saved valid_predictions has rows in the same order as that filtered set.
heldout_to_validmask = {}
for mod in MODIFICATION_NAMES:
    k = n2i[mod]
    heldout_to_validmask[mod] = valid_y_full[:, k] < 0.5

# === Sweep alpha on validation ===
print("=" * 90)
print("Validation selection: mean seen-mod AUCm at each alpha (averaged across 12 LOMO runs)")
print("=" * 90)
print(f"{'alpha':>6s}  {'valid_seen_mean_AUCm':>24s}  {'#runs':>6s}")
valid_seen_at_alpha = {}
for alpha in ALPHAS:
    run_means = []
    for mod in MODIFICATION_NAMES:
        k = n2i[mod]
        pdir = find_pred_dir(mod)
        if pdir is None:
            continue
        v_prob, v_label = load_pred(pdir / "valid_predictions.npz")
        v_logit = logit(v_prob)
        mask = heldout_to_validmask[mod]
        bm = biomatch_valid_full[mask]
        if bm.shape != v_logit.shape:
            raise RuntimeError(
                f"shape mismatch for {mod}: biomatch {bm.shape} vs valid_logit {v_logit.shape}"
            )
        combined_logit = v_logit + alpha * bm
        combined_prob = 1.0 / (1.0 + np.exp(-np.clip(combined_logit, -50, 50)))
        seen_aucs = []
        for j, seen_mod in enumerate(MODIFICATION_NAMES):
            if seen_mod == mod:
                continue
            y_j = v_label[:, j]
            if y_j.min() == y_j.max():
                continue
            try:
                seen_aucs.append(roc_auc_score(y_j, combined_prob[:, j]))
            except ValueError:
                continue
        if seen_aucs:
            run_means.append(float(np.mean(seen_aucs)))
    if run_means:
        valid_seen_at_alpha[alpha] = float(np.mean(run_means))
        print(f"{alpha:>6.1f}  {valid_seen_at_alpha[alpha]:>24.5f}  {len(run_means):>6d}")

alpha_star = max(valid_seen_at_alpha, key=valid_seen_at_alpha.get)
print()
print(f"alpha* (chosen on VALID, by mean seen-mod AUCm): {alpha_star}")
print(f"   valid_seen_mean_AUCm at alpha*: {valid_seen_at_alpha[alpha_star]:.5f}")
print(f"   valid_seen_mean_AUCm at alpha=0: {valid_seen_at_alpha[0.0]:.5f}")
print()

# === Report test AUCm at alpha* (no further tuning) ===
print("=" * 90)
print(f"TEST AUCm per heldout mod, alpha=0 (v2 baseline) and alpha=alpha* (valid-selected)")
print("=" * 90)
print(f"{'mod':6s}  {'alpha=0':>10s}  {'alpha*':>10s}  {'delta':>8s}  {'PASS@0':>7s}  {'PASS@*':>7s}")
test_at0 = {}
test_at_alphastar = {}
for mod in MODIFICATION_NAMES:
    k = n2i[mod]
    pdir = find_pred_dir(mod)
    if pdir is None:
        continue
    t_prob, t_label = load_pred(pdir / "test_predictions.npz")
    t_logit = logit(t_prob)
    bm = biomatch_test
    base_prob_k = 1.0 / (1.0 + np.exp(-np.clip(t_logit[:, k], -50, 50)))
    star_prob_k = 1.0 / (1.0 + np.exp(-np.clip(t_logit[:, k] + alpha_star * bm[:, k], -50, 50)))
    try:
        a0 = float(roc_auc_score(t_label[:, k], base_prob_k))
        astar = float(roc_auc_score(t_label[:, k], star_prob_k))
    except ValueError:
        continue
    test_at0[mod] = a0
    test_at_alphastar[mod] = astar
    p0 = "Y" if a0 > 0.5 else "N"
    ps = "Y" if astar > 0.5 else "N"
    print(f"{mod:6s}  {a0:>10.4f}  {astar:>10.4f}  {astar-a0:>+8.4f}  {p0:>7s}  {ps:>7s}")
print()
print(f"mean over reported mods: alpha=0 {np.mean(list(test_at0.values())):.4f} "
      f" alpha*={alpha_star}: {np.mean(list(test_at_alphastar.values())):.4f}")
print(f"#PASS: alpha=0 {sum(1 for v in test_at0.values() if v > 0.5)}/{len(test_at0)}"
      f"   alpha*: {sum(1 for v in test_at_alphastar.values() if v > 0.5)}/{len(test_at_alphastar)}")

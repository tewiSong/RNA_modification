"""S1.5 gate: verify combined [chemistry || bio_prior] cosines satisfy
cos(m6A, m6Am) < 0.5, cos(Am, m6Am) < 0.5, cos(m6A, Am) < 0.5.

If gate fails, training should not proceed. Search over bio_weight values
to find one that passes the gate, or report that no weight works.
"""
import pickle
import sys

import numpy as np

sys.path.insert(0, "/ibex/user/songt/MultiRM/Scripts")
from v0_data import MODIFICATION_NAMES, build_chemical_feature_matrix, load_modification_table


def normalise_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def cosine_matrix(X):
    Xn = normalise_rows(X)
    return Xn @ Xn.T


def combined_features(chem, bio, bio_weight):
    # Per-row L2-normalise each block, then scale bio by bio_weight, then concat.
    chem_n = normalise_rows(chem)
    bio_n = normalise_rows(bio)
    combined = np.concatenate([chem_n, bio_weight * bio_n], axis=1)
    return combined


def report_gate(chem_cos, bio_cos, combined_cos, bio_weight):
    pairs = [("m6A", "m6Am"), ("Am", "m6Am"), ("m6A", "Am")]
    n2i = {n: i for i, n in enumerate(MODIFICATION_NAMES)}
    print(f"\nbio_weight = {bio_weight}")
    print(f"{'pair':14s}  {'chem_cos':>9s}  {'bio_cos':>8s}  {'combined':>9s}  {'gate':>6s}")
    print("-" * 60)
    all_pass = True
    for a, b in pairs:
        i, j = n2i[a], n2i[b]
        c_c = chem_cos[i, j]
        b_c = bio_cos[i, j]
        cb = combined_cos[i, j]
        passed = cb < 0.5
        all_pass = all_pass and passed
        flag = "PASS" if passed else "FAIL"
        print(f"  {a:5s}-{b:5s}    {c_c:+9.3f}  {b_c:+8.3f}  {cb:+9.3f}  {flag}")
    return all_pass


def main():
    mod_table = load_modification_table("/ibex/user/songt/MultiRM/Data/modifications.csv")
    chem = build_chemical_feature_matrix(mod_table, site_weight=0.0)
    bio_pack = pickle.load(open("/ibex/user/songt/MultiRM/Data/bio_priors.pkl", "rb"))
    bio = bio_pack["feature_matrix"]
    assert chem.shape[0] == bio.shape[0] == 12

    chem_cos = cosine_matrix(chem)
    bio_cos = cosine_matrix(bio)

    print(f"chem feature dim = {chem.shape[1]}")
    print(f"bio feature dim  = {bio.shape[1]}")

    for bw in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        combined = combined_features(chem, bio, bw)
        c_cos = cosine_matrix(combined)
        passed = report_gate(chem_cos, bio_cos, c_cos, bw)
        if passed:
            print(f"\n  >>> bio_weight={bw} passes the gate for all three critical pairs.")

    # Save the combined feature matrix at the smallest passing weight
    best_bw = None
    for bw in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        combined = combined_features(chem, bio, bw)
        c_cos = cosine_matrix(combined)
        n2i = {n: i for i, n in enumerate(MODIFICATION_NAMES)}
        if (c_cos[n2i["m6A"], n2i["m6Am"]] < 0.5
            and c_cos[n2i["Am"], n2i["m6Am"]] < 0.5
            and c_cos[n2i["m6A"], n2i["Am"]] < 0.5):
            best_bw = bw
            break

    if best_bw is None:
        print("\nNO bio_weight in tested range passes the gate. Investigate.")
        return

    print(f"\nselected bio_weight = {best_bw}")
    combined = combined_features(chem, bio, best_bw)
    print(f"combined feature dim = {combined.shape[1]}")
    print("\nfull combined cosine matrix (saved for reference):")
    c_cos = cosine_matrix(combined)
    header = "        " + " ".join(f"{n:>6s}" for n in MODIFICATION_NAMES)
    print(header)
    for i, n in enumerate(MODIFICATION_NAMES):
        print(f"  {n:>6s} " + " ".join(f"{c_cos[i,j]:+6.3f}" for j in range(12)))


if __name__ == "__main__":
    main()

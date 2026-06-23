"""Make Tani vs AUCm scatter plot for all 12 modifications."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (Tani, AUCm, label, verdict)
data = [
    (0.736, 0.478, "Am",   "FAIL"),
    (0.446, 0.786, "Cm",   "PASS"),
    (0.638, 0.911, "Gm",   "PASS"),
    (0.517, 0.927, "Um",   "PASS"),
    (0.596, 0.671, "m1A",  "PASS"),
    (0.583, 0.769, "m5C",  "PASS"),
    (0.583, 0.897, "m5U",  "PASS"),
    (0.780, 0.216, "m6A",  "FAIL"),
    (0.780, 0.418, "m6Am", "FAIL"),
    (0.406, 0.943, "m7G",  "PASS"),
    (0.392, 0.931, "Psi",  "PASS"),
    (0.577, 0.554, "I",    "PASS"),
]

fig, ax = plt.subplots(figsize=(8, 6))

for tani, aucm, name, verdict in data:
    color = "tab:red" if verdict == "FAIL" else "tab:green"
    ax.scatter(tani, aucm, c=color, s=80, edgecolors="black", linewidths=0.5, zorder=3)
    # offset labels slightly to avoid overlap
    offsets = {"m6A":(0.01, -0.04), "m6Am":(0.005, 0.025), "Am":(0.005, 0.025),
               "m1A":(-0.04, 0.02), "I":(0.005, -0.03), "m5C":(-0.04, -0.03), "m5U":(0.005, 0.02),
               "Um":(0.005, -0.025), "Gm":(0.005, 0.02), "Cm":(0.005, 0.005),
               "m7G":(0.005, -0.025), "Psi":(0.005, 0.02)}
    dx, dy = offsets.get(name, (0.005, 0.005))
    ax.annotate(name, (tani + dx, aucm + dy), fontsize=10)

# AUCm = 0.5 line (random)
ax.axhline(0.5, ls="--", c="gray", lw=0.8, label="AUCm = 0.5 (random)")

# Empirical Tani threshold band
ax.axvspan(0.65, 0.73, color="lightgray", alpha=0.4, zorder=0, label="Tani threshold band (no data)")

ax.set_xlabel("Tanimoto similarity to nearest training neighbor (Morgan FP r=2)")
ax.set_ylabel("LOMO held-out AUCm")
ax.set_title("Chemical-conditioned LOMO performance vs chemical-neighbor closeness\n(12 RNA modifications, chemical_v1_bilinear, weighted_bce)")
ax.set_xlim(0.35, 0.82)
ax.set_ylim(0.10, 1.00)
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)

# Add summary annotation
ax.text(0.78, 0.18, "Tani > 0.70:\n3/3 FAIL\n(m6A, m6Am, Am)", fontsize=9, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", edgecolor="red"))
ax.text(0.42, 0.20, "Tani ≤ 0.64:\n9/9 PASS", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", edgecolor="green"))

out = "/ibex/user/songt/MultiRM/LOMO_scatter.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print(f"wrote {out}")

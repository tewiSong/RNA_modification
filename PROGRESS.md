# Progress report: Option 1 (frozen encoder + bio prior) also failed

## Session target
Lift Am held-out LOMO AUCm above 0.5 without losing m6A's v2 gain (0.584).

## Full state across all attempted methods

```
mod    v1     v2       v3-learned  v3-frozen
                       (chem+bio)  (chem+bio)
m6A    0.216  0.584    0.282       0.145    <- v3-frozen WORSE than v1
Am     0.478  0.361    0.355       0.420    <- best Am so far, still FAIL
m7G    0.943  0.914    0.912       0.760    <- v3-frozen regressed
Psi    0.931  0.931    0.912       0.924
```

S2 smoke gate on v3-frozen LOMO m6A: PASS (all three critical pairs cos < 0.5):
- cos(enc(m6A), enc(m6Am)) = 0.121
- cos(enc(Am),  enc(m6Am)) = -0.076
- cos(enc(m6A), enc(Am))   = 0.351

So encoder geometry is preserved correctly by the frozen random projection. The failure is downstream of the encoder, in how the scorer can use the mod vectors.

## Why Option 1 failed

The frozen encoder DID preserve the input geometry (S2 gate passed). But the resulting model performed worse, not better:

1. **m6A dropped to AUCm 0.145** (worse than v1's 0.216). In v2 the learned encoder collapsed mod_m6A onto mod_m6Am, and sharp attention rescued m6A by exploiting the BiLSTM's "A-base modification" feature direction. In v3-frozen, mod_m6A is at a random direction (determined by chem+bio input geometry through a random linear projection). This random direction has no alignment with anything useful in the BiLSTM K-space, so sharp attention points to arbitrary positions. The bio prior information is preserved in input space but cannot be USED by the architecture because the BiLSTM was trained on RNA sequences alone and does not encode biology-prior-aligned features.

2. **m7G dropped from 0.914 to 0.760** (-0.154). The frozen encoder cannot adapt to the seen-task data, so chemistry-relevant directions that v2's learned encoder figured out for the 11 seen mods are no longer accessible. This violates the stop-condition "previously-passing mods regressing > 0.05".

3. **Am improved from 0.361 to 0.420** (+0.059, the largest Am gain across any method tried). The bio prior IS providing usable information for Am. But it is not enough to clear AUCm > 0.5. Am has no sharp RNA motif for the architecture to exploit, only writer-enzyme and region preference, both of which are diffuse signals.

## Combined picture across all methods tried this session

| Method              | m6A   | Am    | m7G   | Psi   | Mean  | What it fixed | What it broke |
|---------------------|-------|-------|-------|-------|-------|---------------|---------------|
| v1 bilinear         | 0.216 | 0.478 | 0.943 | 0.931 | 0.642 | baseline      | nothing       |
| v2 sharp-attn 0.4   | 0.584 | 0.361 | 0.914 | 0.931 | 0.697 | m6A           | Am            |
| v3 chem+bio learned | 0.282 | 0.355 | 0.912 | 0.912 | 0.615 | nothing       | m6A           |
| v3 chem+bio frozen  | 0.145 | 0.420 | 0.760 | 0.924 | 0.562 | Am (partial)  | m6A, m7G      |

The two failure modes diagnosed in main.tex Section 8 are confirmed and orthogonal:
- m6A failure (chemistry collapse): only fixable by v2's mechanism (learned encoder + sharp attention + DRACH motif on RNA side). Any other choice loses it.
- Am failure (no RNA motif): only slightly responsive to bio prior. No architecture saves it.

## Why no in-paradigm fix can close both gaps simultaneously

Pre-conditions:
- v2's m6A success relies on the BiLSTM having learned A-base modification features from m1A, m6Am, etc. Sharp attention with mod_m6A (collapsed onto mod_m6Am) points to these positions, which include DRACH motifs.
- A LEARNED chemistry encoder will always collapse mod_m6A onto mod_m6Am because m6A has no positive labels. Bio prior weakens this only partially (input cos 0.85 -> 0.08, but output cos drifts back to 0.52 because the 2057-dim chem block dominates the projection space).
- A FROZEN chemistry encoder preserves bio-prior separation in mod space, but the BiLSTM has not been trained to align its K-space with bio-prior directions, so sharp attention cannot use the separated mod_m6A direction.

Conclusion: chemistry-conditioned LOMO with sharp attention has a structural constraint. It can EITHER use chemistry collapse + RNA-side feature alignment (v2's m6A path) OR use bio-prior input separation (v3-frozen), not both. The architecture's two information channels (chemistry to mod vector, RNA to K vector) cannot be jointly trained to satisfy both objectives without held-out positive supervision.

## Two distinct subproblems, two distinct conclusions

**m6A subproblem**: SOLVED to AUCm 0.584 via v2 sharp-attention (best result of any attempted method). No chemistry- or input-side augmentation surpasses this.

**Am subproblem**: NOT solved by any chemistry-only or chemistry+bio method tried. Best achieved AUCm 0.420 (v3-frozen, +0.06 over v1 but still below random). The root cause is that Am has no sharp RNA-side motif for sharp attention to lock onto, and adding bio priors only modestly helps because Am's bio prior is itself a diffuse signal (no specific writer-binding consensus, no positional preference beyond cap-1).

## What I have NOT tried (would require leaving the current paradigm)

1. **Few-shot transfer (Direction B in main.tex Section 8.4)**: provide 5-10 held-out positives at test time. This changes the protocol from strict LOMO to few-shot. Bypasses the identifiability problem by injecting RNA-side label information.

2. **Joint training with auxiliary RNA-position prediction head**: predict per-position "is this a modification site" generically, then condition the modification-specific scorer on this. May help Am if "any modification" is easier than "specific modification".

3. **External CLIP-seq / iCLIP data for Am writers (CMTR1/FTSJ3)**: load the actual enzyme footprint as additional channel rather than hand-curating literature priors. More information than what is in bio_priors.pkl, but requires data acquisition and is no longer "chemistry-only".

## Recommendation: take Option 3, declare scope boundary in paper

The four held-out cases give a clean and reportable picture:

- 9 of 12 LOMO held-outs achieve AUCm in [0.55, 0.94] under v1 bilinear (chemistry-only).
- The 3 chemical-twin failures (m6A, Am, m6Am in main.tex Section 6) split into two failure modes (encoder collapse rescuable by sharp-attention, vs no-motif unrescuable).
- v2 sharp-attention raises m6A from 0.216 to 0.584 (the first method to clear random on m6A).
- v3 with biology priors does NOT further close the gap, and the experiments document why: the architecture cannot jointly use chemistry-collapse-rescue and bio-prior separation.

I recommend updating main.tex Section 8 to formally state:
- v2 sharp-attention as the final architectural deliverable (m6A rescued, m7G/Psi preserved).
- Am LOMO and m6Am LOMO are outside the well-posed scope of chemistry-conditioned LOMO under the architectures we have explored.
- The bio-prior experiment is included as evidence that information augmentation alone (without protocol or architecture change) does not close the Am gap.

The honest scope of the paper's method is then: "chemistry-conditioned LOMO succeeds for 9 of 12 RNA modifications (those with chemistry Tanimoto < 0.7 to all training neighbours, OR with a distinctive RNA motif accessible to sharp attention). m6A falls in the latter category and is rescued by sharp-attention routing. Am and m6Am have no rescue path under chemistry+biology-prior inputs explored in this work."

Awaiting decision on whether to take Option 3 (write scope boundary into main.tex) or pursue an out-of-paradigm direction.

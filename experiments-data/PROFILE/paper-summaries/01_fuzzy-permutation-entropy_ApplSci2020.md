# Paper Summary — Fuzzy Entropy & Permutation Entropy for sEMG Hand-Gesture Characterization

| | |
|---|---|
| **File** | `data-science-related-papers/applsci-10-07144.pdf` |
| **Citation** | Marri et al. / *"...Fuzzy Entropy and Permutation Entropy for hand-gesture characterization"*, **Applied Sciences** 2020, 10(20), 7144. https://www.mdpi.com/2076-3417/10/20/7144 |
| **Maps to** | **Module 1** (signal-level complexity/non-stationarity) — primary parameter source for Fuzzy Entropy (FEn) & Permutation Entropy (PEn) |
| **PLAN §5 item** | #1 |

---

## 1. What the paper is
A characterization study (not a new model) that asks whether two complexity measures —
**Fuzzy Entropy (FEn)** and **Permutation Entropy (PEn)** — are good *features* for distinguishing
hand gestures from a low-cost 8-channel **Myo armband** (200 Hz, 8-bit), across 10 healthy subjects
and 14 gestures grouped into three sets (finger / wrist / grasp). It evaluates them three ways:
(a) k-means gesture clustering quality via the gap statistic, (b) mRMR predictive-importance ranking
against 18 classical TD/FD features, (c) LDA classification accuracy when FEn/PEn are *added* to
standard feature sets.

## 2. Why it matters to our use case
This is the **cleanest published recipe** for computing FEn and PEn on real sEMG *and* direct evidence
that these complexity features carry **independent discriminative information** beyond amplitude/spectral
features. Our Paper-2 Module 1 thesis is exactly this: "complexity/non-stationarity features exist because
sEMG is nonlinear, and they capture what linear features miss." This paper supplies both the parameters
we need and the empirical justification for including them.

## 3. What we take (exact recipes — fold into Module 1)
**Fuzzy Entropy (FEn), Chen-style:**
- Embedding dimension **m = 2** (fixed).
- Chebyshev (max) distance between baseline-removed embedding vectors.
- Fuzzy membership: `Γ = exp( −(d^n) / r )`.
- Gradient **n ∈ {1, 2, 5}**; tolerance **r ∈ {0.1, 0.3, 0.5}**.
- **Finding: n = 1 fails** (cannot recover the correct number of gesture clusters for any r).
  **n = 5 gives the best clustering**; for the classification comparison they used **r = 0.3, n = 5**.
  `n > 5` is *not recommended* (approaches a hard Heaviside → loses detail).

**Permutation Entropy (PEn):**
- Embedding dimension **d ∈ {3, 5, 7}**, delay **τ = 1**.
- Normalise by `ln(d!)` → range [0, 1].
- **Finding: d = 5 is best** (d = 7 fails to recover cluster count).
- **★ Critical practical rule (from the discussion):** reliable PEn needs **`d! ≪ N`** (window length).
  d=7 → 5040 permutations vs a 600-sample Myo window → fails; d=5 → 120 ≪ 600 → works. For our datasets
  choose `d` per window length: at 100 ms × 1–2 kHz (100–200 samples), **d=4 or 5 is the safe ceiling**.
- **PEn is amplitude-blind** (pure ordinal) → naturally amplitude-invariant, i.e. robust to cross-subject
  gain differences — a *plus* for our cross-subject characterisation, at the cost of ignoring amplitude info.

**★ Include BOTH FEn and PEn (they are complementary, not redundant):** FEn uses a distance metric,
PEn uses ordinal patterns — different core logic, so adding both gave a *dramatic* accuracy jump vs either
alone (avoids feature redundancy). Also note **PEn is ~57× cheaper than FEn** (for N=10⁴: 0.08 s vs 4.61 s),
and **Sample Entropy (SampEn) was also consistently top-5 in mRMR** → include SampEn as well.
General guidance: entropy needs **≥ ~200 samples** per window for a valid estimate.

**Preprocessing they used:** 2nd-order zero-phase Butterworth **high-pass at 10 Hz** only (no band-pass,
because the Myo's 200 Hz / 8-bit limits bandwidth). Entropy computed **per channel, per active window**.

## 4. What mattered / what didn't
- **Mattered:** the *gradient n* of the fuzzy function is the single most impactful knob — get it wrong
  (n=1) and the feature is useless; n=5 makes it the top-ranked feature. This is a non-obvious result we
  bake straight into our defaults.
- **Mattered:** mRMR ranked **FEn and PEn as the #1 / top-few** most-relevant features for 2 of 3 gesture
  sets, *above* WA, RMS, MNF, etc. → complexity features are not redundant with classical ones.
- **Mattered:** adding FEn+PEn to the HDF (Hudgins+Du+freq) set lifted LDA accuracy **74.5% → 88.2%**
  (+13.7 pp) — the strongest single argument that these features add real information.
- **Didn't matter for us:** the specific Myo hardware, the k-means/gap-statistic clustering machinery
  (we use separability indices in Module 2 instead), and their exact 14-gesture taxonomy.

## 5. Novelty it lends to *our* Paper 2
It converts a hand-wavy "entropy features are nice" into a **defensible, parameterised measurement** with
a known best-operating-point. In Module 1 we can report FEn/PEn per subject/class/channel across all our
datasets and *cite this paper for the parameter choice*, so a reviewer cannot dismiss the values as
arbitrarily tuned. It also gives us a ready-made **"complexity adds info" validation**: if our
Module-2 separability analysis shows FEn/PEn improving class separation, we are replicating a published
effect on new datasets (EMAHA, NinaPro, etc.).

## 6. Methodology we adopt
Compute FEn(m=2, r=0.3, n=5) and PEn(d=5, τ=1) per channel per window as Module-1 features; report their
distributions and their mRMR / separability rank against our classical TD/FD sets. Use the HP-only
preprocessing note as a caution: for our higher-rate datasets (1–2 kHz) we keep the standard band-pass,
but we record that entropy features are computed on the same windows as everything else.

## 7. Caveats / what NOT to borrow
- Their `r` values (0.1/0.3/0.5) appear to be used as *absolute* tolerances on Myo-scaled signals. The
  entropy literature (see summaries 02, 03) defines `r = k·std(x)`. **We standardise on `r = k·std`**
  (k≈0.25–0.3) so the tolerance is amplitude-invariant across our datasets — otherwise FEn is not
  comparable across subjects/datasets with different gains. Flag this reconciliation in the module code.
- Don't inherit their clustering-based evaluation; our separability story is Module 2.

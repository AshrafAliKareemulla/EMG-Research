# Paper Summary — Multiscale Fuzzy Approximate Entropy of Biceps sEMG

| | |
|---|---|
| **File** | `data-science-related-papers/navaneethakrishna2015.pdf` |
| **Citation** | Navaneethakrishna M, Karthick P A, Ramakrishnan S, *"Analysis of Biceps Brachii sEMG signal using Multiscale Fuzzy Approximate Entropy"*, **IEEE EMBC / conf. 2015**, pp. 7881–7884. https://pubmed.ncbi.nlm.nih.gov/26738119/ |
| **Maps to** | **Module 1** (multiscale complexity + fatigue/non-stationarity validation) |
| **PLAN §5 item** | #2 |

---

## 1. What the paper is
Differentiates **fatigue vs non-fatigue** biceps sEMG (50 subjects, dynamic biceps-curl to failure) using
**Multiscale Fuzzy Approximate Entropy (MSfApEn)** — fuzzy ApEn computed on coarse-grained versions of the
signal at time scales 1–10 — and summarises the multiscale curve with three scalar features (MED, LS-MED,
HS-MED). Fatigue *lowers* complexity at all scales, and this separates the two conditions with high
statistical significance (p ≪ 0.001).

## 2. Why it matters to our use case
Two things. First, it gives the **multiscale extension** of the entropy recipe — a principled way to turn a
per-window entropy into a small feature vector that captures structure at multiple temporal resolutions,
which is directly reusable in Module 1. Second, it is a **built-in validation of what our complexity
features are supposed to measure**: fatigue = motor-unit synchronisation = lower complexity. When our
Module-1 features respond to known physiological drift the same way, we can trust them as descriptors of
non-stationarity (and it ties into Module 3's temporal-drift / inter-day story).

## 3. What we take (exact recipes — fold into Module 1)
**Fuzzy ApEn (fApEn):**
- Embedding dimension **m = 2**.
- Chebyshev (max) distance; **baseline removal** of each embedding vector (the ApEn→fApEn generalisation).
- Fuzzy membership **Gaussian: `Ω(d) = exp( −(d²)/r )`** (exponent fixed at 2 here).
- Tolerance **`r = k·std(x)`, with k = 0.25** (their empirical choice).

**Multiscale coarse-graining:**
- Scale `t_s = 1…10`; coarse-grained series `y_j = (1/t_s) Σ x` over non-overlapping windows (moving-average
  decimation, no overlap). Compute fApEn at each scale → MSfApEn curve.
- **Summary features:** `MED` (median over all 10 scales), `LS-MED` (median over scales 1–5, "low scale"),
  `HS-MED` (median over scales 6–10, "high scale"). These 3 scalars are the actual per-channel features.

**Preprocessing they used:** downsample to 1000 Hz, band-pass **10–400 Hz**, **50 Hz notch**.

## 4. What mattered / what didn't
- **Mattered:** the coarse-grain → per-scale-fApEn → {MED, LS-MED, HS-MED} pipeline is a compact,
  citable way to add multiscale complexity to our feature bank without exploding dimensionality.
- **Mattered:** fatigue lowered fApEn *and* increased its inter-subject variance — evidence that
  complexity features are sensitive to the exact temporal drift Module 3 wants to quantify.
- **Mattered (a caution we keep):** they normalise `r` to the signal std (k=0.25). This is the amplitude-
  invariant convention we adopt globally (reconciles the ambiguity in summary 01).
- **Didn't matter for us:** single-muscle (biceps only) focus; the specific fatigue protocol; the t-test
  reporting. We only borrow the estimator and its parameters.

## 5. Novelty it lends to *our* Paper 2
Lets Module 1 report **multiscale** complexity, not just single-scale entropy — a richer descriptor that
distinguishes datasets/subjects whose signals differ in *temporal structure* rather than amplitude.
Combined with the fatigue result, it gives us a physiological anchor: "our complexity features move in the
direction a fatigue study predicts," which pre-empts the reviewer question "do these numbers mean anything?"

## 6. Methodology we adopt
Add MSfApEn with scales 1–10 to Module 1 and emit MED/LS-MED/HS-MED per channel. Reuse the fApEn core
(m=2, Gaussian membership, r=0.25·std, Chebyshev, baseline removal) — the same core as summaries 01/03, so
one implementation serves all three. Where a dataset has repeated reps or long trials, optionally compute
the low-vs-high-scale split as a coarse "non-stationarity index" per subject for the Module-3 drift figures.

## 7. Caveats / what NOT to borrow
- Coarse-graining shortens the series by `t_s`×; at scale 10 a short window can become too few points for a
  stable fApEn. Enforce a **minimum coarse-grained length** (reuse the existing `_ensure_min_length` guard
  from `emaha_features.py`) and skip/flag scales that underflow, per-dataset.
- Their 10 kHz→1 kHz downsample is dataset-specific; we compute at each dataset's native rate and treat
  sampling-rate sufficiency as a *separate* Module-4 question, not something to hard-code here.

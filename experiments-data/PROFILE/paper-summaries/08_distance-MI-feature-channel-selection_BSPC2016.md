# Paper Summary — Distance & Mutual-Information Methods for EMG Feature/Channel Subset Selection

| | |
|---|---|
| **File** | `data-science-related-papers/1-s2.0-S1746809416300040-main.pdf` |
| **Citation** | Al-Angari H M, Kanitz G, Tarantino S, Cipriani C, *"Distance and mutual information methods for EMG feature and channel subset selection for classification of hand movements"*, **Biomedical Signal Processing and Control** 27, 2016, 24–31. |
| **Maps to** | **Module 2** (separability, distance & MI families) + **Module 4** (channel selection). |

---

## 1. Why it matters
The plan's Module-2 revision says "use BOTH filter and wrapper/MI families." This paper gives two concrete,
deterministic, **feature-and-channel joint** selection recipes — one distance-based, one MI-based — that we can
implement directly as the two separability families, and it independently confirms our entropy tolerance
convention (SampEn m=2, r=0.25·SD).

## 2. What we take (two recipes)
**DFSS (distance family) — Mahalanobis separability index.** For feature/channel F/C(i), compute a separability
index `SI = ½ · avg over class-pairs of min Mahalanobis distance` between a class ellipse centre and the nearest
point of other classes (covariance-aware). Rank by SI. → Module 2 "distance-based separability."

**CFSS (MI family) — symmetrical uncertainty.** `SU(C; F/C) = 2·I(C; F/C)/(H(C)+H(F/C))` with histogram-based
entropy. "Good subset = high MI with class, low MI among features" (max-relevance/min-redundancy). Rank by SU,
then greedily prune redundant F/C. → Module 2 "MI-based separability" + Module 4 min-redundancy channel subset.

## 3. Key findings
- **CFSS (MI) selected AR coefficients and SampEn most** (highest class MI); DFSS (distance) selected wavelet
  features most → the two families *disagree on which features matter* — exactly why the plan says report both.
- SampEn alone classifies poorly but **adds information when combined** (consistent with summaries 01/04).
- Both beat all-channels and Hudgins-TD under limb-position variation; **posterior forearm channels more
  discriminative** than anterior (montage/robustness note for Module 4).
- SampEn params: **m=2, r=0.25·SD** (confirms our shared entropy core).

## 4. Methodology we adopt / caveats
- Implement Mahalanobis-SI and MI/symmetrical-uncertainty as Module-2's distance and MI separability scorers;
  report which one better predicts downstream accuracy (the plan's open question).
- **Caveat:** their study is 9-arm-position robustness (position effect), not cross-subject. We reuse the
  *selection metrics*, and evaluate them under our LOSO framing where the ranking may differ.

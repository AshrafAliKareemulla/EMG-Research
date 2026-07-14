# Paper Summary — Electrode Placement × Gesture Category (HD-sEMG)

| | |
|---|---|
| **File** | `data-science-related-papers/fnins-19-1750792.pdf` |
| **Citation** | Qiu F, Liu X, Ye X, *"Influence of electrode placement on the recognition of different gesture categories using high-density sEMG"*, **Frontiers in Neuroscience** 19:1750792, 2025 (open access). 256-ch HD-sEMG, 20 subjects, 2 days, 3 forearm regions. |
| **Maps to** | Robustness (G1/G2 placement) + **Module 3** (a second inter-subject-vs-inter-day data point) + Module 4 (region-conditional information). |

---

## 1. Why it matters
A clean, modern validation-protocol template and a **second independent confirmation of the shift hierarchy**
(intra > inter-day > inter-subject), plus the finding that *which region carries class information depends on the
gesture category* — relevant to our ADL angle and to Module 4's "where is the information."

## 2. Key findings we cite
- **Shift hierarchy (single-DoF, distal wrist): intra-subject 98.63% > inter-day 79.73% > inter-subject 75.47%.**
  Confirms **inter-subject shift > inter-day shift** (aligns with Li 2024, summary 10) → strengthens our A4 answer.
- **Gesture-category × placement:** single-DoF (isolated finger) gestures classify best from the **distal wrist**;
  daily-used (compound, multi-finger) gestures best from **mid-forearm / proximal elbow**. → for ADL (compound)
  gestures, mid/proximal regions carry more class information.
- Clean **validation-protocol template** we mirror: leave-one-movement-out (intra), day1-train/day2-test
  (inter-day), LOSO (inter-subject); stats via **Friedman + Wilcoxon signed-rank + Bonferroni**.

## 3. What we take (methodology)
- Adopt the three-protocol framing and the **Friedman/Wilcoxon/Bonferroni** non-parametric stats for Module 2/3
  comparisons (accuracy/separability are non-normal → non-parametric tests, not ANOVA).
- Preprocessing/feature reference: 8 features (RMS, MYOP, SSC, Skew, WAMP, Kurt, MNF, MDF); window 500 ms/125 ms
  (75% overlap), exclude first 0.25 s reaction; **z-score norm**; PCA-95% with `D ≤ N_s−1`; outlier reconstruction
  via mean±2·std → all consistent with our planned pipeline.

## 4. Caveats
- HD-sEMG (256 ch) and region-comparison design; our datasets are sparse and single-montage, so the *placement*
  conclusions are context (robustness discussion) rather than something we reproduce. We reuse the **protocol +
  stats + shift-hierarchy evidence**, and note the ADL-relevant "compound gestures favour proximal regions" point
  for the discussion.

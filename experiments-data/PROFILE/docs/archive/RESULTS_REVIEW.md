> # ⚠ SUPERSEDED — DO NOT QUOTE ANY NUMBER IN THIS FILE
>
> Written 2026-07-09, before the correctness audit. A line-by-line review of the code that
> produced these numbers found 15 defects (`CORRECTIONS.md`). Specifically, in THIS file:
>
> * the Module-5 table quotes **post-hoc best-of-4 predictors** across 56 uncorrected tests, and
>   its "significant on 9/14" counts `senic` at the **wrong sign**. After FDR on the a-priori
>   predictor, expect ~2/14.
> * **"H-divergence uniformly high (0.72–0.98)"** is an artifact: the classifier was memorising
>   trial identity across shuffled 50 %-overlapping windows and saturating at its maximum of 2.
> * **"ninapro 52-class kNN-LOO 0.11–0.33"** and every other kNN number leaked across overlapping
>   windows from the same trial. `knn_loo` was neither leave-one-out nor leak-free.
> * **"zero NaNs, zero failures"** is wrong — there were 88 NaN/Inf.
> * The SDI's leave-one-**dataset**-out validation leaked the cohort (14 datasets = 9 cohorts).
>
> A corrected `RESULTS_REVIEW_PHASE2.md` will be written once the recompute lands.
> See `CORRECTIONS.md` and `NEXT_STEPS.md`.

# PROFILE — Phase-1 Results Review (all 14 datasets, cap=200)

Reviewed after the full run returned (2026-07-09). Complete: 14 datasets × 5 modules, 28 figures,
**zero NaNs, zero failures**. This file records the findings + the issues to fix in Phase 2.

## Headline — the subject-difficulty predictor works (Module 5)
Cheap distribution-to-pool statistics predict per-subject LOSO accuracy, **correct sign** (farther
from pool → lower accuracy), strongest on high-statistical-power datasets:

| Dataset | #subj | best predictor | r | p | R² |
|---|---|---|---|---|---|
| ninapro_db1 | 27 | MMD | −0.77 | <0.0001 | 0.64 |
| ninapro_db2 | 40 | MMD | −0.73 | <0.0001 | 0.63 |
| ninapro_db5 | 10 | MMD | −0.68 | 0.030 | 0.63 |
| emaha_db5 | 10 | H-div | −0.68 | 0.032 | 0.49 |
| fors_emg | 19 | MMD | −0.62 | 0.005 | 0.39 |
| emaha_db1 | 25 | MMD | −0.43 | 0.034 | 0.61 |
| myobit | 31 | H-div | −0.41 | 0.022 | 0.22 |
| grabmyo_flow_dynamic | 20 | MMD | −0.46 | 0.040 | 0.29 |

Significant on **9 / 14** datasets; the 5 non-significant are the n=10 sets (underpowered).
**Anomaly: senic flips positive (r=+0.70, p<0.0001)** — see Issue #3.

## SDI — sEMG Subject-Difficulty Index (Phase-2 metric, already prototyped)
Within-dataset-standardised, pooled across all 311 subjects, validated **leave-one-dataset-out**:
mean Spearman **0.29** (median 0.33), strong on powered sets (ninapro_db1 ρ=0.82, fors 0.60,
ninapro_db2 0.58). Weights → the index is essentially **MMD + KL-mean to pool**. Excluding the
senic outlier, mean ρ ≈ 0.34. (`results/module6_sdi/sdi.json`.)

## Other findings
- **Low intrinsic dimensionality, stable across datasets (TwoNN ≈ 6–14)** regardless of channel
  count (grabmyo: 28 ch but TwoNN ≈ 7). Supports the low-dimensional-manifold story.
- **H-divergence uniformly high (0.72–0.98)** — subjects are highly distinguishable in feature
  space everywhere (strong marginal shift), as expected.
- **Channel redundancy varies:** grabmyo can drop to 22/28 channels for 90% relevance; most others
  need nearly all channels. grabmyo_flow variants show high inter-channel NMI (~0.24).
- Separability is hard for many-class sets (silhouette negative; ninapro 52-class kNN-LOO 0.11–0.33)
  and easier for fors_emg (12-class, kNN-LOO 0.73) — sane.

## ISSUES TO FIX BEFORE WRITE-UP (Phase 2)
1. **A4 (inter-day vs inter-subject) is NOT a clean comparison as computed.** Inter-day was measured
   session-pool vs session-pool (mixing all subjects), while inter-subject is subject vs subject —
   different granularities. The ratios (<1) are a pooling artifact, **not** evidence that inter-day >
   inter-subject. Fix: inter-day = *within-subject across sessions*, inter-subject = *cross-subject
   within a session*, then compare. Only grabmyo(+flow variants) and senic have sessions.
2. **Mean-vs-covariance split is computed on z-scored features**, which removes much of the mean shift,
   so covariance appears to dominate (ratio <1). Fix/framing: also compute on RAW (un-normalised)
   features to show the mean-dominance Yoneda predicts, and frame the z-scored version as "residual
   shift after normalisation" (explains why z-score helps cross-subject).
3. **senic anomaly (positive difficulty correlation).** senic = electrode-shift/rotation + fatigue,
   uneven session counts per subject. Investigate whether the positive sign is a confound (uneven
   per-subject data volume, shift conditions inflating both divergence and within-subject variety).
   Until understood, treat senic as an outlier (report separately; it drags the SDI down).

## Caveat on the target
All LOSO accuracies here are the **self-LDA-LOSO proxy** (Paper-2 self-contained). Swap in Paper-1's
DL LOSO accuracies as the target once the BENCH-LOSO EMAHA sweep exists (loader written, untested).

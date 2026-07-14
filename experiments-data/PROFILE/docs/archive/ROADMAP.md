# PROFILE — Paper 2 (Data-Science Characterization of sEMG ADL Datasets) — ROADMAP

> **WARNING (2026-07-10):** novelty hook 3 below is VOID as posed (see the strikethrough), and the
> SDI's *actionability* claim is a null result (~1 pp oracle ceiling). Every result produced before
> 2026-07-10 is superseded. Read `CORRECTIONS.md` and `NEXT_STEPS.md` before using this document to
> draft the paper. The methods/parameters in section 3 remain correct except E3 (rewritten) and E6
> (needed an anti-alias filter).
>
> **Status:** Roadmap FINAL (all 14 supplied papers reviewed — see `paper-summaries/01`–`14` + `INDEX.md`).
> Code build follows this document. Companion to Paper 1 (`../../experiments-dl/BENCH-LOSO/`).
> **Author context:** BTech-honors researcher; Track-1 classical ML + Track-2 DL for ADL classification;
> primary datasets EMAHA / NinaPro / BioPatRec / GrabMyo; hardware A5000 24 GB. Healthy subjects only.

---

## 1. What this paper is (the one-paragraph thesis)

**A statistical & mathematical characterization of sEMG ADL datasets — what the data itself tells us *before any
model is trained*, and which cheap, calibration-window statistics predict cross-subject difficulty.** There is
**no accuracy axis**: the contribution is *understanding* + a *predictive tool*, not a SOTA number, so the paper
cannot "look bad" the way a LOSO accuracy can. It is **CPU-only** and runs **in parallel** with Paper 1 (zero
extra GPU wall-clock), which is the deliberate risk hedge: even if Paper 1's method gains are modest, Paper 2
stands on its own. It is built entirely on the existing `semg/` L1 loaders + handcrafted features — **no new
data, no training**. The single dependency on Paper 1 is Module 5, which consumes Paper 1's per-subject LOSO
accuracies as the *target* its difficulty predictor tries to forecast.

### Why anyone should read it (novelty hooks, grounded in the 14 papers)
1. **Multi-dataset extension of a single-dataset method.** The cross-subject-difficulty predictor of
   Albuquerque et al. 2022 (Frontiers AI, summary 07) was shown on **one EEG dataset**. We run its conditional-
   /marginal-shift estimators across **6+ sEMG ADL datasets** and correlate them with real LOSO accuracies — the
   multi-dataset validation that paper explicitly calls for.
2. **Resolving a genuine disagreement in the literature.** Li 2024 (summary 10) and Qiu 2025 (summary 14) find
   **inter-subject shift > inter-day shift**; other sources claim temporal drift can exceed between-subject
   variation. We quantify it across many datasets and settle it (scientific question **A4**).
3. ~~**A mean-vs-covariance decomposition of distribution shift.**~~ **VOID AS POSED (2026-07-10).**
   The plan was to contrast the Gaussian-KL mean/cov split on RAW vs GLOBALLY Z-SCORED features. That
   contrast is **not identifiable**: the split is *exactly invariant* to any invertible global affine map
   applied to both subjects, and a global z-score is such a map (verified to 4e-12 relative error). The
   `+1e-3*I` ridge is the only reason the old numbers appeared to differ — it moves them by 49-219 %.
   Only a **per-subject** map changes between-subject divergence. **Replacement (implemented, measurable):**
   contrast `pooled` vs `subject_center` vs `subject_zscore`, null-corrected against a TRIAL-DISJOINT
   within-subject estimation-noise floor (`block_c.E3`). The Yoneda quantity is `mean_share_of_excess`;
   the "why normalisation helps" quantity is `kl_excess_removed_by_subject_center`. See `CORRECTIONS.md`.
4. **Complexity/nonlinear features carry independent, cross-subject-relevant information** — validated
   independently by the winning IDFS feature set (Botros 2025, summary 05: SampEn, Hjorth complexity, MFL) and by
   mRMR rankings (summaries 01, 08). We characterize them across datasets.

---

## 2. Decisions (your doubts 1–4 — recommended approach, now LOCKED)

| # | Question | Decision (recommended, adopted) |
|---|----------|---------------------------------|
| 1 | Which datasets? | **Start with the Paper-1 six**, then expand to all 14. Six = `emaha_db1, fors_emg, grabmyo, ninapro_db1, ninapro_db2, ninapro_db5`. Matches Paper 1 so Module 5's LOSO targets line up. |
| 2 | Module 5 timing | **Modules 1–4 start immediately** (no Paper-1 dependency). **Module 5 is wired now but its predictor is finalized once Paper-1's EMAHA LOSO sweep exists.** Module 5 does **not** block Phases A. |
| 3 | Windowing | **Report Module 1 at a longer window (≈250 ms) for entropy stability** (papers want ≥~200 samples; 250 ms × 1–2 kHz = 250–500 samples), and **keep Track-1's 100 ms/50 % as a secondary sensitivity axis.** Note the trade-off explicitly. |
| 4 | ADL taxonomy | **Also profile EMAHA's FAABOS coarse grouping** alongside the native label scheme (it is our ADL-specific taxonomy and the only dataset with a hierarchical ADL label). Other datasets: native label only. |

Additional locked cross-cutting choices:
- **Normalization for all cross-subject metrics: train-only global z-score** (`semg/splits/splitter.py`
  `Normalizer(mode="global")`). Consensus best across summaries 05, 07, 10, 14; avoids leakage.
- **Scope: healthy subjects only** (per project memory). Do not flag amputee/clinical absence as a gap.
- **Inter-day axis (A4) is answerable only where multiple sessions exist:** of the six, **grabmyo (3 sessions/
  subject)** and **senic (uneven sessions, has position/shift/fatigue metadata)** support it; emaha/fors/ninapro
  are single-session → inter-subject-only. State this honestly per dataset.

---

## 3. The five modules — methods, techniques, exact parameters

Every method below is traced to a reviewed paper. "Cheap" = CPU, closed-form or O(N·features), no training,
except the two kNN/RF estimators in Module 5 which are still light.

### Module 1 — Signal-level characterization  *(per subject / channel / class)*
**Goal:** describe the raw statistical & complexity structure of each dataset's signals.

- **Amplitude / energy:** RMS, MAV, IEMG, VAR, SSI, WL, DASDV, AAC, LOG, **LogRMS**, **Normalized Log Energy
  (NLE)** (LogRMS/NLE = best single features per Abbaspour, summary 12); **active-vs-rest RMS ratio → SNR proxy**
  (generalize the Track-1 rest computation).
- **Distributional shape:** skewness, kurtosis, 75th percentile (summary 12).
- **Spectral (skip on envelope datasets):** mean frequency (MNF), median frequency (MDF), spectral entropy,
  bandwidth, mean/total power (MNP/TTP), band-power ratios. (FD features track fatigue but are weak for class
  separability — summary 04; report the distinction.)
- **Complexity / non-stationarity (the block that makes Module 1 novel):**
  - **Hjorth** activity / mobility / complexity (summary 12, eqs. 16–17).
  - **Higuchi Fractal Dimension (HFD)** (k=1..10) and **Maximum Fractal Length (MFL)** (summary 12, eqs. 10–12).
  - **Sample Entropy (SampEn):** m=2, r=0.2·std (standard; summaries 01, 08 use r=0.25·SD).
  - **Fuzzy Entropy (FEn):** m=2, Chebyshev distance, baseline-removed embedding, membership `exp(−dⁿ/r)`,
    **n=5, r=0.3·std** (summary 01 best operating point; n=1 fails).
  - **Fuzzy ApEn (fApEn):** m=2, Gaussian membership `exp(−d²/r)`, **r=0.25·std** (summaries 02, 03).
  - **Multiscale fuzzy ApEn (MSfApEn):** coarse-grain (moving-average, non-overlap) scales 1–10 →
    **{MED, LS-MED (scales 1–5), HS-MED (scales 6–10)}** (summary 02).
  - **Permutation Entropy (PEn):** τ=1, normalized by ln(d!); **d chosen so `d! ≪ N`** → d=4 or 5 at our window
    lengths (summary 01). PEn is amplitude-blind (cross-subject robust) and ~57× cheaper than FEn.
- **Shared entropy core (one implementation for SampEn/FEn/fApEn/MSfApEn):** m=2, Chebyshev, baseline removal,
  membership exponent parameterized (n∈{1,2,5}); tolerance **`r = k·std`** (amplitude-invariant, k≈0.2–0.3);
  guard near-zero std; require **≥~200 samples/window**; O(N²) → **compute on a subsample of windows per
  (subject,class)** and use the existing joblib parallelism pattern.
- **Deliverable:** per-dataset "data card" (tables + distributions per subject/channel/class), and validation
  that complexity features respond to known physiology (fatigue → lower complexity; summaries 02, 03).

### Module 2 — Class structure & separability  *(per dataset)*
**Goal:** which classes are intrinsically confusable, and which features/feature-families actually separate.

- **De-duplicate the feature bank first** using Phinyomark's 4-group map (summary 04): energy (keep **MAV**),
  complexity (keep **WL**), frequency-info TD (keep **WAMP**), prediction-model (keep **AR4**); drop
  time-dependence (poor). Representative basis = {MAV, WL, WAMP, AR4} + the complexity block (so distances aren't
  dominated by collinear amplitude features). Add **inter-channel correlation + Hjorth** (best compact set,
  summary 12).
- **Filter separability indices:** Fisher discriminant ratio, Davies–Bouldin index, silhouette, kNN-LOO
  (leave-one-out) accuracy proxy.
- **Distance-family separability:** **Mahalanobis-distance separability index (DFSS)** — covariance-aware,
  class-pair min distance (summary 08).
- **MI-family separability:** **symmetrical uncertainty / mutual information (CFSS)** — `SU = 2·I(C;F)/(H(C)+H(F))`,
  max-relevance/min-redundancy (summary 08). Also wrapper (RFE) per the plan's revision.
- **★ Report which family predicts downstream accuracy best** — the plan's open question; the distance and MI
  families disagree on which features matter (summary 08), and mRMR/MI ranks entropy & AR highly (summaries 01, 08).
- **Test ranking stability:** does Phinyomark's within-subject ordering survive **cross-dataset / LOSO**?
  Instability is itself a result (scientific question **L4**).
- **Intrinsic dimensionality:** PCA-95%-variance dim (with `D ≤ N_s−1`, summary 14) **and** nonlinear **TwoNN**
  (Dwivedi motivates nonlinear, summary 06) vs channel count; a large linear-vs-nonlinear gap is a finding.
- **Stats:** non-parametric — **Friedman + Wilcoxon signed-rank + Bonferroni** (summary 14), not ANOVA.

### Module 3 — Distribution shift, quantified  *(the core novelty)*
**Goal:** measure inter-subject vs inter-day vs inter-class shift, decomposed.

- **Distance metrics (cheap, on z-scored features):** Maximum Mean Discrepancy (MMD, RBF kernel), energy
  distance / Wasserstein, and **closed-form Gaussian-KL** `½[tr(Σ₁⁻¹Σ₀)+(μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀)−d+ln(detΣ₁/detΣ₀)]`
  (summary 09) — the KL naturally **splits into a mean term and a covariance term**.
- **★ Mean-vs-covariance decomposition** (summary 11): report the KL's mean component vs covariance component per
  dataset. Expected headline: inter-subject shift is **mean-dominated** → explains why z-score (which equalizes
  first/second moments) helps, and why re-estimating just the mean is a cheap calibration.
- **H-divergence (marginal shift):** `d_H = 2(1−2ε)`, ε = pairwise subject-classifier (RandomForest 20 trees,
  5-fold CV) error → **Hermitian matrix H**, aggregate = rescaled Frobenius norm (summary 07).
- **Inter-subject vs inter-day (A4):** compute the same distances between subjects vs between sessions on
  grabmyo/senic; report the ratio. Expected: **inter-subject > inter-day** (summaries 10, 14) — quantify and
  contrast with the mixed literature.
- **Deliverable:** pairwise distribution-shift **heatmaps** (subject×subject, and session×session where
  available), per-dataset shift scalars, and the mean/covariance decomposition bar figure.

### Module 4 — Channel & sensor analysis
**Goal:** channel redundancy, minimal channel subset, sampling-rate sufficiency.

- **Channel redundancy:** normalized mutual information (NMI) and Pearson correlation between channels → minimal
  channel subset (mRMR / min-redundancy; summaries 08, 12).
- **Channel selection cross-check:** optional **MCCSP** (variance/CSP-based, feature-independent; summary 13) —
  report whether MI-based and variance-based selection agree on the minimal set (mainly for grabmyo / any dense
  montage; sparse datasets rely on NMI/mRMR).
- **Sampling-rate sufficiency:** re-estimate separability (Module 2) after downsampling → is 2 kHz needed, or does
  200–500 Hz suffice? (K2.)
- **Region/placement note (discussion, not reproduced):** class information is region-dependent and gesture-
  category-dependent — single-DoF gestures favour distal wrist, compound/ADL gestures favour mid-forearm/proximal
  elbow (summary 14). Relevant to our ADL focus.

### Module 5 — ★ Subject-difficulty predictor  *(the money result; bridge to Paper 1)*
**Goal:** predict each subject's LOSO accuracy (and calibration-curve position) from cheap statistics of a short
calibration window, *before training*.

- **Target:** each subject's actual LOSO accuracy from Paper-1's EMAHA sweep (and, where available, from a cheap
  Module-2 kNN-LOO proxy on the other datasets so Module 5 isn't blocked).
- **Predictors (cheap, per subject):** distribution distance to the training pool (MMD / H-divergence /
  Gaussian-KL, from Module 3), the subject's mean-vs-pool shift, entropy/complexity summaries (Module 1), SNR
  proxy, intrinsic-dim.
- **Method (Albuquerque backbone, summary 07):** build the **conditional-shift disparity matrix** (pairwise kNN
  label-disagreement → rescaled Frobenius norm) and **marginal-shift H-divergence matrix**; regress the LOSO
  accuracy / generalization gap on each subject's row-aggregate. **If a cheap statistic predicts who the model
  fails on → the headline result** that unifies the two papers.
- **Calibration-curve framing (Botros, summary 05):** frame difficulty as "where on the zero→one→few-shot curve a
  subject lands," aligned with Paper-1's `--k_shot` sweep so both papers share one x-axis. Include a cheap
  **Adaptive-LDA (μ̃=τμ_cal+(1−τ)μ_src, Σ̃=λΣ_cal+(1−λ)Σ_src, τ=0.75, λ=0.9)** classical baseline (summaries 05, 11).
- **Caveat (summary 05):** calibration-data amount dominates feature-set choice — the story is the *curve*, not
  feature tuning. Negative result (no predictor works) is still reportable with the statistics tried.

---

## 4. Deliverables
- Per-dataset **data cards** (Module 1 tables + distribution figures; Module 2 separability tables).
- **Cross-dataset comparison matrix** (all six, later 14: subjects, channels, fs, classes, shift scalars,
  intrinsic dim, separability, complexity summaries).
- **Distribution-shift heatmaps** (subject×subject and session×session) + the **mean-vs-covariance decomposition**.
- **Channel-redundancy / minimal-subset** tables + sampling-rate-sufficiency curves.
- **Subject-difficulty correlation plot** (cheap statistic vs LOSO accuracy) — the money figure.

---

## 5. Build plan (Python) — package layout & phases

```
experiments-data/PROFILE/
  profile/                     # the CPU-only analysis package (self-contained, reuses semg loaders)
    __init__.py
    config.py                  # six datasets, window params, entropy params, feature lists, paths
    features_extra.py          # numpy complexity/nonlinear + classical features on (B,C,T) windows
    windows.py                 # build a (meta + feature) frame via semg build_window_index + h5, z-scored,
                               #   with per-(subject,class) subsampling for the O(N^2) entropy features
    module1_signal.py          # signal-level characterization -> per-dataset data card
    module2_separability.py    # Fisher/DBI/silhouette/kNN-LOO + Mahalanobis(DFSS) + MI(CFSS) + intrinsic dim
    module3_shift.py           # MMD / energy / Gaussian-KL(+mean/cov split) / H-divergence + disparity matrix
    module4_channels.py        # NMI/correlation redundancy, mRMR subset, downsample sufficiency, (MCCSP opt)
    module5_difficulty.py      # difficulty predictor (regress LOSO acc on cheap stats) + Adaptive-LDA baseline
    stats.py                   # Friedman/Wilcoxon/Bonferroni helpers
    datacards.py               # assemble per-dataset cards + cross-dataset matrix (to xlsx/parquet)
    viz.py                     # heatmaps, decomposition bars, difficulty scatter (matplotlib)
  run_profile.py               # CLI: --module {1,2,3,4,5,all} --datasets ... --window-ms ...
  results/                     # outputs (tables, figures) — created on run
  ROADMAP.md  paper-summaries/  data-science-related-papers/
```

**Design rules:** reuse `semg.data.window_index.build_window_index` (identical leakage-safe windows),
`semg.splits.splitter.Normalizer(mode="global")` (train-only z-score), and `ensure_features` for classical
Track-1 features where convenient. All outputs are atomic files (parquet/xlsx/png) so partial runs are safe.
Every module writes a small JSON/parquet of numbers + a figure.

**Execution phases:**
- **Phase A (now, CPU, parallel to Paper-1 GPU):** Modules 1–4 on the six datasets. Produces data cards,
  cross-dataset matrix, shift heatmaps, channel/intrinsic-dim tables.
- **Phase B:** Module 5 — wire predictors on Phase-A statistics now; finalize the regression once Paper-1's
  EMAHA LOSO per-subject accuracies land. Then the difficulty scatter.
- **Phase C (optional):** expand from six to all 14 datasets; add MSfApEn multiscale figures, MCCSP cross-check,
  sampling-rate sweep.

---

## 6. Limitations & open queries (state these honestly in the paper)
- **Inter-day (A4) is limited:** only grabmyo (3 sessions) and senic (uneven) support it among the six; the rest
  are single-session. The A4 claim is therefore across a *subset*, not all datasets.
- **Healthy only:** no amputee/clinical validation (scope choice, not a gap to apologize for).
- **Entropy cost & subsampling:** O(N²) entropy is computed on a per-(subject,class) subsample; report the
  subsample size and check stability. `r=k·std` is ill-defined for near-silent windows — those are guarded/flagged.
- **Module 5 depends on Paper-1 LOSO targets** for the strongest version; the kNN-LOO proxy target is a fallback.
- **Distances are feature-representation-dependent** — we fix the representation (de-duplicated basis + z-score)
  and report sensitivity to it.
- **Intrinsic-dim / minimal-channel numbers are montage- and method-conditional** (summary 06) — report method
  alongside every number; do not claim a single canonical minimal channel set.
- **We are not doing streaming drift detection** (shown ineffective for EMG, summary 09) nor the heavy Bayesian
  GCM / MRD models (summaries 06, 11) — we borrow their *conclusions/decompositions*, not their algorithms.
- **Open query:** which distance metric (MMD vs H-divergence vs Gaussian-KL) best predicts LOSO accuracy? This is
  an empirical output of Module 5, not assumed.

---

## 7. Scientific questions this paper targets (from the review protocol)
- **A4** inter-day vs inter-subject variability — Module 3 (answered, quantified across datasets).
- **A6** does cross-dataset transfer depend on electrodes / taxonomy / population — Module 3 groundwork.
- **A7/A8/A9** calibration-free difficulty, few-shot curve, subject-difficulty predictor — Module 5.
- **C-series** features: handcrafted vs complexity, which set separates, dimensionality reduction — Modules 1, 2.
- **K1/K2** minimal channels, sampling-rate sufficiency — Module 4.
- **L4/L5** is feature/method ranking stable across datasets / does it survive LOSO — Module 2.

---

## 8. STATUS LOG (append-only)
- **2026-07-09 — Roadmap FINAL after full review of all 14 supplied papers** (6 original + 8 added). Per-paper
  summaries written (`paper-summaries/01`–`14` + INDEX). Decisions 1–4 locked (six datasets first; Modules 1–4
  start now, Module 5 wired to Paper-1 LOSO; Module 1 at ~250 ms window; profile FAABOS too). Methods per module
  fixed with exact params + paper provenance.
- **2026-07-09 — `dsprofile/` package BUILT + all 5 modules SMOKE-TESTED GREEN on emaha_db1.** Package (renamed
  from `profile/` to avoid the stdlib clash): `config, features_extra, windows, module1..5, stats, datacards,
  viz` + `run_profile.py` CLI. Reuses semg `build_window_index` + `Normalizer(mode="global")`. Entropy inner
  loop vectorised (full pairwise Chebyshev matrix); MSfApEn made opt-in (`COMPUTE_MSFAPEN`); entropy windows
  decimated to `ENTROPY_MAX_SAMPLES=400`.
  - **Early real signal (emaha_db1, smoke caps):** Module 5 difficulty predictor — `kl_mean_to_pool` vs LOSO
    accuracy **Pearson r=−0.56, p=0.004** (MMD r=−0.42, p=0.035): a cheap distribution statistic significantly
    predicts which subjects the model fails on = Paper-2's thesis, confirmed on dataset #1. Module 3 inter-subject
    MMD 0.34; Module 2 Fisher/silhouette/kNN-LOO/TwoNN all sane; Module 4 5-ch NMI 0.05 (low redundancy).
  - **NEXT: full (non-smoke) Phase-A run of Modules 1–4 on the six datasets** (`python run_profile.py --module
    1234 --datasets six`), then figures; Module 5 target upgrades from self-LDA-LOSO proxy to Paper-1 DL LOSO
    once the EMAHA sweep lands. Usage: `run_profile.py --module {1..5|all} --datasets {six|all|<names>} [--smoke]
    [--figures]`.

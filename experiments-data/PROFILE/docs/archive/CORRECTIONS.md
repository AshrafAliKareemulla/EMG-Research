# PROFILE — 2026-07-10 correctness audit & fixes

Audit of the completed Phase-1 + Phase-2 `results/` (309 files) against the code that produced
them. **Nothing crashed**; every run completed and wrote well-formed JSON. The defects are in
what the code *computed*. Each fix below is pinned by a ground-truth regression test in
`tests/test_corrections.py` (32 checks).

**All prior numbers in `RESULTS_REVIEW.md`, `PHASE2_PLAN.md` status log and
`cross_dataset_matrix.xlsx` are superseded.** Do not quote them.

> **READ THE 2026-07-10 (second pass) SECTION AT THE BOTTOM BEFORE QUOTING §2 OR §3 BELOW.**
> Running the fixed code showed that defect **#2 ("every kNN accuracy was leaked")** was real in
> principle but **immaterial in practice** (0.4213 -> 0.4193), and that the H-divergence claim was
> **overstated** — high d_H between *different* subjects is genuine. It also exposed a **blocker in
> the fix itself** (`subject_center` built on raw-scale features). Both are corrected at the bottom.
> Current status lives in **`STATE.md`**.

---

## The three that change the paper's claims

### 1. E3 (mean-vs-covariance) measured nothing — ROADMAP novelty pillar #3 is void as posed

The Gaussian-KL split
`mean = ½(μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀)`, `cov = ½[tr(Σ₁⁻¹Σ₀) − d + ln(detΣ₁/detΣ₀)]`
is **exactly invariant** under any invertible global affine map `x ↦ Ax + b` applied to both
subjects. A global z-score is such a map. Verified: relative difference 7e-10 … 3e-5 across all
real dataset shapes (d=35…196, n=400).

The old E3 compared raw vs globally-z-scored features and reported ratios of 0.093 vs 0.250
(emaha_db1), 0.235 vs 0.238 (grabmyo). Those differences came **entirely from the `+1e-3·I`
ridge**, which is not scale-invariant and bites differently when raw feature scales span 10⁶.
With the ridge, the same comparison moves by 49 %–219 %.

Why the numbers looked *nearly* identical rather than wildly different: under a signal-level
global scale `x → a·x`, six of the seven `REPR_BASIS` features transform as an exact diagonal
affine map (measured, not assumed) — MAV/WL/RMS scale by `a`, HJ_MOB/HJ_COM are invariant, MFL
shifts by `log₁₀ a`. Only WAMP (threshold-based) is nonlinear.

**Fix** (`block_c.meancov_decomposition`): work in feature space; assert the invariance
numerically; then measure what actually changes between-subject divergence — a **per-subject**
map. Three representations (`pooled`, `subject_center`, `subject_zscore`), `ridge=0` sample
covariance, PCA truncation to keep n/d ≥ 10, and a **null floor** from within-subject
split-halves at matched n.

The null floor is not optional: on two *identical* distributions the uncorrected statistic
reports a mean share of 0.07, i.e. "covariance dominates", from estimation noise alone. That is
what the original E3 was reporting.

New headline quantity: `representations.pooled.mean_share_of_excess` (Yoneda's claim), and
`kl_excess_removed_by_subject_center` (why per-subject normalisation helps).

### 2. Every kNN accuracy was leaked

`module2.knn_loo` (neither leave-one-out nor leakage-safe), `block_b.per_class_difficulty`,
`block_d._knn_acc`, `faabos` all did:

```python
idx = rng.choice(len(X), max_n, replace=False)   # shuffles row order
cross_val_score(KNeighborsClassifier(5), X[idx], y[idx], cv=5)
```

Windows overlap 50 %, so a test window's nearest neighbour is routinely its own sibling from the
same trial, sitting in the training fold. The split was also subject-pooled. On a synthetic frame
with strong trial identity and a weak class signal the old protocol scores **1.000** where the
truth is **0.306**.

This contaminated: `module2.knn_loo_acc` (emaha_db1: 0.42 vs a real LOSO of 0.25), Block B's
hard/easy class ranking, Block D's K1 (minimal channels) and K2 (sampling rate) answers, the
FAABOS ADL result, and — via `knn_loo` — the meta headline.

**Fix** (`dsprofile/cv.py`): `knn_trial_cv` (GroupKFold on `(subject, session, label, repetition)`)
and `knn_loso` (GroupKFold on subject). Both are reported everywhere, so the within-vs-cross gap
is visible. Standardisation is fit on the training fold only. Subsampling now selects whole
trials, never rows.

### 3. The meta headline was circular

`what_makes_hard` reported `knn_loo` predicting dataset difficulty at ρ=0.93. But `knn_loo` is a
classifier accuracy and the target `loso_acc` is a classifier accuracy — "accuracy predicts
accuracy", and a leaked accuracy at that. 14 predictors were tested against one target on n=14
points with **no correction**, though POST_RESULTS_PLAN Stage 1 required FDR.

**Fix**: accuracy-valued predictors are flagged `circular: true` and excluded from the ranking
and from FDR; Benjamini–Hochberg (`stats.fdr_bh`, matches the textbook definition) is applied to
the rest.

---

## The rest

| # | Defect | Fix |
|---|--------|-----|
| 4 | **Winner's curse.** `best_predictor` = best-of-4 per dataset → 56 uncorrected tests. "9/14 significant" included `senic` at r=**+0.70**, the wrong sign. | `config.PRIMARY_PREDICTOR = "mmd_to_pool"` fixed a priori; all four still reported; per-dataset FDR in `meta`. Post-hoc pick kept as `best_predictor_posthoc` with a warning. |
| 5 | **`combined_linear_r2` was in-sample.** emaha_db1 R²=0.608 from 4 predictors on n=25 while the best single r=−0.43 (r²=0.18). | `combined_cv_r2` = leave-one-subject-out out-of-sample R². Negative values are legitimate and reported. |
| 6 | **E6 had no anti-aliasing.** `wins[:, :, ::q]`. A 400 Hz tone folds to 100 Hz at full power. Also decimated 200 Hz sets to 44–50 Hz (11-sample windows). | `scipy.signal.decimate` (zero-phase FIR). Guards: native fs ≥ 1000 Hz, effective fs ≥ 500 Hz. Untestable datasets are skipped, not reported. Cache key bumped to `_dec{q}v2`. |
| 7 | **Entropy computed on 25–50 sample windows.** `ENT["min_samples"]=200` was defined and never used. SampEn NaN'd; FuzzyEn/fApEn/PermEn/HFD returned meaningless finite numbers on `ninapro_db1` (100 Hz), `myobit` (176 Hz), `ninapro_db5`/`senic` (200 Hz). | Guard enforced in `features_extra.slow_features`; masked on read (`windows.mask_invalid_complexity`) so the expensive `complex` caches need no rebuild. `block_a` no longer `nan_to_num`s them into **zeros**. |
| 8 | **Actionability was a null result presented as a win.** `guided_advantage` 0.0002–0.0087 (0.02–0.9 pp), 3/14 negative, no CI, and the *oracle* ceiling is ~1 pp — no policy can help. The premise "predicted-hard users gain most from calibration" was never tested. | Reports `oracle_ceiling`, `fraction_of_ceiling_captured`, an empirical p vs shuffled orderings, and `mmd_vs_calibration_gain` — the premise, now tested. |
| 9 | **senic verdict unsupported.** `PHASE2_PLAN` claims "confound confirmed (loso & mmd both fall with session count)"; the probe's own criterion is both *rising*, neither correlation was significant (p=0.14, 0.11), and `n_trials` was a deterministic function of `n_sessions` (identical r to 16 dp) — one variable tested twice. | Collinearity detected explicitly; partial correlation controlling for `n_sessions`; an explicit `verdict` field. **The sign reversal remains unexplained; senic is an outlier, not a solved case.** |
| 10 | **Transfer matrix erased what it measured.** Each dataset was z-scored independently before the MMD, deleting the between-dataset location/scale difference. Distances 0.002–0.108 were shape-only. No transfer accuracy was ever measured. | Native-scale (`normalize="none"`) frames + a single pooled scaler → `compatibility_mmd`; the old view kept as `shape_only_mmd`. Marked `validated: false`. |
| 11 | **k=14 datasets are not independent.** grabmyo_flow_static/dynamic are one cohort; the four EMAHA sets one; ninapro_db4/db5 one. → **9 cohorts.** I²=0.79. SDI's "leave-one-dataset-out" leaked the cohort. | `config.COHORTS`; SDI validates **leave-one-cohort-out**; meta adds `pooled_one_per_cohort` and `pooled_without_outliers`. |
| 12 | `Normalizer` fit on all subjects incl. the held-out one. (Minor: LDA is affine-invariant, so LOSO accuracy is unaffected — but the roadmap promised train-only.) | Train-fold-only standardisation inside every CV loop. |
| 13 | `RESULTS_REVIEW.md` claimed "zero NaNs"; there were **88**. | `validate_results.py` distinguishes sanctioned NaN (`mean_share_of_excess` when no shift is detectable) from bugs. |
| 14 | `datacards.py` line 27: `n_subjects = m2.get("n_classes") and m3.get("n_subjects")`. | Rewritten against the corrected keys. |

---

---

## Found by adversarial review of the fixes themselves (2026-07-10, second pass)

An independent reviewer was given the plan, the audit and the diff. It confirmed C1–C4 but found
three defects — two of them in code I had just written. All three are now fixed and pinned by
tests in `tests/test_corrections.py` (`test_hdiv_leak`, `test_subsample_keeps_all_groups`,
`test_null_floor_trial_disjoint`).

### R1 (BLOCKER) — the E3 null floor was split by ROW, not by TRIAL

`_null_terms` used `rng.permutation(idx)`, scattering a trial's 50 %-overlapping windows across
both halves. The halves therefore *shared trials* and looked far more alike than two independent
samples, so the floor was **~14× too small** (measured: row-permuted 0.32 vs trial-disjoint 4.29
on trial-structured data with zero true shift). The "excess" above it was mostly noise —
reintroducing the exact artifact the null exists to remove. `shift_detectable` fired spuriously.

**Fix**: `_matched_halves` splits each subject's **trials** into two disjoint halves and draws
the same number of rows from each. Between-subject pairs use each subject's first half, so both
sides of the subtraction have identical (trials, rows) budgets — the effective sample size of
correlated windows is set by the trial count, not the row count. Verified: zero-shift
trial-structured data now returns `shift_detectable: false`; mean- and cov-dominated ground
truths are recovered exactly.

### R2 (MAJOR) — `h_divergence` was still leaked, and I had not noticed

`module3_shift.h_divergence` did `_sample` (which shuffles whenever `len > n`, i.e. always on the
real call path) then `cross_val_score(RandomForest, cv=5)` — byte-for-byte the idiom fixed
everywhere else. Trial identity is perfectly predictive of which group a window belongs to, so
the RF memorised trial fingerprints. Measured on two groups from an **identical** distribution
(true d_H = 0): **1.975** against a maximum of 2.0. Trial-grouped: **0.000**.

This is the origin of Phase-1's "H-divergence uniformly high (0.72–0.98)", which
`RESULTS_REVIEW.md` recorded as a *finding*. It is an artifact. It contaminated `hdiv_to_pool`
(an SDI predictor), `is_hdiv` (a meta predictor), and the post-hoc "best predictor" for
`emaha_db5`, `myobit` and `senic`.

**Fix**: `h_divergence` takes `groups_a`/`groups_b` (trial ids) and uses `GroupKFold`; it warns
loudly if called without them; the result is clamped at 0. Trial ids are threaded through
`_grouped` → `_pair_metrics` (module3) and `subject_shift_stats` (module5), and all four call
sites (`module5`, `robust_difficulty`, `actionability`, `faabos`) now pass them.

**`mmd_rbf` — the a-priori PRIMARY predictor — is a population two-sample statistic with no
cross-validation and is therefore unaffected. The headline stands.**

### R3 (MAJOR) — my own `_subsample_by_group` collapsed `knn_loso` to one subject

I had subsampled *whole groups* until a row budget filled, on the false premise that row-level
subsampling could re-introduce the fold leak. It cannot: `GroupKFold` assigns folds by group id
regardless. With `subjects` as the grouping variable and `max_n=4000`, it retained **1–2
subjects** (ninapro_db2: 1 → `n_splits=1` → NaN), so `knn_loso_acc`, `within_minus_cross` and
Block D's `min_channels_for_95pct_loso` would have trained on a single subject.

**Fix**: proportional row-level subsampling with a floor of one row per group — every subject and
every class is retained. Also renamed to `knn_subject_cv`, because subject-grouped 5-fold is
*not* leave-one-subject-out; `knn_loso` remains as an alias.

### Also adopted from the review

* `test_math`'s "H-div(same) ~ 0" used iid rows and **passed even with the leak present**. It now
  exercises trial-clustered data and asserts both the leaked and the honest value.
* BH-FDR across per-*predictor* tests on the same 14 datasets is only *approximately* valid
  (dependent tests). The per-*dataset* meta FDR is properly valid (disjoint subject samples).
* The EMAHA 4-set cohort merge rests on a shared lab, not verified shared subjects. It
  *over*-merges, which deflates the evidence — conservative, not an inflation risk.
* `grabmyo_flow_static`/`_dynamic` = the same 20 people (confirmed in the dataset's `STATE.md`);
  `grabmyo` (subjects 1–43) vs `grabmyo_flow` (44–63) = different people. Cohort map is correct.

---

## What survives unchanged

* **A4 (E2)**: inter-subject > inter-day, ratios 2.4–3.6×. Now with Mann-Whitney + rank-biserial
  and honest caveats (the grabmyo variants are one cohort; senic's "sessions" are electrode-shift
  conditions, not days, and most senic subjects have one session). Rests on ~2 independent cohorts.
* **Classifier-agnostic difficulty** (`robust_difficulty`): LDA/SVM/RF agree on who is hard
  (Spearman 0.47–0.94). Untouched by every defect above. Currently undersold.
* **ninapro_db1 (r=−0.77, n=27) and ninapro_db2 (r=−0.73, n=40)** difficulty correlations survive
  any correction.
* **Low intrinsic dimensionality** (TwoNN 6.2–13.7 regardless of channel count) and universally
  negative silhouette.
* `synth` behaves as a control (r=−0.98). `DROP_REST=True` is correct.

## Open question the numbers now force

The difficulty predictor works best exactly where LOSO accuracy is near the floor
(ninapro_db1 12 % on 53 classes, db2 14 %) and fails where accuracy is healthy
(grabmyo 0.70 → r=+0.03; emaha_db4 → r=−0.01). Whether this is difficulty prediction or a
floor effect is not settled by the current experiments and must be addressed before the write-up.

## How to re-run

```bash
python tests/test_corrections.py          # 32 ground-truth checks
python invalidate_stale.py --dry-run      # then without --dry-run
python run_profile.py --module 12345 --datasets all --jobs 8
python run_phase2.py  --exp all --datasets all --jobs 8
python validate_results.py                # gates the write-up
```

---

## 2026-07-10 (second pass) — reading the CORRECTED results

Running the fixed code and reading its output exposed **one blocker in the new code** and three
statistical repairs, plus **two overstatements in this document**. See `STATE.md` for the full
table. Summary:

### Blocker (in my own fix)
`_repr_matrix` built `subject_center` on RAW-scale features, then PCA'd them. `pooled` was
z-scored first and `subject_zscore` is scale-free, so only `subject_center` got a rank-deficient
basis: condition number 12 -> 3.9e7, and with `ridge=0` the `tr(S1^-1 S0)` term exploded
(ninapro_db2 `sc_cov` = 2.8e9 against a `pooled` cov of 3.1e3), driving
`kl_excess_removed_by_subject_center` to **-597703**. Fixed: the global z-score now precedes every
per-subject map. **`mean_share_of_excess` and `shift_detectable` were never affected** — they live
in the already-z-scored `pooled` arm.

### Statistical repairs
* **A4 p-values were pseudo-replicated** (Mann-Whitney over 917 pairs from 14 subjects -> senic
  p = 3.9e-108). Replaced with a subject-level paired Wilcoxon. The effect (rank-biserial 0.80-0.89)
  was never in doubt; only the p-value was fiction.
* **`difficulty_cv_r2` = -278 / -198** on the n=10 datasets — LOO-CV of a 4-predictor OLS on 9
  training points. Suppressed below 5 subjects per predictor.
* **`min_channels_95pct_loso` = 2** for emaha_db5 (full LOSO acc 0.142 vs chance 0.100). Now
  chance-corrected, and returns `None` when the model is not meaningfully above chance.

### ⚠ Two claims ABOVE that this document OVERSTATED

1. **"Every kNN accuracy was leaked" — real in principle, immaterial in practice.** Old leaky
   `knn_loo` = 0.4213; new trial-grouped = 0.4193. A 0.002 difference. `MAX_WINDOWS_PER_CLASS=200`
   had already thinned ~245 trials/class to a few windows each, so the near-duplicates were largely
   gone before CV ran. The synthetic worst case (1.000 vs 0.306) overstated the real-data effect.
   **The large number was never leakage — it was within-subject vs cross-subject conflation**
   (0.419 -> 0.159), which is a different and more interesting sin.

2. **"H-divergence uniformly high was an artifact" — too strong.** The fix is active (senic's
   minimum pairwise d_H = 0.25, myobit's = 0.735; under the leak nothing could be that low). But
   between genuinely *different* people d_H really does run 1.3-1.95, so Phase-1's *qualitative*
   claim survives. What was an artifact is that the estimator returned ~1.97 even for two halves of
   the **same** distribution. A `hdiv_within_subject_null` diagnostic now measures exactly that.
   Relatedly, the `hdiv_frob > 0.9` validator check was measuring `_frob` — RMS of
   `offdiag/offdiag.max()`, a **uniformity** statistic, not a magnitude. Replaced.

### The open question the corrected numbers force
The floor effect is **confirmed as a real association** (|r| vs mean LOSO accuracy = -0.593,
p=0.033; partial controlling n_subjects = -0.632, p=0.020; *not* a power artifact, |r| vs
n_subjects = -0.026). But it is **not separable from class count** at n=13 (partials -0.44 vs +0.32,
both n.s.). An intervention is required — see `STATE.md` §5 and `NEXT_STEPS.md` Stage 2.

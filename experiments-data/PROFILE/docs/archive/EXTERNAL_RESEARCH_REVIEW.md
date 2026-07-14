# EXTERNAL RESEARCH REVIEW — PROFILE (Paper 2)
## Data-Science Characterization of sEMG ADL Datasets

**Review date:** 2026-07-11
**Revision:** 2 (2026-07-11) — the v1 review was independently cross-checked by a second agent
*against the code and committed numbers* (not against the prose); every load-bearing finding was
confirmed (the `floor_effect.json` self-contradiction, the missing two-term model, the `su=2*mi/hc`
bug, `_frob` as a uniformity statistic, fixed MMD γ, seed-blind cache, absent exp_D/E artifacts,
A/B/C complete). This revision (a) refines two wordings per that cross-check (Modules 1–4 framing;
proxy-target severity), (b) records the process-risk pattern behind the blocker (§9 R0), and
(c) **adds §14 — a detailed, execution-ready experimental program**, since the decision has been made
to invest in more experiments rather than only trim claims.
**Reviewer stance:** External, adversarial, evidence-based — as for a top-tier ML / biomedical-AI venue.
**Scope reviewed:** the planning corpus (`STATE.md`, `ROADMAP.md`, `CORRECTIONS.md`, `NEXT_STEPS.md`,
`PHASE2_PLAN.md`, `POST_RESULTS_PLAN.md`, `RESULTS_REVIEW.md`, `FILES.md`), the analysis package
(`dsprofile/*`), the five add-on experiments + `floor_effect.py`, the committed results
(`results/experiments/`, `results/meta/`, `results/block_c/`, `results/floor_effect/`, and the
per-module JSONs), the test suite (`tests/*`), and the paper summaries (`paper-summaries/*`).
Equations were verified line-by-line where feasible.

**How to read this document.** Findings are tagged **[OBS]** when grounded in the project artifacts
(file paths / committed numbers are cited) and **[REC]** when they are my recommendation. Nothing in
the codebase was modified during this review.

---

## Table of contents

- [0. Headline judgment](#0-headline-judgment)
- [1. Research Direction](#1-research-direction)
- [2. Dataset Design](#2-dataset-design)
- [3. Signal-Processing Pipeline](#3-signal-processing-pipeline)
- [4. Machine-Learning Pipeline & the Equations](#4-machine-learning-pipeline--the-equations)
- [5. Experimental Design](#5-experimental-design)
- [6. Current Experiment Results — trends, not silos](#6-current-experiment-results--trends-not-silos)
- [7. Reproducibility](#7-reproducibility)
- [8. Scalability](#8-scalability)
- [9. Research Risks (prioritized)](#9-research-risks-prioritized-by-impact-on-validity)
- [10. Comparison with Current sEMG Literature](#10-comparison-with-current-semg-literature-last-35-yr)
- [11. Research Gaps](#11-research-gaps-why--impact--priority--next-experiment)
- [12. Roadmap Review](#12-roadmap-review)
- [13. Final Assessment](#13-final-assessment)
- [14. Detailed Experimental Program — novelty, fixes, ground-truth checks](#14-detailed-experimental-program-to-support-a-publishable-paper)

---

## 0. Headline judgment

This is an **unusually self-aware, methodologically mature** project — well above the median for a
BTech-honors track and competitive on rigor with published sEMG-characterization work. The internal
audit trail (`CORRECTIONS.md` → adversarial re-review → post-results repairs) is exemplary; several
fixes (trial-grouped CV, the affine-invariance proof, pseudo-replication → subject-level Wilcoxon,
cohort-aware pooling, random-effects meta-analysis) are exactly what a good reviewer would demand,
done *before* the reviewer asked.

**But** the project's own gating item — the **floor-effect confound**, which `NEXT_STEPS.md` states in
bold "BLOCKS THE WRITE-UP" — is marked "RESOLVED" in `STATE.md` on the basis of an analysis that
(a) is **not in the committed code or results**, (b) **contradicts the committed per-dataset
evidence**, and (c) reintroduces the very **pseudo-replication** the team fixed elsewhere. That, plus
a **shared-representation coupling** between the predictor and its target, are the two issues that
stand between this and a defensible publication. Neither is fatal, and neither requires collecting new
data — both, and every other gap in this report, are turned into a concrete, execution-ready experiment
in **§14**. Underneath the blocker is a *process* pattern worth naming (§9 R0): the tracking docs
narrate resolution slightly ahead of the evidence, and a "RESOLVED" label sitting on self-contradictory
data is exactly what makes a reviewer distrust the parts that are genuinely airtight.

---

## 1. Research Direction

**[OBS] The thesis is sound and well-hedged.** "Characterize sEMG ADL datasets *before* training; find
cheap calibration-window statistics that predict cross-subject difficulty" is a genuine, publishable
contribution with a deliberate risk hedge (CPU-only, no accuracy axis, runs parallel to Paper 1). The
pivot documented in `NEXT_STEPS.md`/`STATE.md` — from four novelty hooks down to a **methodological
contribution** (KL affine-invariance proof) + a **leakage audit** + a **difficulty predictor** — is
the right call and makes the paper narrower but far more defensible.

**[OBS] Questionable assumptions that survive into the current plan:**

1. **The self-LDA-LOSO proxy target is treated as interchangeable with the real DL LOSO target.** The
   entire "difficulty predictor" edifice (Module 5, SDI, robust_difficulty, floor_effect, exp
   A/B/C/D/E) regresses cheap statistics on a *linear-classifier* accuracy computed from the *same
   7-feature basis*. `STATE.md` §6 correctly flags the DL swap as "the single most valuable scientific
   upgrade," but every current headline number rests on the proxy. This is the deepest assumption in
   the project.
2. **"Distribution shift" ≈ amplitude shift.** `REPR_BASIS = [MAV, WL, WAMP, RMS, HJ_MOB, HJ_COM, MFL]`
   (`config.py:119`) is dominated by amplitude features (MAV/WL/RMS/MFL all scale with contraction
   strength). After a *global* z-score, per-subject amplitude differences remain, so the MMD/KL "shift"
   is substantially a **contraction-amplitude** difference between people. This should be stated
   explicitly — it reframes what the "shift predicts difficulty" result actually means.

**[OBS] Prioritization is mostly right,** with one inversion: the roadmap spends heavily on descriptive
modules (1–4) whose findings mostly did **not** survive (silhouette is the *only* dataset-level
predictor surviving FDR; `is_mmd/is_hdiv/is_klmean/is_klcov/twonn/pca95/sampen/hfd` all fail —
`meta.json:52–150`), while the load-bearing risk (floor effect) got a single 4-dataset probe.

**[REC]** Re-center the paper on the three results that are actually strong and novel (invariance
proof; within>cross gap; recalibration actionability from exp_B) and **reposition Modules 1–4 as a
deliberate, labelled *dataset-atlas* contribution.** (Nuance adopted from the independent cross-check:
these modules are not *wrong* — "no dataset-level property except silhouette predicts hardness, after
FDR" is a legitimate **negative finding**, not a defect. Report it as a result rather than burying it;
this is an editorial reframing, not a repair.)

---

## 2. Dataset Design

**[OBS] Strengths:** 14 datasets is a large, genuinely multi-corpus panel for this literature; the
**cohort structure is correctly modeled** (`config.COHORTS`, `meta.json:4–19`) as **9 independent
cohorts**, and every pooled statistic is reported at both k=14 and k=9
(`meta_analysis.pooled_one_per_cohort`). This is better than most published sEMG meta-work.

**[OBS] Limitations, by severity:**

- **The multi-session (A4/temporal) axis rests on ~2 cohorts, both GrabMyo-lab.** `exp_C` reports
  "3/3 true-day datasets" but all three are the GrabMyo family (`exp_C_summary.json`); `senic`'s
  "sessions" are electrode-shift conditions, not days (correctly excluded). So A4 and the entire
  temporal extension are effectively **one acquisition setup**. Honestly flagged in `STATE.md` §7, but
  it caps the generality of two headline claims.
- **Class balance/imbalance is measured but not controlled.** exp_B reports
  `median_class_imbalance_ratio` per dataset but the difficulty analyses don't stratify by it.
- **Label quality, electrode placement, and montage heterogeneity are unaudited.** The atlas explicitly
  "mixes device, fs and channel count with population differences" (`meta.json:1005`) — so cluster
  structure is confounded by acquisition, not just population.
- **Healthy-only** is a scope choice, correctly not apologized for.

**[OBS] Bias sources:** amplitude-dominated basis (§1); global-normalization fit on *all* subjects
including held-out ones (`windows.py:84`, defect #12 — mitigated because LDA is affine-invariant, but
the shift metrics are computed post-global-normalization so cross-dataset comparability is
normalization-conditional); biased-MMD sample-size dependence (§3).

---

## 3. Signal-Processing Pipeline

**[OBS] Filtering / windowing / segmentation are leakage-safe and well-reasoned.** `build_window_index`
+ 250 ms/50% is defensible; per-window DC-detrend for non-envelope data (`windows.py:125`); envelope
datasets correctly skip spectral features; the E6 anti-alias fix (`scipy.signal.decimate`, zero-phase
FIR — `windows.py:176`) is correct and the cache is version-bumped (`_dec{q}v2`).

**[OBS] Weak choices / gaps:**

- **The entropy-feature decimation to `ENTROPY_MAX_SAMPLES=400` uses naive `x[..., ::step]`**
  (`features_extra.py:253`), *without* the anti-alias filter that the same team correctly added for E6.
  Inconsistent; for 2 kHz→250-sample entropy windows this can alias into the entropy estimate. Low
  impact (entropy features aren't in the headline) but it's an internal inconsistency.
- **MMD kernel bandwidth is fixed at `gamma = 1/d`** (`module3_shift.py:37`), sklearn's default,
  **not** the median heuristic. MMD magnitude is bandwidth-sensitive; a fixed γ makes cross-dataset MMD
  comparisons (different d) not strictly comparable. **[REC]** use the median-pairwise-distance
  heuristic and report sensitivity.
- **The reported shift scalars (`inter_subject.mmd_frob`, `hdiv_frob`, `kl_*_frob`) use `_frob`, a
  *uniformity* statistic** — `off / off.max()` then RMS (`module3_shift.py:173`). The team itself
  identified this normalization as "uniformity, not magnitude" for the validator (`CORRECTIONS.md` F6)
  but **still uses it for the reported per-dataset shift scalars**. An all-equal matrix scores 1.0
  regardless of absolute shift; two datasets with very different absolute MMD but similar spread get
  similar `mmd_frob`. The raw matrices are saved (`.npz`), so this is recoverable, but the JSON
  headline scalars are scale-free and easy to misread. **[REC]** report a magnitude-preserving
  aggregate (mean off-diagonal) alongside.

**[OBS] Feature math is correct** (verified against `test_math.py` and by hand): MAV/RMS/WL/VAR/SSI/
DASDV/AAC/LOG/MFL, Hjorth (activity/mobility/complexity), spectral MNF/MDF/SENT, SampEn (brute-force
validated), PermEn, Higuchi FD (white-noise→2, sine→1). See §4 for the one nuance.

---

## 4. Machine-Learning Pipeline & the Equations

I checked every load-bearing equation. **Most are correct.** Findings:

### Correct and validated

- **Gaussian-KL split** (`module3_shift.py:54`):
  `KL(N_A‖N_B) = ½[(μ₁−μ₀)ᵀΣ₁⁻¹(μ₁−μ₀) + tr(Σ₁⁻¹Σ₀) − d + ln(detΣ₁/detΣ₀)]`, correctly split into
  mean/cov terms. The **affine-invariance claim underpinning the methodological contribution is
  mathematically sound** — I verified both terms are separately invariant under `x↦Mx+b`, and the
  committed check confirms it numerically (grabmyo: raw vs global-z agree to 1.1e-8; ridge breaks it by
  0.60 — `results/block_c/grabmyo__block_c.json:38–44`). This is a genuine, correct contribution.
- **H-divergence** `d_H = 2(1−2ε)` (`module3_shift.py:93`) — correct, and the trial-grouped GroupKFold
  leak fix is validated by a dedicated clustered-data test (`test_math.py:218–237`).
- **DerSimonian–Laird random-effects meta-analysis** (`meta.py:147`): Fisher-z, weights `n−3`, Q,
  τ² = max(0,(Q−df)/C), I² — textbook-correct.
- **Benjamini–Hochberg FDR** (`stats.py:21`) — correct step-up with monotonicity enforcement.
- **exp_C fixed-effects within-subject correlation** (`exp_C_cross_session.py:100`) with
  `df = N − n_subjects − 1` — a proper Frisch–Waugh–Lovell estimator with the *correct* degrees of
  freedom. This is sophisticated and right.
- **exp_E permutation test** — correct two-sided, add-one-smoothed.
- Fisher ratio (trace form), Mahalanobis SI (pooled within-class covariance), TwoNN (Facco linear
  fit), PCA-95% — all correct.

### Two math issues

1. **[OBS] MI symmetric-uncertainty is mis-normalized (real bug, low impact).**
   `module2_separability.py:74` computes `su = 2*mi/(hc + 1e-9)` — denominator is **H(C) only**, but the
   docstring and the cited definition (summary 08) are `SU = 2·I(C;F)/(H(C)+H(F))`. Because H(C) is
   constant across features, the reported `mi_su_top` ranking collapses to a **raw-MI ranking** (H(F),
   which differs per feature, is dropped). The **values are wrong** and the ranking is not true SU. It
   feeds only the descriptive `mi_su_top`, and there is **no test** for it. **[REC]** add `H(F)`
   (estimate per-feature entropy) or rename the field to "normalized MI (relevance)."

2. **[OBS] Fuzzy-entropy membership is faithful to the cited papers but not strictly amplitude-invariant.**
   `fuzzy_entropy` uses `μ = exp(−(dⁿ)/tol)`, `tol = r·std` (`features_extra.py:166`). I confirmed this
   **exactly matches** the cited sources (Marri 2020 `exp(−dⁿ/r)`; Xie 2010 `exp(−d²/r)`, `r=k·std` —
   `paper-summaries/01,03`). So it is **not a transcription error.** However, for n≠1 the exponent
   `dⁿ/(r·std)` is **dimensionally inconsistent** (numerator ∝ amplitudeⁿ, denominator ∝ amplitude¹),
   so under `x→ax` the entropy changes by `a^(n−1)` — i.e. FuzzyEn (n=5) and fApEn (n=2) retain a mild
   amplitude dependence that the `r=k·std` parameterization is usually assumed to remove. Global
   z-scoring mitigates but does not eliminate this (per-window std still varies). **Crucially, the
   `test_fuzzyen_reference` test is self-referential** — its brute-force reference uses the *same*
   `exp(−dⁿ/tol)` formula (`test_math.py:146`), so it validates internal consistency, **not** the
   invariance property, and it exercises n=2 not the n=5 operating point. Impact is **low** (entropy
   features are not in `REPR_BASIS`, and `sampen`/`hfd` failed FDR as dataset predictors), but the paper
   should not claim these features are amplitude-invariant. **[REC]** either switch to `exp(−(d/tol)ⁿ)`
   or add an explicit invariance unit test and report the residual sensitivity.

**[OBS] Validation methodology is otherwise strong:** trial-grouped CV vs subject-grouped CV are both
reported (the `within_minus_cross` gap is a legitimate headline), train-fold-only standardization
inside every loop, honest naming (`knn_subject_cv` is 5-fold subject-grouped, *not* LOSO — correctly
disclaimed at `cv.py:112`), out-of-sample `combined_cv_r2` with a reliability floor.

---

## 5. Experimental Design

**[OBS] Present and good:** trial-grouped vs subject-grouped baselines; classifier-agnostic robustness
(LDA/SVM/RF agree, `robust_difficulty.py`); window-length robustness (exp_A); permutation null
(exp_E); predictor bake-off with LODO (exp_D); a within-dataset floor probe (floor_effect); actionable
recalibration (exp_B); temporal analog (exp_C); FDR everywhere; random-effects pooling with I² and
cohort/outlier sensitivity.

**[OBS] Missing / weak:**

- **No representation-robustness experiment.** "Difficulty is classifier-agnostic" is shown
  (LDA/SVM/RF), but all three run on the *same* amplitude-heavy 7-feature basis, and the predictor
  (MMD-to-pool) is computed on that *same* basis. The result is robust to the *classifier* but untested
  against the *feature representation* — which is the more important axis given the shared-representation
  coupling (§4/§9).
- **No cross-dataset transfer accuracy.** `transfer.json` exists but is marked `validated:false`
  (`CORRECTIONS.md` #10); no model is actually transferred across datasets.
- **No domain-adaptation baseline beyond mean/z-score.** exp_B tests centering and per-subject z-score;
  the roadmap's Adaptive-LDA (τ,λ shrinkage) and CORAL are named but **not implemented**
  (`calibration.py` implements a simpler source+k-shot curve instead).
- **Error analysis is thin** — which *classes*/*subjects* drive the correlations is only partially
  explored (Block B class ranking, exp_C rank checks).

---

## 6. Current Experiment Results — trends, not silos

Reading across the completed runs (A/B/C are in fact **complete**, not "running" as `STATE.md` still
says):

**[OBS] Trend 1 — the difficulty predictor is strong only near the accuracy floor.** In `meta.json`,
the 3 correct-sign FDR-significant datasets are ninapro_db1 (acc 0.12, r=−0.77), ninapro_db2 (acc 0.14,
r=−0.73), fors_emg (acc 0.50, r=−0.62). The **largest, highest-accuracy** dataset — grabmyo (43 subj,
acc 0.70) — gives **r=+0.03 (wrong sign, n.s.)**. This is the central tension and it is *not* resolved
(see §9/Trend 2).

**[OBS] Trend 2 — the floor-effect "resolution" is contradicted by its own committed data.**
`results/floor_effect/floor_effect.json`:

- grabmyo: trend **+0.95** → "floor effect: predictor strengthens as accuracy drops"
- ninapro_db2: trend **−0.93** → "**reverse**: predictor weakens as accuracy drops"
- fors_emg: −0.55 (n.s.); emaha_db1: −0.02 (flat)

These are **opposite-signed**. grabmyo's within-dataset curve is clean (r −0.44→−0.06 as acc
0.22→0.70), but ninapro_db2 stays entirely in the 0.05–0.14 floor and its predictor *strengthens* with
accuracy (−0.35→−0.75). `STATE.md` §5 declares this "RESOLVED" via a pooled "two-term ceiling+variance"
partial correlation (|r| vs mean_acc = −0.87; |r| vs spread = +0.76) — but **that analysis is nowhere
in `floor_effect.py` or the results tree.** The script computes only the single per-dataset trend. So
the resolution is a **post-hoc narrative not backed by a committed artifact**, reconciling
contradictory per-dataset signs. (Detail in §9 — this is the #1 publication risk.)

**[OBS] Trend 3 — recalibration is the strongest *new* result (exp_B).** Per-subject mean-centering
improves LOSO on **13/14** datasets (mean +3.6 pp, up to +8.5 pp on grabmyo_flow_static), all with
tight paired Wilcoxon p (`exp_B_summary.json`). This is genuinely valuable and, notably, **contradicts
the `STATE.md` §5.5 prediction** that centering probably wouldn't help. It also creates an interpretive
tension with E3: grabmyo's KL is "covariance-dominated" (`mean_share_of_excess`=0.28) and centering
removes only 27% of the excess KL (`kl_excess_removed_by_subject_center`=0.267) — yet centering yields a
large LDA gain. **KL-divergence share ≠ classifier impact.** The paper must frame this carefully (the
mean component, though a minority of the KL, disproportionately moves the linear decision boundary)
rather than asserting "covariance dominates, so alignment barely helps."

**[OBS] Trend 4 — window-length robustness holds, honestly (exp_A).** 12/14 keep the negative
difficulty sign, 13/14 keep within>cross. The 2 exceptions are the known near-zero (grabmyo) and the
known sign-reversed outlier (senic). Metrics rise monotonically with window length (expected).
Reportable as-is.

**[OBS] Trend 5 — the temporal result is real but narrow (exp_C).** Fixed-effects within-subject
r = −0.46/−0.61/−0.67 on the three GrabMyo-family sets (all p<3e-5), null on senic. Strong statistics,
**one lab.**

**[OBS] Trend 6 — senic is a consistent, unexplained sign reversal** across every analysis (meta
r=+0.51 FDR-sig wrong sign; exp_A difficulty_r=+0.49→+0.59; senic_probe confound unsupported).
Correctly quarantined.

**Which experiments to repeat/upgrade:** floor_effect (redo properly, §9); Module 5 and exp_D/E once a
DL target exists; add a second feature basis to robust_difficulty.

---

## 7. Reproducibility

**[OBS] Strong:** git present; seeds threaded (seed=42, 54 sites); atomic writes + resume/skip +
shard-safe collect (`exp_common.py`); parquet caching; **214 tests** including a real
equation-validation suite; results are per-dataset JSON with parameters.

**[OBS] Gaps:**

- **No dependency/environment pinning** — no `requirements.txt`/lockfile/`pyproject` at PROFILE root or
  repo root. Results depend on version-sensitive estimators (`RandomForestClassifier`,
  `mutual_info_classif` kNN-MI, `silhouette_score`). Exact numeric reproduction is at risk. **[REC]**
  pin `numpy/scipy/scikit-learn/pandas` versions.
- **The frame cache key omits the seed** (`windows.py:29`). The per-(subject,class) subsample depends on
  seed, but a re-run with a different seed silently reuses the seed-42 cache. Since frames feed
  *everything*, "seed robustness" is currently unfalsifiable without deleting the cache. **[REC]** add
  seed to the cache key (or document that frames are seed-frozen).
- **exp_D and exp_E outputs are absent from `results/experiments/`** despite `STATE.md` quoting their
  numbers as "done" (MMD/KL-mean tied at LODO ρ≈0.29; 3 correct-sign FDR-sig). Both scripts *do* write
  JSON (`exp_D:157`, `exp_E:83`), so they simply weren't persisted/brought back. **[REC]** re-run and
  commit; until then those numbers are unverifiable.
- **No provenance stamping** (git SHA / config snapshot) inside result files.
- **The floor-effect "two-term" model has no code path** (§6/§9).

---

## 8. Scalability

**[OBS] Good bones:** joblib/loky parallelism, per-(subject,class) caps, shared cached frames across
modules, incremental atomic outputs, shard-by-dataset design. Ran 14 datasets with "zero crashes."

**[OBS] Bottlenecks:**

- **O(N²) pairwise shift matrices** (`_pairwise_all`) scale quadratically in subjects — fine at ≤43,
  painful at hundreds.
- **O(N²) entropy** is capped by aggressive subsampling (`ENTROPY_MAX_WINDOWS_PER_CLASS=40`,
  `ENTROPY_MAX_SAMPLES=400`) — cheap but its stability vs subsample size is not reported.
- **Global mutable config for window length** (`exp_A` sets `config.WINDOW_MS`) is process-safe but
  fragile; a bug here silently mislabels window lengths.
- **No DL/GPU path** — the "swap to Paper-1 DL target" is a manual, format-tolerant loader
  (`load_paper1_loso`), not an integrated pipeline.

---

## 9. Research Risks (prioritized by impact on validity)

**[OBS — PROCESS RISK] R0 — the tracking docs narrate resolution ahead of the evidence.** `STATE.md`
speaks in a conclusion-forward register ("SOLID", "RESOLVED", "money result", "bulletproof"). That
register front-runs the numbers: the sharpest case is the floor-effect "RESOLVED" label sitting on
self-contradictory data (#1 below), but the same reflex recurs (e.g. exp_B being pre-narrated as
unlikely-to-help, then helping on 13/14). This is not a technical flaw — the *craftsmanship* is sound —
but it is the highest-leverage thing to fix, because a reviewer who finds one confident claim
contradicted by your own supplementary numbers will discount the airtight parts too. **Fix:** state
every claim at the confidence its committed artifact supports, and downgrade a verdict to "open" the
moment the code/results do not close it. This risk is the *why* behind §14's "pre-register the decision
rule, report the honest branch" discipline.

1. **[OBS — HIGHEST] The floor-effect confound is not actually resolved, and the "resolution"
   reintroduces pseudo-replication.** The committed per-dataset trends are contradictory (grabmyo +0.95
   vs ninapro_db2 −0.93, `floor_effect.json`). The pooled "two-term ceiling+variance" claim in
   `STATE.md` §5 (a) isn't in the repo, and (b) pools **~29 nested channel-count rungs** that are *not
   independent* — rungs within a dataset are nested subsets (k=1⊂k=2⊂…) and clustered by dataset, and
   `mean_acc`↔`spread` are collinear at +0.877. Partialing two collinear predictors over
   non-independent rungs and quoting p<0.001 is the **same pseudo-replication error the team correctly
   fixed in F3 (A4) and flagged in F4 (`combined_cv_r2`)**. The honest status: grabmyo shows a floor
   effect, ninapro_db2 shows the opposite, and whether the predictor measures *difficulty* or
   *distance-from-floor* is **unsettled** — exactly as `NEXT_STEPS.md` Stage 2 warned. This directly
   threatens the money result.
2. **[OBS — HIGH] Shared-representation coupling between predictor and target.** MMD-to-pool and
   LDA-LOSO accuracy are both monotone functions of "how far this subject sits from the training
   distribution in the *same* 7-feature space." The correlation is partly mechanical. The de-confounding
   test (predict a *DL* model's failures, or use a *different* representation for predictor vs target)
   is acknowledged but undone. Pooled r=−0.39 (not −0.9) suggests it's not fully tautological, but the
   coupling must be disclosed and, ideally, broken.
3. **[OBS — MEDIUM standalone / HIGH entangled] Everything rests on the self-LDA-LOSO proxy.**
   *As a standalone limitation this is disclosed (`STATE.md` §7), so on its own it is a scope/ambition
   limit — it bounds how strongly the Paper-1 bridge can be claimed, not correctness* (nuance adopted
   from the independent cross-check). Its real severity comes from **entanglement with #2**: the
   coupling is worrying precisely because the target is a *linear* classifier on the *same* basis the
   predictor is built from, so a **single change — a DL target on a different representation — resolves
   the proxy limit and breaks the coupling at once.** That is why the DL-target swap (§14 X3) is the
   highest scientific-value single experiment in the whole program.
4. **[OBS — MEDIUM] Amplitude-dominated basis** means "distribution shift" is largely "contraction-
   amplitude shift" — an interpretive risk if sold as general covariate shift.
5. **[OBS — MEDIUM] `_frob` uniformity aggregation** on reported shift scalars invites misreading;
   **MMD γ=1/d** makes cross-dataset shift magnitudes non-comparable.
6. **[OBS — MEDIUM] Missing artifacts (exp_D/E) + no dependency pinning + seed-blind cache** —
   reproducibility debt.
7. **[OBS — LOW] MI-SU bug; FuzzyEn non-invariance; entropy decimation without anti-alias** — real but
   low-impact, in descriptive features that failed FDR anyway.

**Overfitting/leakage:** the *obvious* leaks (overlapping-window CV, H-div trial memorization,
in-sample R², cohort leakage) are all **fixed and tested** — commendable. The *subtle* residual is the
shared-representation coupling (#2), which is a validity risk, not a leak per se.

---

## 10. Comparison with Current sEMG Literature (last 3–5 yr)

**[OBS] Where it's aligned or ahead:** cross-subject distribution-shift quantification (Albuquerque
2022; Li 2024; Qiu 2025); trial-grouped leakage discipline (a known but under-practiced issue);
random-effects meta-analysis across corpora (rare in this field); the KL affine-invariance
identifiability result (novel and correct).

**[OBS] Where it lags / is missing:**

- **No modern representation.** The field has largely moved to learned representations
  (CNN/TCN/Transformer embeddings, and 2023–2025 **self-supervised sEMG** and **foundation-model**
  work). Characterizing shift/difficulty in a fixed handcrafted amplitude basis is a real but dated
  lens. At minimum, repeat the shift/difficulty analysis in a learned embedding.
- **No CORAL / statistical-alignment / adversarial DA baseline**, despite the mean-centering result
  begging for the covariance-alignment comparison (the paper even predicts CORAL as future work —
  `STATE.md` §5.5 B).
- **MMD is single-kernel, fixed-γ.** Multi-kernel MMD or the median heuristic is standard.
- **No benchmark against an established difficulty/OOD score** (e.g., energy/Mahalanobis-to-pool as an
  OOD detector, deep-ensemble disagreement) to show the cheap statistic is competitive.
- **Electrode-shift robustness** (a major 2023–2025 theme; you have senic!) is under-exploited — senic
  is treated only as an outlier rather than as a shift-robustness testbed.

---

## 11. Research Gaps (why / impact / priority / next experiment)

| Gap | Why it matters | Impact | Priority | Next experiment |
|---|---|---|---|---|
| Floor-effect not truly resolved | The money result may be "predicts distance-from-floor" | Validity of the headline | **High** | Match-accuracy design done *properly* with a clustered/mixed-effects model, not pooled rungs |
| Predictor ⊥ target representation | Correlation partly mechanical | Novelty & validity | **High** | Compute MMD in basis A, target LDA in basis B; and use the DL target |
| DL target absent | Thesis is "predict the model's failures" | Framing strength | **High** | Wire `load_paper1_loso`; re-run Module 5/D/E |
| Learned-representation shift | Field standard | Relevance | **Med** | Repeat Module 3/5 on a CNN/SSL embedding |
| CORAL / covariance alignment | Closes the exp_B story | Method contribution | **Med** | Add CORAL as a 4th arm in exp_B |
| MMD bandwidth & aggregation | Cross-dataset comparability | Correctness of scalars | **Med** | Median-γ + mean-off-diagonal aggregate; sensitivity table |
| Entropy stability & invariance | Feature validity | Low (failed FDR) | **Low** | Invariance unit test; subsample-stability curve |

---

## 12. Roadmap Review

**Remove/reposition:** the SDI-as-actionable-tool (already a null, oracle ceiling ~1 pp — keep only as
a *predictor*); the deprecated `mean_vs_cov_ratio` and `A4_..._mmd` (already deprecated in code); and
**reposition Modules 1–4 as a labelled *dataset-atlas* contribution — their "only silhouette survives
FDR" outcome is a reportable negative finding, not a defect to hide.**

**Add:** (i) a *correct* floor-effect analysis; (ii) a representation-decoupled predictor test; (iii)
the DL-target swap; (iv) CORAL baseline; (v) one learned-embedding replication.

**Reorder:** the floor-effect + coupling de-confound must come **before** figures/write-up (as
`NEXT_STEPS.md` originally insisted) — not after, and not declared "resolved" prematurely.

**Revisit assumptions:** proxy target ≈ DL target; MMD as "the" statistic (exp_D reportedly ties MMD
with KL-mean — verify from committed output first); "covariance-dominated ⇒ mean alignment secondary"
(exp_B refutes the operational version).

**Suggested revised sequence:** (1) fix + re-run floor_effect with a mixed-effects model; (2)
representation-decoupling + DL-target Module 5/D/E; (3) CORAL in exp_B; (4) commit exp_D/E, pin deps,
seed the cache; (5) figures; (6) write-up around invariance-proof + within>cross + recalibration, with
the difficulty predictor stated at its *honest* scope.

---

## 13. Final Assessment

**1. Executive summary.** A rigorous, self-critical, methodologically strong characterization of sEMG
datasets whose *process* (audit → adversarial review → repair, 214 tests) is exemplary and whose
salvaged contributions (KL affine-invariance identifiability proof; within>cross-subject separability
gap; actionable per-subject recalibration) are real and defensible. Two issues block publication-grade
claims: the **floor-effect confound is declared resolved on evidence that is contradictory and not in
the repo**, and the **difficulty predictor is coupled to its target through a shared feature
representation and a linear-proxy target.** Both are fixable without new data.

**2. Strengths.** Correct, tested core math (KL split + invariance proof, H-div, meta-analysis, FDR,
fixed-effects panel estimator); leakage discipline far above field norm; cohort-aware,
heterogeneity-aware pooling; honest reporting of nulls (SDI actionability, senic); excellent experiment
engineering (atomic/shard-safe/resumable); strong window-length and classifier-agnostic robustness.

**3. Major weaknesses.** (a) Floor-effect "resolution" unsupported and internally contradictory;
(b) predictor↔target shared-representation coupling; (c) entire difficulty edifice on a self-LDA-LOSO
proxy (*disclosed*, so a scope limit standalone — but entangled with (b), and one DL-target swap fixes
both); (d) amplitude-dominated basis reframes "shift"; (e) reproducibility debt (no dep pinning,
seed-blind cache, missing exp_D/E artifacts).

**4. Highest-priority limitations.** Floor effect; representation coupling; proxy target;
GrabMyo-only temporal axis; 9-cohort / I²=0.79 heterogeneity (well-disclosed).

**5. Critical research gaps.** Proper floor analysis; representation-decoupled + DL target;
learned-embedding replication; CORAL baseline.

**6. Methodological concerns.** MI-SU denominator bug (`H(C)` vs `H(C)+H(F)`); FuzzyEn non-invariance
(faithful to sources but mis-sold if called invariant; self-referential test); `_frob` uniformity
aggregation on reported scalars; fixed MMD γ; entropy decimation without anti-alias; KL-share conflated
with classifier impact in the E3↔exp_B narrative.

**7. Risks to publication-quality research.** A reviewer *will* ask "difficulty or distance-from-floor?"
and "isn't MMD-to-pool mechanically tied to LDA-LOSO?" — the current artifacts do not answer either.
The `STATE.md` "RESOLVED" framing, contradicted by `floor_effect.json`, is the kind of overclaim that
sinks a submission.

**8. Recommended next experiments.** (i) Floor: match accuracy across datasets via a mixed-effects
model with dataset random effects (not pooled nested rungs); (ii) decouple: predictor-in-basis-A vs
target-in-basis-B, and swap in the DL LOSO target, then re-run Module 5/D/E; (iii) CORAL as a 4th arm
in exp_B; (iv) one learned-embedding replication of Module 3/5.

**9. Recommended roadmap revisions.** See §12 — de-confound *before* figures; reposition the
descriptive modules as a labelled atlas (their null predictors are a finding); commit exp_D/E; pin the
environment. **Full execution-ready program in §14.**

**10. Top-10 actionable improvements (ranked by expected research impact).**

1. **Redo the floor-effect analysis properly** (mixed-effects, per-dataset random effects) and
   **retract the premature "RESOLVED."** — *decides whether the headline survives.*
2. **Break the predictor↔target coupling** (different basis for predictor vs target) and **swap in the
   DL LOSO target.** — *converts a possibly-mechanical correlation into a real prediction.*
3. **Promote exp_B recalibration to a method contribution** and add a **CORAL** arm — reconcile with E3
   explicitly (KL-share ≠ boundary impact).
4. **Commit exp_D/E outputs; pin dependencies; add seed to the cache key.** — *makes the paper
   reproducible.*
5. **Reframe "distribution shift" as substantially amplitude shift** given `REPR_BASIS`; test a
   de-amplituded basis.
6. **Fix the MI-SU denominator** (or rename) and **add a FuzzyEn amplitude-invariance test.**
7. **Report magnitude-preserving shift scalars** (mean off-diagonal) beside `_frob`; **use
   median-heuristic MMD γ.**
8. **State the temporal (exp_C) and A4 results as one-lab (GrabMyo)** findings, prominently.
9. **Add one learned-representation replication** of the shift/difficulty pipeline to meet
   current-literature expectations.
10. **Lead the paper with the invariance proof + within>cross gap + recalibration**, and state the
    difficulty predictor at its honest scope ("reliable where cross-subject accuracy has headroom is
    *not yet* established").

---

### Bottom line

The framework is sound and the craftsmanship is high; the equations are, with two low-impact
exceptions, correctly implemented and well-tested. The gap between the project's *actual* evidence and
its *stated* conclusions is concentrated in exactly one place — the floor-effect confound and its
coupled cousin, the shared-representation predictor — and closing that gap (not more descriptive
modules) is what turns this from a strong internal report into a publishable paper.

---

## 14. Detailed Experimental Program to Support a Publishable Paper

*(Added in revision 2, at the author's request. The decision has been made to invest in more
experiments rather than only trim claims — so this is a complete, prioritized, execution-ready program.
Every experiment reuses the existing cached frames / `dsprofile` infrastructure unless noted, and every
one is designed so that **each outcome branch is publishable**. For each item I give: the **question**,
the **design**, what it **reuses**, its **novelty**, what it **establishes**, a **ground-truth check**
(a synthetic control with a known answer, to validate the experiment's own code before trusting its
output — the same discipline as `tests/test_math.py`), and the **decision rule** for each branch.)*

### 14.0 How to run this program (principles + testing discipline)

1. **Pre-register the decision rule.** Before running each experiment, write down what each outcome
   *means*. This is the direct antidote to §9 R0 (premature "RESOLVED" labels).
2. **Every branch is a result.** "Predictor only works near the floor," "shift is mostly amplitude,"
   "cheap statistic predicts LDA but not a deep net" — each is publishable if stated honestly.
3. **Validate the code against synthetic ground truth first.** Each experiment below ships with a
   synthetic control whose answer is known analytically; the experiment must reproduce it before being
   trusted on real data. Add each as a test in `tests/` so it is pinned like the existing 214.
4. **Sequence:** code fixes + reproducibility (F1–F5) → Tier 0 blockers → Tier 1 core → Tier 2
   breadth. Figures and write-up only after Tier 0 lands.

### 14.1 The novelty ledger — what is genuinely new, and which experiment secures it

| # | Candidate contribution | Why it is novel (vs 2020–2025 literature) | Status today | Secured / tested by |
|---|---|---|---|---|
| **N1** | The Gaussian-KL mean/cov split is **exactly affine-invariant**, so the raw-vs-normalized "which moment dominates" contrast used in this literature is **non-identifiable** | To our knowledge no sEMG/BCI paper states this identifiability result; it *retracts* a common analysis and replaces it with a per-subject-map estimator + trial-disjoint null | **Solid** — algebraic proof + committed numerics (`block_c`, rel-diff 1.1e-8; ridge breaks it) | Formalize as a 1-page proof appendix (X0); already numerically verified |
| **N2** | **Leakage audit**: trial-grouped CV, and a demonstration that H-divergence over 50%-overlapping windows **saturates at its max regardless of true divergence** | Overlapping-window leakage is known but rarely *quantified on distribution-shift estimators*; the saturation demo (d_H≈1.97 for identical dists) is new | **Solid** — fixed + pinned by a clustered-data test | `tests/test_math.py:218–237` |
| **N3** | A **cheap calibration-window statistic predicts cross-subject difficulty across many sEMG datasets** | Extends Albuquerque 2022 (one EEG dataset) to **9 sEMG cohorts** + a random-effects meta-analysis | **AT RISK** — floor-confounded, representation-coupled, proxy-targeted | **X1, X2, X3** (this is the make-or-break trio) |
| **N4** | **Within-subject separability systematically overstates cross-subject**, quantified across 14 datasets | Clean, rarely quantified this explicitly across a multi-corpus panel | **Solid** | already computed (`within_minus_cross`); X7 for robustness |
| **N5** | **Unsupervised per-subject recalibration is actionable**: +3.6 pp (up to +8.5) on 13/14, and it operationalizes *why* z-scoring helps | Feature-centering DA is old; the **systematic multi-dataset demonstration + reconciliation with the KL mean/cov decomposition** is the new part | **Strong but under-sold** | X4 (CORAL), X13 (imbalance) |
| **N6** | **Inter-subject > inter-day** quantified across cohorts + a **temporal difficulty analog** | Settles a genuine disagreement (Li 2024 / Qiu 2025 vs others) with a fair, subject-level test | **Solid but one-lab** (GrabMyo) | honest scoping; X10 for the shift axis |
| **N7** | **SDI** — a portable cross-dataset difficulty index | No cross-dataset sEMG difficulty index exists | **Weak transfer, honestly reported** | X3, X8, X11 |

**Reading:** N1, N2, N4 are already publication-grade. N5 is your most under-sold asset. N3/N7 — the
"money result" — is the one at risk, and X1/X2/X3 decide whether it survives in strong or scoped form.

### 14.2 Coverage map — every finding in this report maps to an action

| Finding (§) | Action(s) | Tier |
|---|---|---|
| Floor-effect confound (§9-1) | **X1**, X11 | 0 |
| Predictor↔target coupling (§9-2) | **X2**, X3 | 0 |
| Self-LDA-LOSO proxy (§9-3) | **X3** | 0 |
| Amplitude-dominated basis (§9-4) | X5 | 1 |
| `_frob` uniformity / MMD γ=1/d (§9-5, §3) | X7, F3, F4 | 1 / 3 |
| Reproducibility debt (§9-6, §7) | F5, X15 | 3 |
| MI-SU bug / FuzzyEn / entropy decimate (§9-7, §4) | F1, F2, F-dec | 3 |
| No learned representation (§10) | X6 | 1 |
| No CORAL / DA baseline (§5) | X4, X14 | 1 / 2 |
| Transfer matrix `validated:false` (§5) | X9 | 2 |
| No OOD/difficulty baseline (§10) | X8 | 1 |
| senic sign reversal (§6-Trend 6) | X10 | 2 |
| Entropy/shift subsample stability (§8) | X12 | 2 |
| Class-imbalance uncontrolled (§2) | X13 | 2 |
| Meta heterogeneity I²=0.79 (§6) | X11 | 2 |

---

### 14.3 Tier 0 — BLOCKERS (nothing is written up before these land)

**X1 — Floor-effect, done correctly** · `Priority: CRITICAL · gates the write-up`
- **Question:** does the cheap statistic predict *difficulty*, or merely *distance from the accuracy
  floor*?
- **Design (three complementary probes, all on cached frames):**
  - **X1a Matched class count.** Sub-sample ninapro_db1 (53 cls) and ninapro_db2 (50 cls) down to
    grabmyo's ~17 classes; refit the per-subject MMD-vs-LOSO r. Repeat over ≥20 random class subsets;
    report mean ± bootstrap CI.
  - **X1b Accuracy-matched crippling.** Degrade grabmyo (drop channels via the mRMR order already in
    `floor_effect.py`, and/or shorten the window) until mean LOSO ≈ 0.15; refit r.
  - **X1c Correct pooled model.** Replace the pooled-nested-rung partial correlation with either
    (i) a **linear mixed-effects model** `|r_rung| ~ mean_acc + acc_std + (1|dataset)`, or better
    (ii) rebuild rungs from **disjoint random channel subsets of fixed size** (not nested k=1⊂k=2⊂…)
    so rungs are exchangeable, then bootstrap over datasets. Never quote a p from nested rungs.
- **Reuses:** `floor_effect.py` channel-sweep + `loso_lda_accuracy` + `mmd_rbf`.
- **Novelty:** turns an *open confound* into a *characterized* one — "the predictor is (or isn't) a
  floor artifact, and here is the accuracy regime where it holds." That regime statement is itself a
  contribution the field lacks.
- **Establishes:** the honest scope of N3.
- **Ground-truth check:** build two synthetic frames. **(a) Pure-floor:** all subjects share one
  distribution, labels near-random so LOSO≈chance and MMD-to-pool≈0 — the analysis must report the
  correlation as an artifact / floor-driven. **(b) Real-difficulty:** inject a latent per-subject
  hardness that lowers that subject's accuracy *and* raises its MMD, **independent of the global
  accuracy level** — X1a/X1c must recover a *floor-invariant* negative r. If the code labels (b)
  "floor," it is wrong.
- **Decision:** r survives matched-class and holds at grabmyo-crippled accuracy → **N3 is real, report
  it**; r collapses when class count / floor is removed → **"cheap statistics predict who a model fails
  on, but only once it is already failing"** — a real, defensible, *scoped* result. Either way, delete
  the "RESOLVED" label until this runs.

**X2 — Representation decoupling** · `Priority: CRITICAL`
- **Question:** how much of the MMD↔LOSO correlation is *mechanical* (both computed in the same
  feature space) vs a real property of the data?
- **Design:** compute the **predictor** (MMD-to-pool) in **basis A** and the **target** (LDA-LOSO) in a
  **disjoint basis B**, and correlate cross-basis; do both directions. Natural split: A = amplitude
  {MAV, WL, RMS, MFL}, B = shape/complexity {HJ_MOB, HJ_COM, spectral shape, entropy where defined}.
  Also A = handcrafted, B = learned embedding (ties to X6).
- **Reuses:** `subject_shift_stats`, `loso_lda_accuracy`, existing feature columns.
- **Novelty:** an explicit **mechanical-coupling control** for difficulty prediction — essentially
  nobody in this literature separates the space the OOD score lives in from the space the classifier
  fails in.
- **Establishes:** whether N3 is a data property or a same-space artifact.
- **Ground-truth check:** two orthogonal feature blocks. **(a) Mechanical-only:** subject shift lives
  only in A, class structure only in B, *no* real per-subject hardness → cross-basis r must be ≈ 0
  (test that the control correctly finds "no signal"). **(b) Real:** a latent hardness perturbs *both*
  blocks → cross-basis r must be strongly negative. If (a) returns a large correlation, the "decoupling"
  is not decoupling.
- **Decision:** cross-basis r survives (both directions) → the predictor tracks a real data property,
  N3 strengthened; only same-basis r holds → coupling is substantial, scope the claim to "MMD is a
  same-representation diagnostic."

**X3 — DL-target swap (the single highest-value experiment)** · `Priority: CRITICAL · depends on Paper-1`
- **Question:** does the cheap statistic predict a **deep net's** LOSO failures, not just an LDA's?
- **Design:** wire `load_paper1_loso` to the BENCH-LOSO EMAHA sweep (and any other DL LOSO available);
  re-run Module 5 + exp_D + exp_E with **DL LOSO accuracy as the target**, keeping predictors cheap.
- **Reuses:** `module5.load_paper1_loso` (already written, format-tolerant), the whole Module-5/D/E
  stack.
- **Novelty:** this is the **bridge to Paper 1** actually delivered — "a training-free statistic
  forecasts which users a *trained deep model* will fail on." That is the headline the two-paper
  program was designed around, and it simultaneously **breaks the coupling of X2 and removes the proxy
  limitation of §9-3 in one move.**
- **Establishes:** N3 and N7 at full strength (or refutes them).
- **Ground-truth check:** (i) **fixture** — a synthetic BENCH-LOSO JSON with known `{subject: acc}`;
  `load_paper1_loso` must return exactly it. (ii) **alignment** — permute the target's subject IDs; the
  correlation must drop to ≈ 0 (proving IDs are *matched*, not positionally aligned). (iii) **shuffled
  null** — MMD-to-pool vs a shuffled DL target → r ≈ 0 across seeds.
- **Decision:** cheap MMD predicts DL failures (r<0, FDR-sig) → **the paper's central claim is earned**;
  predicts LDA but not DL → **major, honest scope correction** ("the statistic tracks *linear*
  separability, not deep-model difficulty") — still publishable, and far better found now than in review.

---

### 14.4 Tier 1 — Core strengthening

**X4 — CORAL / covariance-alignment arm in exp_B** · `Priority: HIGH`
- **Question:** does aligning per-subject **covariance** beat aligning only the **mean**?
- **Design:** add **subject-CORAL** (whiten each subject toward a common covariance) and **center+CORAL**
  as 3rd/4th normalization arms in `exp_B_recalibration.py`; paired Wilcoxon vs baseline and vs center.
- **Reuses:** exp_B LOSO-LDA loop.
- **Novelty:** closes the loop E3 → exp_B → CORAL: it directly tests the mechanism the mean/cov
  decomposition predicts, which the roadmap named as future work.
- **Establishes:** the mechanistic claim behind N5, and reconciles "covariance-dominated KL" (E3) with
  "mean-centering helps a lot" (exp_B).
- **Ground-truth check:** synthetic with a **pure per-subject covariance rotation** (means aligned) →
  CORAL must recover the no-shift LOSO accuracy while centering does *not* help; **pure per-subject mean
  offset** → centering helps and CORAL ≈ centering. Validates each transform removes the moment it
  claims to.
- **Decision:** CORAL > center → covariance alignment is the lever (future-work becomes present-work);
  CORAL ≈ center → mean alignment already captures the achievable gain (the surprising, citable result).

**X5 — De-amplituded basis ablation** · `Priority: HIGH`
- **Question:** is "distribution shift" here anything more than **contraction-amplitude** shift?
- **Design:** repeat Module 3 (shift), Module 5 (difficulty), and A4 with amplitude features removed
  (basis = {HJ_MOB, HJ_COM, spectral shape, complexity}).
- **Reuses:** module3/module5 with a restricted `REPR_BASIS`.
- **Novelty:** an explicit amplitude-vs-shape decomposition of cross-subject shift — rarely separated.
- **Establishes:** the honest meaning of N3/N6 ("shift" = amplitude or structure?).
- **Ground-truth check:** synthetic where between-subject difference is a pure per-subject amplitude
  scale → MMD>0 in the amplitude basis, MMD≈0 in the invariant basis; a shape-difference synthetic →
  MMD>0 in both.
- **Decision:** findings survive de-amplituding → shift is structural (stronger claim); collapse →
  "cross-subject sEMG shift is dominated by contraction amplitude" (itself a clean, citable finding).

**X6 — Learned-representation replication** · `Priority: HIGH (meets 2023–25 reviewer expectations)`
- **Question:** are the shift/difficulty findings artifacts of the handcrafted amplitude basis, or do
  they replicate in a learned embedding?
- **Design:** train a small **subject-agnostic** conv autoencoder (or contrastive/SSL encoder) on
  windows; extract embeddings; re-run shift + difficulty + within>cross in that space.
- **Reuses:** windowing + module3/5 metrics; needs one lightweight training (CPU-feasible; A5000
  available).
- **Novelty:** brings the characterization into the representation regime the field now uses; a
  handcrafted-vs-learned contrast of *difficulty predictability* is itself novel.
- **Establishes:** robustness of N3/N4 to representation; feeds X2's A=handcrafted / B=learned split.
- **Ground-truth check:** AE reconstruction MSE below threshold on held-out windows; **label-permutation
  null** (shuffle labels → difficulty-r ≈ 0 in the embedding); on a controlled separable synthetic the
  embedding's silhouette ≥ the handcrafted basis (sanity that geometry is preserved).
- **Decision:** findings replicate → robust and modern; differ → the handcrafted lens was the story
  (report the contrast).

**X7 — MMD kernel & aggregation sensitivity** · `Priority: MED`
- **Question:** are the shift/difficulty/A4 conclusions stable to kernel bandwidth and matrix
  aggregation?
- **Design:** recompute MMD with (i) **median-heuristic γ**, (ii) **multi-kernel MMD** (sum over a γ
  grid); report difficulty-r and the A4 ratio under each. Add a **magnitude-preserving** aggregate
  (mean off-diagonal) beside `_frob`.
- **Reuses:** `mmd_rbf`, `module3`, `block_c` A4.
- **Novelty:** a robustness envelope for the shift metrics — rarely reported in this literature.
- **Establishes:** that N3/N6 are not γ artifacts; fixes the misreadable `_frob` scalar (see F3/F4).
- **Ground-truth check:** identical distributions → MMD ≈ 0 for median-γ and *every* kernel in the grid;
  a family of increasing-mean Gaussian shifts → multi-kernel MMD monotone increasing; single-kernel
  biased MMD vs the **closed-form MMD² for two Gaussians** within tolerance.
- **Decision:** conclusions stable → report as robustness; unstable → report the sensitivity band
  honestly and justify the chosen γ.

**X8 — Difficulty / OOD baseline bake-off** · `Priority: MED`
- **Question:** is cheap MMD-to-pool actually the best cheap difficulty predictor, or does a standard
  OOD score beat it?
- **Design:** compare MMD-to-pool against **Mahalanobis-to-pool**, **kNN-mean-distance-to-pool**,
  **energy distance**, and (if DL lands) **deep-ensemble disagreement**, ranked by leave-one-cohort-out
  Spearman — extending exp_D honestly.
- **Reuses:** exp_D LODO machinery + module5 parquets.
- **Novelty:** positions the "cheap statistic" claim against established OOD detectors instead of
  asserting it.
- **Establishes:** empirical justification for N3/N7's choice of statistic.
- **Ground-truth check:** inject one subject drawn from a clearly shifted distribution; **every** score
  must rank that subject top-1 most-OOD before any are trusted for ranking.
- **Decision:** MMD competitive/best → "cheap MMD suffices," earned; another score dominates → report
  the better one (still a positive result).

---

### 14.5 Tier 2 — Generalization, robustness, completeness

**X9 — Cross-dataset transfer accuracy (validate the transfer matrix)** · `Priority: MED`
- **Question:** does `compatibility_mmd` predict real cross-dataset transfer accuracy?
- **Design:** where montages/label-spaces intersect (within the NinaPro family, within EMAHA, GrabMyo
  variants), train on A / test on B; measure transfer accuracy; correlate with `compatibility_mmd`.
- **Reuses:** `transfer.py` (currently `validated:false`), native-scale frames.
- **Novelty:** turns an unvalidated shape-only matrix into a validated (or honestly null) transfer
  result.
- **Ground-truth check:** **A→A** (same dataset, subject-split) transfer must match within-dataset LOSO
  within tolerance; two identical synthetic datasets → `compatibility_mmd`≈0 and transfer≈within; a
  deliberately incompatible pair → large MMD, transfer near chance.
- **Decision:** correlation present → the matrix is a usable compatibility tool; null → report that
  marginal MMD does not predict transfer (also informative).

**X10 — senic as an electrode-shift testbed + sign-reversal probe** · `Priority: MED`
- **Question:** why does senic reverse sign, and can its shift conditions serve as a covariate-shift
  benchmark?
- **Design:** (a) use senic's shift/rotation/fatigue **conditions** as a within-subject covariate-shift
  axis — does MMD-across-conditions predict per-condition accuracy drop? (b) decompose senic's
  MMD-to-pool into amplitude vs shape (via X5's bases) and test whether the reversal is a
  montage/data-volume/label artifact vs a real effect.
- **Reuses:** `senic_probe.py`, exp_C machinery, X5 bases.
- **Novelty:** converts an outlier you currently quarantine into an **electrode-shift robustness
  contribution** — a hot 2023–25 topic you are uniquely positioned to address.
- **Ground-truth check:** within-condition null — MMD between two **trial-disjoint halves of the same
  condition** ≈ 0 (mirrors the existing `hdiv_within_subject_null`), so the condition-shift signal is
  not trial memorization.
- **Decision:** any branch is reportable; a mechanistic explanation of the reversal removes a standing
  embarrassment.

**X11 — Meta-regression of the difficulty effect** · `Priority: MED`
- **Question:** which dataset properties moderate the difficulty correlation (formal test of the floor
  effect at the between-dataset level)?
- **Design:** model per-dataset Fisher-z(r) ~ mean_acc + n_classes + fs + n_channels (≤2 predictors at
  a time given k=14), weighted by n, with cohort-robust SEs; complements X1's within-dataset view.
- **Reuses:** `meta.pool_random_effects` + `meta_analysis` per-dataset table.
- **Novelty:** a proper meta-regression across sEMG corpora — essentially absent in the field.
- **Ground-truth check:** simulate K per-dataset r's from `r_k = β·mean_acc_k + noise` with known β →
  recover β's sign and approximate magnitude with CI covering truth; null sim (β=0) → CI covers 0 at
  ~95% over repeats.
- **Decision:** reports, at the dataset level, exactly how much of N3 is accuracy-moderated — the
  between-dataset complement to X1.

**X12 — Subsample-stability / convergence** · `Priority: MED`
- **Question:** are the entropy and shift/difficulty estimates converged at the current caps
  (40 windows/class, 600/subject)?
- **Design:** sweep subsample size; report metric mean ± std vs n.
- **Ground-truth check:** on a synthetic stationary source the curve must flatten and its variance
  shrink ∝ 1/n. **Establishes** that the caps are adequate (or exposes where they aren't).

**X13 — Class-imbalance stratification** · `Priority: MED`
- **Question:** does per-subject centering hurt on class-imbalanced subjects (exp_B's own caveat)?
- **Design:** re-run exp_B and the difficulty correlations stratified/controlled by
  `median_class_imbalance_ratio`.
- **Ground-truth check:** synthetic strongly-imbalanced subject with a class-biased global mean →
  centering provably degrades it; balanced subject → neutral/positive; the stratified analysis must
  surface the interaction. **Establishes** the boundary conditions of N5.

**X14 — Adaptive-LDA / calibration curve (implement the roadmap baseline)** · `Priority: MED`
- **Question:** how does difficulty translate to onboarding cost (zero→one→few-shot)?
- **Design:** implement the roadmap's Adaptive-LDA shrinkage `μ̃=τμ_cal+(1−τ)μ_src`,
  `Σ̃=λΣ_cal+(1−λ)Σ_src`; produce the calibration curve tied to Paper-1's k-shot axis.
- **Reuses:** `calibration.py` (currently a simpler source+k-shot curve).
- **Novelty:** frames subject difficulty as a **calibration-budget** quantity aligned with Paper-1 —
  the shared x-axis the two-paper program wanted.
- **Ground-truth check:** boundary conditions — `(τ=1,λ=1)` equals target-only LDA to tolerance;
  `(τ=0,λ=0)` equals source-only LDA; the curve is monotone non-decreasing in calibration data on a
  well-behaved synthetic.
- **Decision:** produces the "difficulty = onboarding cost" figure (A8), whatever its shape.

---

### 14.6 Tier 3 — Code fixes & reproducibility (cheap; run in parallel)

Each fix says **what it does** and its **ground-truth test** (add all to `tests/`).

- **F1 — MI symmetric-uncertainty denominator** (`module2_separability.py:74`).
  **Does:** restore `SU = 2·I(C;F)/(H(C)+H(F))` (currently divides by `H(C)` only, collapsing the
  ranking to raw MI). **GT test:** a feature that is an exact copy of the label → SU **= 1.0**; a
  feature independent of the label → SU **= 0**; noisy copy → 0<SU<1. *(The current code returns ≈ 2.0
  for the exact-copy case — an "uncertainty" > 1 is self-evidently wrong.)*
- **F2 — FuzzyEn amplitude invariance** (`features_extra.py:166`).
  **Does:** either switch the membership to `exp(−(d/tol)ⁿ)` (dimensionally consistent, invariant) or
  keep the source formula `exp(−dⁿ/tol)` and **document the residual amplitude sensitivity**. **GT
  test:** `FuzzyEn(x) == FuzzyEn(a·x)` for `a ∈ {0.1, 10}` within tolerance (an *independent* invariance
  check, unlike the current self-referential reference test), exercised at the real operating point
  **n=5**. If you keep the source formula, the test documents the sensitivity rather than asserting
  invariance.
- **F3 — magnitude-preserving shift aggregate** (`module3_shift.py:173`).
  **Does:** report **mean off-diagonal** (and a properly scaled Frobenius) alongside `_frob`. **GT
  test:** matrices `M` and `10·M` must give **different** magnitude aggregates but the **same** `_frob`
  — proving `_frob` is scale-free and the new aggregate is not.
- **F4 — MMD median-heuristic γ** (`module3_shift.py:37`).
  **Does:** default γ to `1/(2·median‖xᵢ−xⱼ‖²)`; keep γ=1/d as an option. **GT test:** identical
  distributions → MMD ≈ 0 for the chosen γ; monotonicity under increasing Gaussian shift.
- **F5 — seed in the cache key** (`windows.py:29`).
  **Does:** add `seed` (and window-ms) to `_cache_file` so a re-run with a different seed rebuilds the
  frame instead of silently reusing seed-42's subsample. **GT test:** `build(seed=1)` and `build(seed=2)`
  produce **different** cache files / subsamples; `build(seed=1)` twice → identical file reused.
- **F-dec — anti-aliased entropy decimation** (`features_extra.py:253`).
  **Does:** replace naive `x[..., ::step]` with `scipy.signal.decimate` (as already done for E6). **GT
  test:** a 400-Hz tone decimated by 4 must not fold into the passband (spectral centroid preserved).
- **X15 — reproducibility hardening.** **Does:** commit exp_D/E outputs; add `requirements.txt` +
  lockfile (pin `numpy/scipy/scikit-learn/pandas`); stamp git SHA + config into every result JSON; add
  one `reproduce.sh`. **GT test:** a fresh clone + `pip install -r requirements.txt` +
  `python tests/test_math.py` → all green; a result JSON contains the SHA it was produced at.

---

### 14.7 Execution order & dependencies

1. **First (½ day):** F1–F5, F-dec, X15 — cheap, and several unblock later work (X7 needs F4; X8 uses
   corrected scores; everything benefits from F5).
2. **Tier 0 (the gate):** **X1** and **X2** start immediately on cached frames; **X3** as soon as the
   BENCH-LOSO EMAHA sweep exists. *Do not touch figures until X1 lands and the "RESOLVED" label is
   corrected.*
3. **Tier 1:** X4 (extends exp_B), X5 (cheap), X7 (after F4), X8; **X6** in parallel (one training run).
4. **Tier 2:** X9–X14 as capacity allows; X10 pairs naturally with X5's bases; X11 complements X1.
5. **Only then:** figures (`run_phase2.py --exp figs` + the floor and window panels) → write-up.

### 14.8 What the paper looks like after this program

Lead with the **secured** contributions — **N1** (invariance identifiability proof), **N4**
(within>cross gap), **N5** (actionable recalibration, now with CORAL) — which no reviewer can attack.
Present **N3/N7** (the difficulty predictor + SDI) at the scope X1/X2/X3 establish: if they pass, it is
"a training-free statistic forecasts *deep-model* cross-subject failures across 9 cohorts"; if they
partly fail, it is "…forecasts failures where cross-subject accuracy has headroom, in a linear
representation" — still novel, and *defensible* precisely because you tested it to destruction. Fold the
descriptive modules in as a labelled **dataset atlas** with an explicit negative finding. Report the
**leakage audit (N2)** and the audit-trail itself as methodological contributions. State the one-lab
temporal caveat (N6) up front. That paper is narrower than ROADMAP §1 and much harder to sink.

---

*Prepared as an independent external review (v2). The only project file created or modified is this
report itself. Line/number citations refer to the state of the repository on the review date above; the
v1 findings were independently cross-checked against the code and committed numbers and all load-bearing
ones were confirmed.*

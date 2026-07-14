# PROFILE — what to do when the corrected `results/` come back

> **BANNER (2026-07-12): the whole X1–X15 suite has RUN. Stage 2 below (the floor-effect confound)
> is RESOLVED; two more Stage-2-class risks are closed too.** Authoritative verdicts now live in
> **`STATE.md` §5.6**, not here.
> - **X1 (floor):** ceiling −0.581 [−0.751,−0.292] **primary**; variance +0.436 [0.228,0.617]
>   **secondary (partly range-restriction)**; class-match survives **6/6**. Full 14-dataset run
>   (398 rungs / 9 cohorts) — *supersedes the 8-dataset −0.644/+0.468 numbers this banner used to quote.*
> - **X2 (coupling):** cross-basis r stays negative + significant on 7 datasets → the MMD↔LOSO link is
>   **not** a same-feature-space artifact.
> - **X5 (amplitude):** difficulty r survives the de-amplituded basis on 13/14 → shift is structural.
>
> **⚠ THE ONE REMAINING BLOCKER IS X3 — AND IT NEVER RAN.** All 14 X3 outputs report
> `target_source: self_lda_loso_proxy`: `load_paper1_loso` filters on a key `"accuracy"` while
> BENCH-LOSO writes **`"test_acc"`**, so it matched 0 rows and fell back silently. The EMAHA-DB1 DL
> LOSO sweep **exists** (25 subjects × 4 models). Fix the key, re-run X3 + Module 5 + exp_D + exp_E on
> emaha_db1 with the DL target, **then** figures (never built — `results/figures/` does not exist),
> then write-up. Pre-register both branches before running it (see `STATE.md` §6.1).
>
> Rewritten 2026-07-10. The previous version was a pre-run checklist written before the
> correctness audit; its assumption that "Phase A gives you ~80% of the paper" no longer holds.
> **Read `CORRECTIONS.md` first.**

The recompute does not produce a paper. It produces a set of numbers whose *values decide which
paper we can write*. This is a decision tree, not a checklist.

---

## Stage 0 — the gate (mechanical, ~5 min)

```bash
python validate_results.py        # must exit 0
```

Then grep the two run logs for three things that should each appear **zero** times:

| grep for | meaning if present |
|---|---|
| `[FAIL]` | an experiment crashed — fix and re-run that one |
| `WARNING h_divergence without trial groups` | a call site passes no trial ids → its d_H is leaked |
| `WARNING subject_shift_stats without trial ids` | same, one level up |

And one thing that must appear **exactly four times** (the sub-800 Hz sets: `ninapro_db1`,
`ninapro_db5`, `myobit`, `senic`):

```
<ds>: window=NN samples < 200 -> complexity features masked to NaN
```

A fifth dataset there means a manifest `fs` or `WINDOW_MS` is wrong.

---

## Stage 1 — the six numbers that decide the paper

Every one of these was previously wrong. None is a formality. Do not start writing until each
branch has been taken.

### 1.1 `module2` — how much of "separability" was leakage?

Compare `knn_trial_cv_acc` (within-subject, trial-grouped) with `knn_loso_acc` (subject-disjoint),
and both with the archived `knn_loo_acc`.

- **Expect** emaha_db1's old 0.42 to fall toward its true LOSO of ~0.25.
- **If the drop is small (<0.05 pp):** the leak was mild on real data — windows within a trial may
  be less redundant than the synthetic worst case. Say so honestly; the fix still stands.
- **If `knn_loso_acc` > `knn_trial_cv_acc` anywhere:** the trial grouping is broken. Stop and fix.
- The `within_minus_cross` column **is a result**, not diagnostics: within-subject separability
  systematically overstates cross-subject. That is `fig01` and it is a new headline.

### 1.2 `module3` — is H-divergence still pinned at the ceiling?

- **Expect** `inter_subject.hdiv_frob` to fall well below the old 0.72–0.98 band.
- **If it is still >0.9 everywhere:** trial ids aren't reaching `h_divergence`. `validate_results.py`
  warns on exactly this.
- Either way, the Phase-1 claim *"subjects are highly distinguishable in feature space everywhere"*
  must be **retracted or restated with the corrected number**. It was a classifier memorising
  trial identity, not a property of EMG.

### 1.3 `block_c` E3 — is there a detectable shift at all?

Read `representations.pooled.shift_detectable` and `snr_excess_over_null` per dataset.

- **If `true` on most datasets** → read `mean_share_of_excess` (the Yoneda quantity, now honestly
  measured) together with `kl_excess_removed_by_subject_center` (how much divergence a per-subject
  mean re-estimation removes — the cheap calibration that actually helps cross-subject).
  **This is the salvaged novelty pillar.**
- **If `false` on many datasets** → *that is the finding.* Once the within-subject estimation-noise
  floor is subtracted at a matched trial/row budget, the between-subject Gaussian-KL is not
  distinguishable from noise in this feature basis. Strong, publishable negative result: shift is
  real under MMD/kNN but not under a Gaussian second-moment model. **Do not bury it.**
- Confirm `affine_invariance_check.ridge_breaks_invariance == true` on every dataset. That one
  field is the quantitative evidence for the methodological contribution (Stage 4).

### 1.4 `module5` / `meta` — does the difficulty predictor survive correction?

- Quote **`primary_pearson_r`** (MMD, fixed a priori). **Never** `best_predictor_posthoc`.
- Quote **`combined_cv_r2`** (leave-one-subject-out). It may be **negative** on the n=10 sets.
  A negative out-of-sample R² is legitimate and must be reported as such.
- In `meta.meta_analysis`: read `n_significant_fdr_correct_sign` and `n_wrong_sign`.
  The old "significant on 9/14" is dead; expect roughly **2/14 after FDR** (ninapro_db1, db2).
- Report `pooled_one_per_cohort` (9 cohorts) alongside the headline pooled r, plus
  `pooled_without_outliers` (senic excluded) and the `heterogeneity_warning` (I² ≈ 0.79).

### 1.5 `block_c` E2 — the A4 result (most likely to survive cleanly)

Confirm the ratio is still 2.4–3.6× with Mann-Whitney `p_value` < 0.05. State plainly that it
rests on the **GrabMyo family + senic** — ~2 independent cohorts — and that senic's "sessions" are
electrode-shift/fatigue conditions, not days, with most senic subjects having only one.

### 1.6 `actionability` — accept the null result

Read `guided_advantage` **against** `oracle_ceiling`. If the ceiling is ~1 pp (it was), no
allocation policy can help, and the honest statement is: *the difficulty signal is real but too
weak to guide a calibration budget at k ≤ 5.* Then read `mmd_vs_calibration_gain`, which tests the
premise the entire policy rested on. If it is ≈ 0, that is more interesting than the policy.

---

## Stage 2 — the one experiment the audit did NOT resolve  ⚠ BLOCKS THE WRITE-UP

**The floor-effect confound.** The predictor works best exactly where LOSO accuracy is near chance
and fails where accuracy is healthy:

| dataset | classes | LOSO acc | MMD r |
|---|---|---|---|
| ninapro_db1 | 53 | 0.12 | **−0.77** |
| ninapro_db2 | 50 | 0.14 | **−0.73** |
| grabmyo | 17 | 0.70 | +0.03 |
| emaha_db4 | 8 | 0.33 | −0.01 |

A reviewer will ask whether we predict *difficulty* or *distance from the accuracy floor*. Nothing
in the current results answers it. Three cheap probes (~1 h CPU, reuses cached frames):

1. **Match task difficulty.** Sub-sample ninapro_db1/db2 to grabmyo's 17 classes; re-fit the
   correlation. r stays ≈ −0.75 → not a class-count artifact. r collapses → it was riding on the
   53-class floor.
2. **Match the accuracy range.** Cripple grabmyo (fewer channels, or a shorter window) until its
   LOSO accuracy lands near 0.15; re-fit. r turns strongly negative → the predictor only works
   near the floor. A genuine, reportable limitation.
3. **Residualise.** Check whether the SDI's leave-one-cohort-out ρ is uniform across the accuracy
   range or concentrated on the low-accuracy datasets.

Whichever way it lands becomes a paragraph. Landing (2) gives the honest headline:
*"cheap distribution statistics predict who a model fails on — but only once the model is already
failing."* Still a real contribution, and far more defensible than pretending the effect is uniform.

---

## Stage 3 — figures (only after Stages 1 and 2)

`python run_phase2.py --exp figs` builds 12 figures from the JSONs (`dsprofile/figures.py`;
opt-in, never part of `--exp all`, already wired to the corrected keys).

Priority order — drop the rest if time is short:

1. `fig01_within_vs_cross_subject` — the generalisation gap. **New headline.**
2. `fig02_forest_difficulty` — per-dataset r + 95 % CI + pooled diamond; filled = survives FDR.
3. `fig05_robustness_by_classifier` — difficulty is classifier-agnostic (the undersold result).
4. `fig04_a4_intersubject_vs_interday` — the A4 answer.
5. `fig03_mean_vs_covariance` — only if `shift_detectable` is true somewhere.
6. `fig06` + `fig07` — actionability and its premise (the honest null).

Then **open every PNG.** The palette validator checks colour, not layout.

---

## Stage 4 — write-up

The paper the corrected numbers support is narrower and more defensible than ROADMAP §1:

> Subject difficulty in sEMG is a **classifier-agnostic property of the data**, partially
> predictable from cheap distribution statistics — reliably on high-n datasets, and (pending
> Stage 2) possibly only where cross-subject accuracy has little headroom. **Inter-subject shift
> exceeds inter-day shift by 2–3×** where both are measurable. **Within-subject separability
> systematically overstates cross-subject separability.**

Two of ROADMAP's four novelty hooks are gone and must not be resurrected:

- ✗ **Hook 3 — "mean-vs-cov explains why z-score helps."** The estimator is affine-invariant; the
  contrast was never identifiable. Replaced by the *per-subject* normalisation contrast.
- ✗ **SDI as an actionable tool.** The oracle ceiling is ~1 pp. It is a predictor, not a tool.

What we gained instead, and should lead the Methods with:

- ✓ **A methodological contribution.** The Gaussian-KL mean/cov split is invariant to any global
  affine reparameterisation, so the raw-vs-normalised comparison used in this literature is not
  identifiable. We prove it, exhibit the ridge artifact that hides it, and give a corrected
  estimator with a matched, trial-disjoint null floor.
- ✓ **A leakage audit.** Trial-grouped CV, plus a demonstration that H-divergence computed over
  shuffled overlapping windows saturates at its maximum regardless of the true divergence. Both
  practices are common in this literature.

State up front (most were in ROADMAP §6; now with numbers): the 14 datasets are **9 cohorts**;
A4 rests on ~2; healthy subjects only (scope, not a gap); `senic` is an unexplained sign reversal,
reported separately; the difficulty target is a self-computed classical LOSO proxy, not Paper-1's
DL LOSO.

---

## Stage 5 — optional, only if a reviewer asks

- Window-length robustness (re-run key blocks at 100 ms; `WINDOW_MS_SECONDARY` already exists).
- Swap Module-5's target from the self-LDA-LOSO proxy to Paper-1's DL LOSO once the BENCH-LOSO
  EMAHA sweep lands (`load_paper1_loso` is written and format-tolerant).
- Entropy/complexity as extra SDI predictors — only on the 10 datasets where entropy is defined.

---

## STATUS LOG

- **2026-07-10** — Rewritten after the correctness audit (`CORRECTIONS.md`) and the adversarial
  code review. Awaiting corrected `results/` from the box. **Stage 2 (floor-effect) is the single
  most important open item and blocks the write-up.**

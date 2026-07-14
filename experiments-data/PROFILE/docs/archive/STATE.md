# PROFILE (Paper 2 — data-science characterization of sEMG datasets) — STATE

**Updated:** 2026-07-12
**Phase:** core results corrected + validated (0 failures); add-ons A/B/C/D/E complete; **the full
`EXTERNAL_RESEARCH_REVIEW.md` §14 program (X1–X15) has now RUN on all 14 datasets** —
`results/floor_effect_x1/` + `results/x2 … x15/`, zero missing datasets, zero crashes. Per-experiment
verdicts are in **§5.6** (new).
**Floor-effect: RESOLVED (X1 re-run on all 14 datasets / 398 rungs / 9 cohorts — supersedes the
8-dataset/248-rung numbers previously quoted here).** Ceiling partial −0.581 [−0.751, −0.292] AND
variance partial +0.436 [0.228, 0.617]; both cohort-clustered CIs exclude 0. The negative r
**survives class-count matching on 6/6** (`x1a_collapses: []`) → NOT a class-count artifact. Only
grabmyo is floor-dependent; emaha_db1 / fors_emg / grabmyo_flow_static work at full accuracy.
`results/floor_effect_x1/pooled.json`. **The file's own caveat, which the write-up must carry:** the
+variance term is partly range-restriction → quote the **ceiling term as PRIMARY, variance as
SECONDARY**.
**⚠ THE ONE REMAINING TIER-0 BLOCKER — X3 (DL-target swap) DID NOT ACTUALLY RUN.** All 14 X3 JSONs
report `"target_source": "self_lda_loso_proxy"`: it silently fell back and merely re-derived the
proxy correlation. Cause: `load_paper1_loso` (`dsprofile/module5_difficulty.py:101`) requires a JSON
key `"accuracy"`, but BENCH-LOSO writes **`"test_acc"`** — so it matched 0 rows. The EMAHA-DB1 DL
LOSO sweep **already exists** (`experiments-dl/BENCH-LOSO/runs/rows/`: **25 subjects × 4 models** —
bilstm / cnn1d / ecnn / tcn, `protocol="loso"`). A one-line key fix unblocks the single
highest-value experiment in the whole program. See §6.
**Authoritative docs:** `CORRECTIONS.md` (what was wrong) · `NEXT_STEPS.md` (decision tree) ·
`ROADMAP.md` (plan — hook 3 is struck through) · `EXTERNAL_RESEARCH_REVIEW.md` (independent review +
detailed experiment program) · `RESULTS_REVIEW.md` (⚠ SUPERSEDED)

---

## 1. Where this stands in one paragraph

Phase-1 + Phase-2 ran to completion on the box (14 datasets, zero crashes). A line-by-line code
audit found 12 defects; an adversarial review found 3 more (2 in the fixes themselves); reading the
*corrected* results then exposed 1 blocker and 3 statistical repairs. All fixed, pinned by **214
green tests**. The corrected core results are SOLID (§2). The **floor effect is now RESOLVED** (§5):
the predictor's strength is governed by two opposing forces (a dominant ceiling effect + a secondary
variance effect). All **five add-on experiments are now complete** (A/B/C finished on the box; D/E were missing from
`results/experiments/` and were re-run 2026-07-11 to persist their JSONs). §5.5 records each verdict.
The remaining work is figures + write-up. **UPDATE (X1, on the box 2026-07-11): the floor effect is now
RESOLVED WITH PROOF.** The old single-trend `floor_effect.json` did self-contradict (grabmyo +0.95 vs
ninapro_db2 −0.93), but the proper analysis `floor_effect_x1.py` (dataset-clustered bootstrap, 248
rungs / 8 datasets) CONFIRMS the two-term model (ceiling −0.644 [−0.811,−0.221]; variance +0.468
[0.204,0.663], both CIs exclude 0) AND shows the negative r survives class-count matching on 5/5
datasets — so it is NOT a class-count artifact. `results/floor_effect_x1/pooled.json`.

---

## 2. Results that are SOLID (quotable after the re-run confirms nothing shifted)

| # | Result | Numbers |
|---|--------|---------|
| 1 | **Methodological contribution.** The Gaussian-KL mean/cov split is exactly invariant to any global affine map, so the raw-vs-z-scored contrast used in the literature is **not identifiable**. | invariance holds to **3.1e-12 … 1.5e-03** (13/14 below 1e-8); the `+1e-3·I` ridge breaks it by **8 %–187 %** on all 14 |
| 2 | **Within-subject separability systematically overstates cross-subject.** *(new headline)* | emaha_db1 `knn_trial_cv` 0.419 → `knn_loso` **0.159**; gap median **0.117**, max 0.315 |
| 3 | **Yoneda REFUTED** — between-subject divergence is **covariance-dominated**, not mean-dominated. | `mean_share_of_excess` 0.031 – 0.384 (median **0.268**); `shift_detectable` on **14/14** |
| 4 | **A4: inter-subject shift > inter-day shift.** | ratios **2.40** (grabmyo), 3.64 / 3.28 (flow), 3.60 (senic); rank-biserial 0.80–0.89 |
| 5 | **Difficulty prediction survives, narrowly.** | FDR: **3/14 correct-sign significant** (ninapro_db1 q<0.001, ninapro_db2 q<0.001, fors_emg q=0.018); senic significant at the **wrong sign** (q=0.006). Pooled r = **−0.391** [−0.597, −0.136], I²=0.79; one-per-cohort (k=9) −0.382; without senic −0.465 |
| 6 | **Difficulty is classifier-agnostic.** | LDA/SVM/RF agree on who is hard, Spearman 0.47–0.94 |
| 7 | **Only `silhouette` explains dataset hardness** after FDR. | ρ=0.824, q=0.004. The excluded circular predictors scored ρ=0.947 / 0.938 — exactly as predicted |
| 8 | **Class difficulty transfers across subjects.** *(unplanned)* | rank stability ρ 0.27–1.00, median **0.73** — hard classes are hard for everyone |
| 9 | **K2: 1 kHz suffices.** *(unplanned, clean)* | 1000 Hz retains ~1.00 of accuracy; **500 Hz costs 4–13 %** (8 testable datasets) |
| 10 | **SDI actionability is a NULL result — coherently so.** | oracle ceiling **0.18–1.36 pp**. But `mmd_vs_calibration_gain` is positive+significant on exactly emaha_db5 (0.81), ninapro_db2 (0.72), senic (0.51) — the three where guided beats random. The policy works precisely where its premise holds |
| 11 | **SDI transfers weakly to an unseen cohort.** | leave-one-**cohort**-out mean ρ = **0.285** (0.338 without senic); 4/14 FDR-significant |
| 12 | **senic is an unexplained sign reversal.** | the data-volume confound is **NOT supported** by its own probe; report separately as an outlier |

**14 datasets = 9 independent cohorts** (grabmyo_flow_static/dynamic are one 20-subject cohort;
the four EMAHA sets one; ninapro_db4/db5 one). Every pooled statistic reports both k=14 and k=9.

---

## 3. Two claims in `CORRECTIONS.md` that I OVERSTATED — corrected here

1. **The kNN leak was real but immaterial in practice.** Old leaky `knn_loo` = 0.4213;
   new trial-grouped = 0.4193. A 0.002 difference. `MAX_WINDOWS_PER_CLASS=200` had already thinned
   ~245 trials/class to a few windows each, so the near-duplicates were mostly gone before CV ran.
   The synthetic worst case (1.000 vs 0.306) overstated the real-data effect. **The big number was
   never leakage — it was within-subject vs cross-subject conflation**, a different and more
   interesting sin. Say this plainly in the paper.
2. **"H-divergence uniformly high was an artifact" is too strong.** The fix *is* active (senic's
   minimum pairwise d_H = 0.25, myobit's = 0.735; under the leak nothing could be that low). But
   between genuinely different people d_H really does run 1.3–1.95. Phase-1's *qualitative* claim
   survives. What was an artifact is that the estimator returned ~1.97 even for two halves of the
   **same** distribution. A `hdiv_within_subject_null` diagnostic now measures exactly that.

Also: the old `hdiv_frob > 0.9` validator warning measured the wrong thing — `_frob` is RMS of
`offdiag / offdiag.max()`, a **uniformity** statistic (an all-equal matrix scores 1.0 regardless of
magnitude). Replaced with the within-subject null.

---

## 4. Fixes applied 2026-07-10 (after reading the corrected results)

| | Defect | Severity | Fix |
|---|---|---|---|
| F1 | `_repr_matrix` built `subject_center` on **raw-scale** features → rank-deficient PCA → `tr(Σ₁⁻¹Σ₀)` exploded (ninapro_db2 `sc_cov` = 2.8e9 vs `pooled` 3.1e3) → **`kl_removed_by_subject_center` = −597,703**. Reproduced: cond 12 → 3.9e7. | **BLOCKER** | global z-score now precedes every per-subject map. `mean_share_of_excess` / `shift_detectable` were never affected (they live in the already-z-scored `pooled` arm) |
| F2 | Invariance check "failed" on myobit at rel=1.53e-03 — actually 31.35 vs 31.33, i.e. float64 at cond≈1e13 (176 Hz → 44-sample windows) | MAJOR | conditioning-aware tolerance (`100·eps·cond`), reports `covariance_condition_number` + `numerically_limited`. Well-conditioned datasets still demand 1e-6 |
| F3 | A4 p-values **pseudo-replicated**: Mann-Whitney over 917 pairs built from 14 subjects → senic p = 3.9e-108 | MAJOR | subject-level **paired Wilcoxon** (n = subjects). Pair-level p retained as `pairwise_mannwhitney_p_PSEUDOREPLICATED`. The effect (rank-biserial 0.80–0.89) was never in doubt |
| F4 | `difficulty_cv_r2` = **−278** (emaha_db5), **−198** (emaha_db7): LOO-CV of a 4-predictor OLS on 9 training points | MAJOR | suppressed below 5 subjects/predictor (`combined_cv_r2: null`, raw value kept). `primary_pearson_r` valid at any n |
| F5 | `min_channels_95pct_loso = 2` for emaha_db5, whose full LOSO acc is 0.142 vs chance 0.100 | MAJOR | chance-corrected: `acc(k) ≥ chance + 0.95·(full − chance)`; returns `None` when the model isn't meaningfully above chance |
| F6 | validator's `hdiv_frob > 0.9` check measured uniformity, not magnitude | MINOR | replaced with `hdiv_within_subject_null` (d_H between trial-disjoint halves of ONE subject; must be ≈ 0) |

Files changed: `dsprofile/block_c.py`, `dsprofile/block_d.py`, `dsprofile/module3_shift.py`,
`dsprofile/module5_difficulty.py`, `validate_results.py`.

---

## 5. Floor effect — RESOLVED (`floor_effect_x1.py`, FULL 14-dataset run)

**Question:** is the difficulty predictor real, or does it only work because some datasets sit near
the accuracy floor? **Answer: a real predictor, NOT a class-count artifact, whose STRENGTH is
moderated by the accuracy headroom.** Three probes, each with a synthetic ground-truth control
(`python floor_effect_x1.py --selftest` = 7/7). Authoritative numbers: `results/floor_effect_x1/pooled.json`.

1. **X1a matched class count** — sub-sample high-class datasets down to ~17 classes; the negative r
   **SURVIVES on 6/6** (emaha_db1, ninapro_db1/db2/db4/db5, grabmyo_flow_dynamic; `x1a_collapses: []`).
   → the predictor is **not** a class-count/floor artifact.
2. **X1b matched accuracy** — cripple a healthy dataset toward the floor. **grabmyo is the only
   floor-dependent set**; **emaha_db1, fors_emg and grabmyo_flow_static keep the negative r at full
   accuracy** → the predictor is not floor-only.
3. **X1c cohort-clustered pooled model** — 398 rungs from **disjoint random** channel subsets (not
   nested), inference by **bootstrap over the 9 cohorts** (not the rungs):

| effect (partial \|r\|, cohort-clustered 95% CI) | value | status |
|---|---|---|
| \|r\| vs **mean accuracy** (controlling spread) | **−0.581 [−0.751, −0.292]** | **PRIMARY** — as accuracy nears ceiling, cross-subject difficulty compresses → predictor fades |
| \|r\| vs **accuracy spread** (controlling mean)  | **+0.436 [0.228, 0.617]** | **SECONDARY** — partly range-restriction (\|r\| is mechanically attenuated when outcome spread is small); do not sell it as an independent mechanism |

Both CIs exclude 0; a MixedLM (`|r| ~ mean_acc + acc_std + (1|dataset)`) agrees in sign on both terms.
**Do NOT quote the old `floor_effect.py` numbers (−0.870 / +0.763 over 29 NESTED rungs)** — pseudo-
replicated; the naive pooled p (1e-21) survives in the output only as a labelled `DO_NOT_QUOTE` contrast.
Two standing caveats printed by the file itself: 9 clusters make the bootstrap CI *solid evidence, not
proof*; and the difficulty target here is still the **self-LDA-LOSO proxy** — X3 on a DL target would
make this floor resolution model-agnostic.

---

## 5.6 ⭐ X-SUITE RESULTS (X1–X15) — the review's §14 program, RUN (2026-07-12)

All 14 datasets, no missing cells. **Net effect: the three risks the external review said "block
publication" (floor confound, representation coupling, amplitude-only shift) are now CLOSED by
evidence — except X3, which never ran.**

| X | Question | Verdict (from `results/`) |
|---|---|---|
| **X1** | difficulty, or distance-from-floor? | **Real predictor, accuracy-moderated.** §5. Not a class-count artifact (6/6 survive matching). |
| **X2** | is MMD↔LOSO mechanical (same feature space)? | **NOT mechanical.** Cross-basis r (predictor in amplitude basis A, target in shape basis B) stays negative *and* significant on emaha_db1 (−0.50), fors_emg (−0.52 / −0.66), ninapro_db1 (−0.74 / −0.72), db2 (−0.65 / −0.60), db4 (−0.68), db5 (−0.61 / −0.73), myobit (−0.43 B→A). Null on grabmyo; senic positive as always. → **review risk §9-2 defused.** |
| **X3** | does the cheap stat predict a **DL** net's failures? | **❌ DID NOT RUN** — `target_source: self_lda_loso_proxy` on all 14 (key-name bug, see header). **The one open Tier-0 item.** |
| **X4** | CORAL (covariance align) vs mean-centering? | **Mean-centering wins outright.** center helps **13/14** (matches exp_B); **CORAL alone helps 0/14** and often *hurts* (grabmyo −11.4 pp, grabmyo_flow_static −4.7 pp); center+CORAL never beats center alone. → the pre-registered "surprising, citable" branch: **mean alignment already captures the achievable gain; covariance alignment is not the lever.** Strengthens N5, and reconciles E3 (KL is cov-dominated) with exp_B (mean-centering helps a lot): **KL-share ≠ decision-boundary impact.** |
| **X5** | is "shift" just contraction amplitude? | **No — it is structural.** difficulty r survives in the shape/invariant basis on 13/14 (`shift_is_amplitude_dominated: true` only on ninapro_db5); inter-subject MMD is comparable across bases. → **review risk §9-4 defused.** |
| **X6** | do findings replicate in a learned representation? | **Yes, 11/14** (PCA + random-Fourier-feature embeddings). RFF is often *stronger* than handcrafted (emaha_db1 −0.59 vs −0.51; ninapro_db2 −0.79 vs −0.72). Fails on emaha_db4, grabmyo, ninapro_db4; senic reverses in every representation. → meets the 2023–25 "learned representation" reviewer expectation. |
| **X7** | is the result an MMD-bandwidth artifact? | **No.** Sign stable-negative on **12/14** under median-γ, γ=1/d and multi-kernel MMD. The 2 exceptions are the two known ones (grabmyo ≈0, senic reversed). |
| **X8** | is MMD actually the best cheap OOD score? | **Yes — earned, not assumed.** LODO Spearman across 9 cohorts / 311 subjects: **mmd 0.332** > energy 0.312 > mahalanobis 0.287 > kNN-dist 0.191 (`mmd_is_best_or_tied: true`). |
| **X9** | does `compatibility_mmd` predict real cross-dataset transfer? | **Honest null / negative result.** Only 4 label-compatible pairs exist, and **transfer accuracy is at or BELOW chance on 3 of 4** (ninapro_db4→db2 0.017 vs chance 0.020; emaha_db1→db4 0.104 vs 0.125; grabmyo_flow static→dynamic 0.081 vs 0.063). `mmd_vs_transfer r = −0.85` but **n=4, p=0.15** → uninterpretable. **Report as: cross-dataset transfer in a fixed handcrafted basis fails; the compatibility matrix stays `validated: false`.** Do not claim the −0.85. |
| **X10** | senic reversal + electrode-shift testbed | **Split verdict.** GrabMyo family: condition-shift predicts condition-difficulty (within-subject fixed-effects r **−0.37 / −0.58 / −0.66**), and **shape shift predicts better than amplitude shift** (−0.55 vs −0.27 on grabmyo) — a genuine shift-robustness result. **senic stays unexplained** (condition r = +0.006, null). ⚠ `within_condition_null` is **NaN (n=0)** for grabmyo_flow_dynamic and senic → the leakage control did not execute on those two; re-run before quoting them. |
| **X11** | which dataset properties moderate the effect? | **Between-dataset floor confirmed, and class count matters too.** `mean_acc` coef (Fisher-z) **+0.198, cohort-clustered CI [0.004, 0.712]** excludes 0 → higher-accuracy datasets have a *weaker* (less negative) r. `n_classes` coef **−0.366 CI [−0.55, −0.14]** → more classes ⇒ stronger predictor. `n_channels` n.s. ⚠ **The `interpretation` string inside the JSON is sign-flipped prose** (a hardcoded string, copy-pasted onto all three predictors) — read the coefficients, not the sentence. |
| **X12** | are the subsample caps converged? | **Yes, 14/14** (`converged: true`); the metric's std shrinks ~3–10× from n=50 to n=600. The caps are adequate — a reviewer question pre-answered. |
| **X13** | does centering hurt imbalanced subjects (exp_B's own caveat)? | **Untestable on this panel — say so.** 8/14 datasets have `median_imbalance = 1.0` (balanced) → the correlation is **NaN**. Where defined, nothing significant (emaha_db1 ρ=−0.37, p=0.072). → no evidence centering hurts, but the caveat is **not** discharged; state it as untestable rather than refuted. |
| **X14** | difficulty as an onboarding-cost curve (A8) | **A strong, paper-worthy figure.** Adaptive-LDA (τ=0.75, λ=0.9): emaha_db1 **k=0 → 0.249; k=1 → 0.493 (+24.4 pp); k=5 → 0.611**. One calibration rep roughly doubles accuracy — the shared x-axis the two-paper program wanted. |
| **X15** | coverage-guaranteed difficulty intervals (novel) | **Works.** Leave-one-cohort-out split-conformal: mean coverage **0.904** at nominal 0.90 (`conformal_valid: true`), better calibrated than the Gaussian interval. Per-cohort honesty: **senic under-covers (0.806)**. → turns a weak point-predictor into a *calibrated* one — the most novel single deliverable in the suite. |

**Reading across the suite:** the difficulty predictor (N3/N7) survived every de-confounding attack
that could be run — different feature basis (X2), de-amplituded basis (X5), learned embeddings (X6),
kernel choice (X7), class-count matching (X1a), and it beats the standard OOD baselines (X8). Its
honest scope is: **strong where cross-subject accuracy has little headroom, absent where it is high
(grabmyo), reversed on senic.** The two experiments that returned *nulls* — X9 (transfer) and X13
(imbalance) — are reportable as nulls. **X4 is the single most valuable NEW result:** covariance
alignment does not help, mean alignment does.

---

## 5.5 ⭐ EXPERIMENT ROADMAP & DECISION TREE (the current work)

Five add-on experiments strengthen the paper by converting descriptive findings into predictive /
actionable ones and pre-empting reviewer attacks. **None adds a new descriptive module** (that would
be scope creep). Status + how each result changes the plan:

### DONE

- **Floor (`floor_effect.py`)** — resolved above. Verdict: two-term (ceiling + variance) model.
- **D — predictor bake-off (`exp_D_predictor_ranking.py`).** MMD and KL-mean are **tied for best**
  (LODO ρ≈0.29); combining does NOT help out-of-sample; KL-cov predicts nothing (0/14); de-leaked
  H-div is weak (4 wrong-sign). *Internal consistency:* the predictors that work (mean/marginal
  shift) are exactly the ones E3 found subject-idiosyncratic; the one that fails (covariance) is the
  one E3 found shared. → **Report MMD as THE cheap statistic**, empirically justified.
- **E — permutation test (`exp_E_permutation.py`).** Assumption-free null matches the parametric p;
  **3 correct-sign FDR-significant + senic reversed** (matches the meta-analysis). → the headline
  survives a distribution-free test.

### DONE — A/B/C (completed on the box 2026-07-11; ACTUAL verdict at the top of each; the pre-registered "expect/plan" branches are kept below as the record of what was predicted)

**A — window-length robustness (100/250/500 ms).**
**✅ ACTUAL (`exp_A_summary.json`):** window-robust — difficulty r stays negative at all 3 windows on
**12/14** (exceptions: grabmyo ≈0/n.s. and senic wrong-sign, both already known); within>cross holds
on **13/14** (only grabmyo_flow_dynamic dips, and it sits near chance). → the hoped-for branch; a
one-sentence claim + a supplementary table.
*Purpose:* the defensive must-have; a reviewer
will ask if 250 ms is an artifact.
- **Expected / hoped:** `difficulty r stays negative at all 3 windows` on ~11-14/14, and the
  within>cross gap holds everywhere. → one sentence + a supplementary table; the paper is unchanged
  and better armored.
- **If a headline FLIPS sign at 100 or 500 ms on several datasets:** that is itself a finding —
  report the window-sensitivity honestly and pick 250 ms as the primary with a stated rationale
  (entropy stability, latency). Do NOT hide it.
- **Watch:** low-fs datasets (ninapro_db1 100 Hz, senic/ninapro_db5 200 Hz) at 100 ms = 10-20
  samples/window; features get noisy there. Expect their metrics to be the least stable — that is
  expected, not a bug. Some (dataset, 500 ms) cells may `[FAIL]` gracefully (short trials).

**B — per-subject mean-recalibration (closes the E3 loop).**
**✅ ACTUAL (`exp_B_summary.json`) — the BEST-CASE branch:** per-subject mean-centering **significantly
improves LOSO on 13/14** (mean **+3.6 pp**, up to +8.5 on grabmyo_flow_static; only emaha_db7 n.s. at
p=0.064). → E3 becomes an ACTIONABLE recommendation; promote it to a method contribution. NB this
**contradicts the "likely won't help" prediction below** — report the actual result. Framing caveat:
KL-share ≠ classifier impact (E3 calls the divergence covariance-dominated, yet mean-centering helps a
lot — the mean term, though a KL minority, disproportionately moves the LDA boundary).
*Purpose:* turn E3's "mean shift is a
component" into "does removing it IMPROVE accuracy?"
- **If `subject_center` beats baseline on the majority (significant):** strong result — E3 becomes an
  ACTIONABLE recommendation ("a cheap, unsupervised per-user mean alignment recovers X pp"). Promote
  it next to the E3 decomposition figure. This is the best-case and turns a descriptive block into a
  method contribution.
- **If it does NOT reliably help (likely, given E3 found divergence is covariance-DOMINATED):** that
  is the *honest, expected* result and still valuable: "the mean component is real but not the
  dominant obstacle; covariance mismatch is what limits cross-subject accuracy → future work is
  covariance alignment (CORAL), not just centering." Either outcome is a paragraph; the covariance
  finding is arguably the more interesting one and is consistent with D (KL-cov predicts nothing =
  covariance is shared, so aligning it is the lever).
- **Watch:** `median_class_imbalance_ratio` per dataset — a strongly imbalanced subject has a
  class-biased mean, which can make centering hurt. Report it as the caveat.

**C — cross-session difficulty prediction (temporal analog).**
**✅ ACTUAL (`exp_C_summary.json`) — the strong branch, with the one-lab caveat:** within-subject
fixed-effects r **< 0 and significant on 3/3 true-day datasets** (grabmyo −0.46, flow_static −0.67,
flow_dynamic −0.61; all p<3e-5); senic (electrode-shift conditions, not days) is null and reported
separately. → the SAME cheap statistic predicts BOTH who and when a model fails — BUT all 3 true-day
sets are the GrabMyo family (≈1 independent lab), so state that limit prominently as planned.
*Purpose:* extend the money result
from the subject axis to the day axis.
- **If within-subject r<0 and significant on grabmyo (the true-day cohort):** big — the SAME cheap
  statistic predicts BOTH who (subjects) and when (days) a model fails. Add it as a second axis of
  the headline; strengthens A3/A4 coverage.
- **If it is null or weak:** report honestly — "the predictor is a cross-subject tool; the temporal
  axis needs more sessions than 3 to resolve." Note only grabmyo(+2 flow variants, same cohort) and
  senic qualify, so this rests on ~1 true-day cohort — state that limit regardless of outcome.
- **Watch:** senic is a shift-condition axis, not days — the script reports it separately and
  excludes it from the headline. The fixed-effects df (N−n_subj−1) is already correct.

### DECISION GATE — A/B/C HAVE LANDED (2026-07-11) ✅

1. ✅ `results/experiments/exp_{A,B,C}_summary.json` read; verdicts folded in above.
2. ✅ All three took a publishable branch (A: robust; B: best-case actionable; C: strong but one-lab).
   D/E also complete (re-run + persisted 2026-07-11).
3. ✅ **Floor-effect confound SETTLED (X1, on the box 2026-07-11)** — the proper dataset-clustered
   analysis (`floor_effect_x1.py`) confirms the two-term model with CIs excluding 0 AND shows the
   negative r survives class-count matching on 5/5 datasets (not an artifact); grabmyo floor-dependent,
   fors_emg/emaha_db1 work at full accuracy. See §5. **The last blocker is cleared.**
4. → **NOW: figures** (`run_phase2.py --exp figs` + a floor-effect panel from `results/floor_effect_x1/`
   + a window-robustness panel) and **write-up** per `NEXT_STEPS.md` Stage 4. Optional: run the rest of
   the `paper_experiments/` suite (X2–X14) on the box to further armour the paper.

---

## 6. NEXT STEPS (in order) — where we are on the path to submission

DONE: the full audit + fixes (§4); `validate_results.py` = 0 failures; add-ons A/B/C/D/E (§5.5);
**the entire X1–X15 program (§5.6)**. The floor confound (§5), the representation coupling (X2) and
the amplitude-basis objection (X5) are all now **closed by evidence**.

**→ NOW, in strict order:**

### 1. ⚠ BLOCKER — make X3 actually run (the DL-target swap)
This is the last Tier-0 item and it is a **one-line bug**, not an experiment to design.

- **The bug:** `dsprofile/module5_difficulty.py:101` filters on `"accuracy" in d`, but every
  BENCH-LOSO row writes **`test_acc`** (plus `test_macro_f1`). Result: 0 rows matched → `load_paper1_loso`
  returned `None` → X3 fell back to the LDA proxy on all 14 datasets **without failing**.
- **The data is already there:** `experiments-dl/BENCH-LOSO/runs/rows/` holds **125 `protocol="loso"`
  rows**, of which **emaha_db1 is a complete sweep: 25 subjects × 4 models** (bilstm, cnn1d, ecnn, tcn).
  The other datasets (fors_emg, grabmyo, ninapro_db1/db2/db5) have **only 1 subject each** → they
  cannot support X3 and must stay on the proxy.
- **Do:** (a) accept `test_acc` (keep `accuracy` as a fallback) and let the caller pick the model
  (per-model *and* mean-over-models targets); (b) add the review's ground-truth checks that already
  exist in `paper_experiments/x3_dl_target.py` (fixture round-trip, subject-ID permutation → r≈0,
  shuffled-target null); (c) re-run **X3 + Module 5 + exp_D + exp_E on emaha_db1 with the DL target**.
- **Pre-register the branches now** (per §14.0, the antidote to premature "RESOLVED"):
  - *MMD predicts the DL nets' failures too (r<0, sig)* → **the central claim is earned**: a
    training-free statistic forecasts which users a trained deep model will fail on. Headline.
  - *Predicts LDA but NOT the DL nets* → **honest scope correction**: "the statistic tracks *linear*
    separability, not deep-model difficulty." Still publishable, and far better found now than in review.
  - Either way the claim is **emaha_db1-only** (n=25, 4 architectures) — state that limit up front.

### 2. Cheap repairs found while reading the X results (do alongside §1)
- **X10:** `within_condition_null` is `NaN (n=0)` for `grabmyo_flow_dynamic` and `senic` → the
  leakage control never executed there. Re-run those two before quoting X10's senic row.
- **X11:** the JSON's `interpretation` field is a hardcoded, sign-flipped sentence copy-pasted onto
  all three predictors. Fix the string so the write-up cannot quote it backwards.
- **Back-port the F-fixes, or document them as known.** `paper_experiments/code_fixes.py` implements
  F1 (MI-SU denominator), F3 (magnitude-preserving shift aggregate), F4 (median-γ MMD), F-dec
  (anti-aliased entropy decimation) **with ground-truth tests — but they were never applied to
  `dsprofile/`.** The committed core results therefore still carry: `su = 2*mi/hc`
  (`module2_separability.py:74`), `gamma = 1/d` (`module3_shift.py:38`), naive `x[..., ::step]`
  (`features_extra.py:253`), `_frob` uniformity scalars, and a **seed-blind frame cache**
  (`windows.py:29`, F5). Impact is low (all feed descriptive fields that failed FDR anyway, and X7
  proves the γ choice does not move the headline) — but this must be **either fixed or stated as a
  known limitation**, not left silent.

### 3. Figures — NOT BUILT YET (`results/figures/` does not exist)
`python run_phase2.py --exp figs` (12 figures, already wired to the corrected keys), **plus five new
panels the X-suite now supports**:
- floor-effect scatter (|r| vs mean accuracy, coloured by dataset; grabmyo's crippling sweep as a line);
- window-robustness panel (exp_A);
- **X4 recalibration bars** — baseline / center / CORAL / center+CORAL across 14 datasets (the "CORAL
  does not help" figure);
- **X14 calibration curve** — accuracy vs k (0→5), the shared Paper-1 x-axis;
- **X15 conformal coverage** — per-cohort coverage vs nominal.
Priority: `fig01` within-vs-cross (headline) → `fig02` forest → X4 bars → X14 curve → `fig05`
classifier-agnostic → `fig04` A4 → floor panel. **Open every PNG.**

### 4. Write-up — per `NEXT_STEPS.md` Stage 4, with the ledger below
Lead with what is now **secured**: **N1** the KL affine-invariance identifiability proof · **N4**
within-subject separability systematically overstates cross-subject · **N5** unsupervised per-subject
mean recalibration (+3.6 pp on 13/14, and **X4 shows covariance alignment does NOT substitute**) ·
**N2** the leakage audit. Then **N3/N7** (the difficulty predictor + SDI) at the scope X1/X2/X5/X6/X7/X8
established, with X15 turning it into a *calibrated* predictor. Report the nulls (X9 transfer, SDI
actionability) as nulls. State one-lab (GrabMyo) temporal scope, 9-cohort/I²=0.79 heterogeneity, and
the senic reversal up front.

**Do not start §4 before §1 lands** — the whole framing of N3 depends on whether the statistic
predicts a deep net or only an LDA.

---

## 7. Standing limitations to state up front

- 14 datasets are **9 cohorts**; A4 rests on ~2 (GrabMyo family + senic).
- senic's "sessions" are electrode-shift / fatigue **conditions**, not days; most senic subjects
  have one session; its difficulty correlation is sign-reversed and unexplained.
- Healthy subjects only (scope choice, not a gap).
- The difficulty target is a **self-computed classical LDA-LOSO proxy**, not Paper-1's DL LOSO.
  ⚠ `load_paper1_loso` is written but **silently non-functional** (it looks for a key `"accuracy"`;
  BENCH-LOSO writes `"test_acc"`) — this is why X3 fell back on all 14 datasets. Fixing it makes the
  DL target available for **emaha_db1 only** (25 subjects × 4 models); every other dataset has 1 LOSO
  subject in BENCH-LOSO, so the proxy limitation stands for them regardless.
- **Known, un-back-ported code defects** (correct versions exist + are ground-truth-tested in
  `paper_experiments/code_fixes.py`, but `dsprofile/` still runs the originals): MI symmetric-
  uncertainty divides by H(C) only (F1); MMD uses γ=1/d not the median heuristic (F4 — X7 shows this
  does not move the headline); entropy decimation is not anti-aliased (F-dec); the reported `*_frob`
  shift scalars are a **uniformity**, not a magnitude (F3); the frame cache key omits the seed (F5),
  so seed-robustness is unfalsifiable without deleting the cache. All feed descriptive fields that
  failed FDR — low impact, but **state them or fix them; do not stay silent**.
- `mean_share_of_excess` is Mahalanobis-normalised: a large mean shift along a high-variance
  direction contributes little. Correct normalisation, but say so.
- Entropy/complexity features are **undefined on 4 datasets** (ninapro_db1 100 Hz envelope,
  myobit 176 Hz, ninapro_db5 / senic 200 Hz) — 250 ms holds < 200 samples.
- I² = 0.79: the per-dataset effects are genuinely heterogeneous. Never quote the pooled r alone;
  pair it with the forest plot.

---

## STATUS LOG (append-only)

- **2026-07-10 (a)** — Audit of the completed results found 12 defects; adversarial review found 3
  more (2 in the new fixes). All fixed, 173 tests green. Results invalidated + recomputed on the box.
- **2026-07-10 (b)** — Read the corrected results. Found 1 blocker (F1, `subject_center` raw-scale
  PCA → `kl_removed = −597703`) and 3 statistical repairs (F3 pseudo-replication, F4 `cv_r2 = −278`,
  F5 near-chance channel criterion). Fixed + tested. **Floor effect tested and CONFIRMED as a real
  association** (|r| vs accuracy −0.59, p=0.033; not a power artifact) but not separable from class
  count at n=13 → intervention required. Corrected two overstatements in `CORRECTIONS.md` (§3).
  **Awaiting the partial re-run of 4 modules / 8 result dirs.**
- **2026-07-10 (c)** — Partial re-run completed on the box (module3/5 + block_c/d + action/faabos/
  sdi/meta). `validate_results.py` -> **0 failures, 1 warning** (myobit numerical, expected). All
  six fixes verified in the real numbers: F1 `kl_removed_by_subject_center` now 0.03-0.39 (was
  -597703); F3 A4 subject-level Wilcoxon p<1e-4 (was 1e-108 pseudo-replicated); F4 `combined_cv_r2`
  = null on n=10 sets; F5 channel criterion chance-corrected (3 datasets -> None near chance);
  h_divergence within-subject null 0.00-0.07 on all 12 testable datasets (was ~1.97). Headline
  stable: pooled r=-0.391 [-0.597,-0.136], 3/14 correct-sign FDR-significant + senic wrong-sign;
  silhouette the only surviving hardness predictor; SDI LODO rho=0.285. NOTE: `combined_cv_r2` is
  negative even on reliable datasets -> the 4-predictor model does not generalise; quote
  `primary_pearson_r` only. Built `floor_effect.py` (verdict logic validated on synthetic ground
  truth). **Only the floor-effect run remains before the write-up.**
- **2026-07-11** — Built + tested the five add-on experiments (D, E done locally; A, B, C built for
  the box). D: MMD/KL-mean tied as best difficulty predictor (LODO ~0.29), combining doesn't help,
  KL-cov predicts nothing (0/14) — the shift decomposition and the predictor tell the same story.
  E: permutation p ~ parametric p; 3 correct-sign FDR-sig + senic reversed (matches meta). A/B/C:
  window-length robustness, per-subject mean-recalibration (closes E3 loop), cross-session
  difficulty (temporal analog). All on shared `exp_common.py` infra (atomic writes, resume/skip,
  shard-safe `--collect`). Ground-truth tests: `tests/test_addon_experiments.py` 41/41; full suite
  214/214. C uses a proper fixed-effects within-subject correlation with corrected df (N-n_subj-1),
  not a naive pooled Pearson. **NEXT: run A/B/C on the box (A is the only frame-builder -> shard by
  dataset; B/C read-only, safe alongside).**
- **2026-07-11 (roadmap)** — Floor effect RESOLVED (two-term ceiling+variance model, §5). Built+tested 5 add-on experiments (D/E done; A/B/C running on box, sharded across 4 terminals). Added §5.5 EXPERIMENT ROADMAP & DECISION TREE: what each pending A/B/C result means and how to plan from it (every branch is publishable; no result is a failure). Rewrote §6 next-steps (stale re-run steps removed). 214 tests green. Awaiting A/B/C -> interpret -> figures -> write-up.
- **2026-07-11 (add-ons COMPLETE + D/E persisted + status verified)** — Verified experiment completion
  directly from `results/experiments/`. **A/B/C properly complete** (`_complete:true`, `_missing:[]`,
  14/14 per-dataset each, zero error/[FAIL] entries): A window-robust (12/14 diff-neg, 13/14
  within>cross); B center-helps 13/14 (+3.6 pp, up to +8.5); C 3/3 true-day significant (r −0.46/−0.61/
  −0.67), GrabMyo-only. **D/E were found ABSENT repo-wide** (searched by filename + unique content keys
  across all `E:/sEMG Research Enhanced`; only the two scripts existed) despite the prior log saying
  "done" — **re-ran both 2026-07-11**; outputs now committed (`exp_D_predictor_ranking.json`,
  `exp_E_permutation.json`) and MATCH the recorded numbers (D: MMD≈kl_mean LODO 0.291, combining
  doesn't help out-of-sample, kl_cov 0/14; E: B=20000, 3 correct-sign FDR-sig [ninapro_db1/db2/fors_emg]
  + senic wrong-sign, perm_p ≈ param_p). Reproducibility gap (missing D/E artifacts) CLOSED. Added
  `EXTERNAL_RESEARCH_REVIEW.md` (independent review; §14 = detailed experiment program w/ novelty +
  ground-truth checks) to authoritative docs. ⚠ Corrected the standing overclaim: floor-effect
  "RESOLVED" is CONTESTED — committed `floor_effect.json` self-contradicts (grabmyo +0.95 vs
  ninapro_db2 −0.93) and the two-term model is not in the repo; a proper analysis (X1) is now the one
  remaining blocker before figures/write-up. **NEXT: floor-effect X1 -> figures -> write-up.**
- **2026-07-11 (experiment suite BUILT + ground-truth PROVEN)** — Built the full
  `EXTERNAL_RESEARCH_REVIEW.md` §14 program as a hardened, tested package `paper_experiments/`
  (foundation `common.py` + X2–X14 + code fixes F1–F4/F-dec) plus `floor_effect_x1.py` (X1), to bury
  every limitation with a proven method: X1 floor-effect (matched-class / matched-accuracy /
  dataset-clustered pooled model), X2 representation-decoupling, X3 DL-target swap, X4 CORAL vs
  centring, X5 de-amplituded basis, X6 learned (PCA+RFF) representation, X7 MMD kernel/aggregation
  sensitivity, X8 OOD baseline bake-off, X9 cross-dataset transfer, X10 senic electrode-shift, X11
  meta-regression, X12 subsample stability, X13 imbalance stratification, X14 Adaptive-LDA calibration.
  Every module has a synthetic ground-truth selftest; `python -m paper_experiments.selftest` =
  **48/48 module checks + X1 PASS (exit 0)**. Delivered as 15 parallel-safe notebooks under
  `notebooks/` (`00_SELFTEST_ALL` first; each reads the cache read-only, writes `results/<tag>/`
  atomically) + `requirements.txt` (pinned deps, X15). Build-here / run-on-Ubuntu; no heavy compute run
  locally. **NEXT (on the box): 00_SELFTEST_ALL -> run X1..X14 in parallel -> fold verdicts -> figures.**
- **2026-07-12 (X-SUITE X1–X15 COMPLETE — read + folded in)** — The whole
  `EXTERNAL_RESEARCH_REVIEW.md` §14 program has RUN on all 14 datasets (`results/floor_effect_x1/` +
  `results/x2 … x15/`; zero missing cells). Read every result file. **Three of the review's blocking
  risks are now CLOSED by evidence:** the floor confound (X1: ceiling −0.581 [−0.751,−0.292] primary,
  variance +0.436 secondary/range-restricted, class-match survives **6/6**, now on 398 rungs / 14
  datasets / 9 cohorts — supersedes the 8-dataset numbers previously in §5); the predictor↔target
  **representation coupling** (X2: cross-basis r stays negative + significant on 7 datasets → not a
  same-space artifact); the **amplitude-basis** objection (X5: difficulty r survives the shape/invariant
  basis on 13/14). Also: X4 — **CORAL helps 0/14 and often hurts (grabmyo −11.4 pp) while mean-centering
  helps 13/14** → covariance alignment is NOT the lever; mean alignment already captures the gain (the
  pre-registered surprising branch; the single most valuable NEW result). X6 replicates in learned
  (PCA/RFF) embeddings 11/14. X7 sign-stable under median-γ/multikernel 12/14. X8 MMD is the best cheap
  OOD score (LODO ρ 0.332 > energy 0.312 > mahalanobis 0.287 > kNN 0.191) — earned, not assumed. X12
  caps converged 14/14. X14 adaptive-LDA: k=0 0.249 → k=1 0.493 (+24.4 pp) → k=5 0.611. X15 LOCO
  split-conformal coverage 0.904 @ nominal 0.90 (valid; senic under-covers at 0.806). **Nulls, reported
  as nulls:** X9 (cross-dataset transfer at/below chance on 3/4 pairs; the −0.85 MMD-vs-transfer r is
  n=4, p=0.15 → do not quote), X13 (imbalance untestable — 8/14 datasets are balanced → ρ is NaN).
  **⚠ THE BLOCKER FOUND: X3 NEVER RAN.** All 14 X3 JSONs say `target_source: self_lda_loso_proxy` —
  `load_paper1_loso` filters on a key `"accuracy"` while BENCH-LOSO writes **`test_acc`**, so it matched
  0 rows and fell back silently. The EMAHA-DB1 DL LOSO sweep **exists**: 25 subjects × 4 models
  (bilstm/cnn1d/ecnn/tcn) in `experiments-dl/BENCH-LOSO/runs/rows/`; the other 5 datasets have only 1
  LOSO subject each, so X3 will be emaha_db1-only. Also found: X10's `within_condition_null` is NaN
  (n=0) on grabmyo_flow_dynamic + senic (control never executed); X11's `interpretation` string is
  sign-flipped prose (the coefficients are right); the F1/F3/F4/F5/F-dec code fixes live only in
  `paper_experiments/code_fixes.py` and were **never back-ported to `dsprofile/`**; `results/figures/`
  does not exist (figures never built). **NEXT: fix the `test_acc` key → re-run X3 + Module 5 + D + E
  on emaha_db1 with the DL target → cheap repairs → figures → write-up.** See §5.6 + §6.
- **2026-07-11 (X1 FLOOR EFFECT RESOLVED WITH PROOF — on the box)** — Ran `floor_effect_x1.py` on 8
  datasets (`--selftest` 7/7 first). **Result (`results/floor_effect_x1/pooled.json`): TWO-TERM model
  CONFIRMED** by dataset-clustered bootstrap over 248 rungs / 8 datasets — ceiling partial |r|~mean_acc
  = −0.644 [−0.811,−0.221] AND variance partial |r|~acc_std = +0.468 [0.204,0.663], both 95% CIs exclude
  0. **X1a: the negative r SURVIVES class-count matching on 5/5** (ninapro_db1/db2/db4/db5, emaha_db1;
  0 collapse; e.g. ninapro_db2 −0.75→−0.62 at 17 classes) → NOT a class-count artifact. **X1b: grabmyo
  is floor-dependent** (−0.06@0.70 → −0.44@0.22) but **fors_emg (−0.63@0.50) and emaha_db1 work at full
  accuracy**. The old `floor_effect.py` −0.870/+0.763 p<0.001 were pseudo-replicated (29 nested rungs) →
  do NOT quote; superseded. §5 rewritten. **The last write-up blocker is CLEARED.** `tests/
  test_floor_effect_x1.py` 14/14. NEXT: run the rest of `paper_experiments/` (X2–X14 + the new X15
  conformal) on the box, then figures + write-up.

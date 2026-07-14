# RESEARCH PREVIEW — what this paper claims, what the number is, and which file proves it

**Updated:** 2026-07-13 22:15 · every number below was read out of a JSON on that date.

**The rule this file lives by:** a claim may appear here only with (a) a number, (b) the path to the
file the number came from, and (c) an honest strength label. If a new result contradicts a claim
here, **the claim changes.** We do not re-run the experiment until it agrees.

Strength labels:
- **SECURED** — evidence across all 14 datasets, survives its de-confounding tests, ready to write.
- **PARTIAL** — real, but narrower than we would like, or a key control is missing.
- **OPEN** — a T-suite experiment is required before this may be stated at all.
- **NULL** — we looked, and there is nothing there. Reported as such, deliberately.

---

## 1. The thesis

Subject difficulty in sEMG is, to a measurable degree, a **property of the data rather than of the
model** — and several things the field routinely does to handle it are either unnecessary, or
actively wrong. We show this across 14 public datasets (9 independent cohorts, 311 subjects), with a
leakage audit and an identifiability proof that between them retract two analyses this literature
performs as a matter of routine.

---

## 2. Claims

### N1 — The mean-vs-covariance decomposition used in this literature is not identifiable · **SECURED**

The Gaussian-KL divergence between two subjects splits into a mean term and a covariance term. Papers
compare that split on raw versus z-scored features and conclude which moment "dominates". **That
comparison measures nothing:** the split is *exactly invariant* to any invertible global affine map,
and a global z-score is such a map. We prove it algebraically and verify it numerically to a relative
error of **1e-11** on all 14 datasets. The only reason the contrast ever appeared to show something is
the small ridge (`+1e-3·I`) added for numerical stability — which breaks the invariance by **8 % to
187 %**, i.e. the "finding" was an artifact of the regulariser.

> Evidence: `results/legacy_v1/block_c/*__block_c.json` → `E3_meancov.affine_invariance_check`
> (`invariant: true`, `ridge_breaks_invariance: true`, 14/14).
> **Still to do:** write the one-page proof as an appendix. The numerics exist; the proof is not yet
> on paper.

### N2 — A leakage audit that invalidates two common practices · **SECURED**

(a) Cross-validation over 50 %-overlapping windows leaks: neighbouring windows share samples, so a
random split puts near-duplicates on both sides. (b) Worse, H-divergence estimated over such windows
**saturates at its maximum regardless of the true divergence** — a classifier memorises trial identity
and reports d_H ≈ 1.97 even for two halves of the *same* distribution. Both practices are common.

After trial-grouped CV, the within-subject d_H null falls to **0.00–0.28** (from ~1.97).

> Evidence: `results/legacy_v1/module3/*__shift.json` → `hdiv_within_subject_null`; the saturation
> demonstration is pinned in `tests/test_math.py`.

### N4 — Within-subject scores systematically overstate cross-subject scores · **SECURED**

On **14 of 14 datasets, with no exceptions**, evaluating on held-out *trials* of the same subjects
gives a materially better number than evaluating on held-out *subjects*. emaha_db1: 0.407 → **0.159**.
The gap ranges from 1 to 30 percentage points.

This is the cleanest result in the paper and the one with the most immediate consequence for how sEMG
work is evaluated.

> Evidence: `results/legacy_v1/module2/*__separability.json` → `knn_trial_cv_acc` vs `knn_loso_acc`;
> `within_minus_cross` column of the cross-dataset matrix. Robust to window length (14/14 at 100 /
> 250 / 500 ms — `results/legacy_v1/experiments/exp_A_summary.json`).

### N5 — Align the mean; do NOT align the covariance · **SECURED (and being sharpened by T3)**

An unsupervised, label-free intervention — each user subtracts the mean of their own unlabelled data —
recovers **+3.6 pp** of cross-subject accuracy on **13 of 14** datasets (up to +8.5 pp).

The surprise is the other half: aligning the full **covariance** (CORAL) helps on **0 of 14** and
sometimes hurts badly (grabmyo **−11.4 pp**). And this sits on top of an apparent paradox — the
between-subject divergence is *covariance-dominated* (the mean is only 3–38 % of it). So the moment
that carries most of the divergence is the one you must not touch, and the moment that carries little
of it is the one worth removing.

Our reading: the between-subject covariance difference and the class-discriminative covariance
structure are **the same structure**, so aligning it destroys the signal you are trying to classify.

> Evidence: `results/legacy_v1/x4/*__x4.json` (centring vs CORAL, paired Wilcoxon);
> `results/legacy_v1/experiments/exp_B_summary.json`; divergence split in `block_c` → `E3_meancov`.
> **T3** fills in the ladder between "mean" and "full covariance" to turn this into a rule. First
> real-data point (emaha_db4): z-score +2.6 pp > centring +1.7 pp > CORAL +0.4 pp.

### N6 — Between-subject shift exceeds between-day shift, by 2.4–4.3× · **PARTIAL (one lab)**

Where both axes are measurable, the difference between two people is far larger than the difference
between two days for one person: **2.37×** (grabmyo), **3.75×** / **3.49×** (grabmyo flow), **4.33×**
(senic). Subject-level paired Wilcoxon, p < 1e-4.

**Scope, stated up front:** only 4 datasets support this, and 3 of them are the GrabMyo family — one
lab. senic's "sessions" are electrode-shift conditions, not days. This settles a genuine disagreement
in the literature, but on a narrow base.

> Evidence: `results/legacy_v1/block_c/*__block_c.json` → `E2_a4_fair`; temporal analogue in
> `experiments/exp_C_summary.json` (3/3 true-day datasets, r = −0.46 / −0.61 / −0.67).

### N3 / N7 — A training-free statistic predicts which users a model fails on · **PARTIAL → T1 decides**

The claim the project has been calling its "money result". The honest version:

- Pooled across 14 datasets: **r = −0.399** [−0.579, −0.181]. The effect is real and the CI excludes
  zero.
- But it is individually significant on only **2 of 14** datasets (ninapro_db1, ninapro_db2), and
  **I² = 0.72** — the datasets genuinely disagree with each other.
- It survives every de-confounding test that was run: a disjoint feature basis (X2), a
  de-amplituded basis (X5, 10/14), learned PCA/RFF embeddings (X6, 10/14), the kernel choice
  (X7, 12/14), class-count matching (X1, 6/6), and it beats the standard cheap OOD scores
  (X8: MMD 0.332 > energy 0.312 > Mahalanobis 0.287 > kNN 0.191).
- Its strength is **moderated by accuracy headroom**: as a dataset's accuracy rises, the predictor
  fades (X1: ceiling partial r = **−0.581** [−0.751, −0.292], cohort-clustered).
- **The test that matters most was never run.** Every target is an LDA. Is this a property of the
  data, or of LDA? The one hint we have is encouraging (it predicts Random-Forest difficulty as well
  as LDA's: 6/14 vs 5/14 significant) but it is a side-result, not an experiment.

> **T1 settles this**, with five learner families across all 14 datasets. Its branches are
> pre-registered: model-agnostic (headline), linear-only (honest scope correction), or the families
> disagree about who is hard (in which case "subject difficulty" is not well defined and N3/N7 must be
> re-scoped). We will report whichever fires.
> Evidence today: `results/legacy_v1/meta/meta.json`, `module5/`, `x1/x2/x5/x6/x7/x8`,
> `robust_difficulty/`.

### The ADL claim — currently stated WRONGLY in our own documents · **OPEN → T4 decides**

EMAHA-DB1's 21 activities roll up into 5 coarse FAABOS categories. The coarse task scores 0.525 and
the fine task 0.253, and the project's documents read that as "coarse ADL categories are more
cross-subject robust."

**That comparison is invalid.** The coarse task also has an easier chance level (0.200 vs 0.048).
Corrected: fine captures 21.6 % of its available headroom, coarse captures 40.6 %. So coarsening may
genuinely help — but on one dataset, with one hand-made taxonomy, and with no control for the fact
that *any* merge of 21 classes into 5 removes exactly the confusions the classifier was making.

> **T4** does it properly: chance-corrected kappa, on all 14 datasets, with a **random-merge control**
> of identical shape. If the confusion-aware merge does no better than a random merge, then a
> hierarchy is not a robustness strategy — and that is a real, useful negative for the ADL literature.

---

## 3. Results we report as NULLS, on purpose

| Finding | Number | Why we publish it |
|---|---|---|
| **Cross-dataset transfer fails** | at/below chance on 2 of 4 label-compatible pairs; the MMD-vs-transfer r = −0.85 has n=4, **p = 0.15** → uninterpretable | It is the honest state of the art. **T5** adds the control X9 never ran (does per-subject alignment rescue it?) so that the null cannot be dismissed as "you forgot to normalise". |
| **The difficulty signal is not an actionable calibration budget** | best possible gain from perfect subject-targeting: **0.2–1.4 pp** | The signal is a predictor, not a tool. Claiming otherwise would not survive review. |
| **The imbalance caveat is untestable on this panel** | 8/14 datasets are perfectly balanced → the correlation is literally NaN | Reported as untestable, not as refuted. **T6** induces the imbalance to convert it into a curve. |
| **Conformal intervals are valid but not better** | coverage 0.908 at nominal 0.90 — but the plain Gaussian interval achieves 0.903 | The file says `conformal_better_calibrated: false`. Our own docs claimed otherwise. |

---

## 3b. THE NOVELTY MAP — what each running experiment buys, and what a failure costs

Read this next to the `pooled.json` verdicts when the results land. Every row is pre-registered: the
"if it succeeds" and the "if it fails" were both written before the experiment ran, and **both are
publishable**. Nothing here gets re-run until it says something nicer.

| Exp | If it succeeds → the novelty | If it fails → what we publish instead |
|---|---|---|
| **T1** model families | *"A training-free statistic forecasts which users **any** learner will fail on."* Kills the shared-representation objection outright and makes N3 the headline. | *"It tracks **linear** separability, not learnability."* An honest scope correction — still useful (deployed sEMG runs LDA), and far better found by us than by a reviewer. **Worst case (families disagree on who is hard): "subject difficulty" is not a well-defined property of the data, and N3/N7 must be re-scoped. That would be the most important negative result in the programme.** |
| **T3** moment ladder | *"Align the mean (and the per-channel gain); **never** the full covariance."* A rule another lab can apply, not just an observation — and it explains the paradox that the covariance carries most of the divergence yet must not be touched. | If no rung has majority support: per-subject alignment is not reliably useful, and N5 shrinks to a per-dataset observation. |
| **T4** ADL granularity | *"Hierarchical ADL taxonomies buy real cross-subject robustness — the benefit is structural, not a chance-level artifact."* The ADL headline. | **Currently the likely outcome.** *"Coarsening makes the task easier, not more subject-robust; a hierarchy is not a robustness strategy."* A real negative for the ADL literature — **and it retracts our own FAABOS claim.** |
| **T5** transfer | *"Cross-dataset transfer is blocked by a per-subject location shift, and removing it unblocks it."* Turns X9's null into a positive. | The null stands — but now **with the alignment control**, so nobody can say we simply forgot to normalise. |
| **T6** imbalance | *"Centring survives a class skew of up to N×."* Discharges N5's own caveat with a number, and gives a deployment rule. | *"Centring needs roughly balanced calibration data."* N5 keeps its precondition — the honest version of a claim is more useful than the loud one. |
| **T7** seeds | Every headline gets a ± and survives. Pre-empts the reviewer question at zero cost. | If signs flip across seeds, the **per-dataset FDR table is noise** and only the pooled effect may be quoted. This one **gates how every other number is read.** |
| **T8** budget | *"A new user's difficulty can be forecast from N seconds of unlabelled recording."* Turns a correlation into a **deployable protocol** — arguably the most practically useful sentence in the paper. | *"It is a research instrument, not a deployment tool."* |
| **T2** senic | The reversal is explained (target mis-specification / session confound) and senic stops being an unexplained outlier. | senic is a genuine counter-example: on some data, distinctive subjects really are *easier*. Costly but honest. |

**Batch 2 (T9–T11, built 2026-07-14, not yet run).** These add *content* rather than settling
interpretation — they are the three gaps a coverage audit found.

| Exp | If it succeeds → the novelty | If it fails → what we publish instead |
|---|---|---|
| **T9** feature families | *"The expensive nonlinear features do not survive the subject boundary."* Thirty years of sEMG feature engineering is defended on **within-subject** numbers — exactly the number this paper shows to be misleading (N4). The first multi-dataset **cross-subject** comparison of Hudgins TD4 / extended TD / frequency / Hjorth / entropy would be directly useful and money-saving. Alternative win: *"four features per channel is enough."* **The biggest content gap, and the closest to our own ML track.** (Smoke on emaha_db4: entropy scores **−0.062 kappa** vs cheap TD.) | Complexity *does* earn its cost cross-subject — equally publishable, and it would change our own pipeline. Or: no family separates, so feature choice is not the lever and **shift is** — which is the paper's thesis anyway. |
| **T10** rest inflation | *"Rest-inclusive accuracies are inflated by N pp, and the gain is one easy class dragging the mean up — gesture discrimination does not improve."* Quantifies what a large slice of the literature's headline numbers are worth, and turns our own protocol choice (drop rest) from a convention into a **finding**. Strengthens the leakage audit (N2). Applicable on 7/14 datasets. | Rest genuinely *helps* gesture discrimination (surprising, report it), or it changes nothing (then our choice is merely defensible, not a finding). |
| **T11** subject scaling | *"Collecting more users does not buy cross-subject accuracy."* The empirical foundation for the whole paper: if the curve saturates, the bottleneck is **shift, not data volume**, and every intervention we propose (align the mean, forecast the hard users, distrust within-subject numbers) is the right agenda. **The scaling law nobody has run.** | The curve keeps rising → the field's instinct (collect more people) is right, our panel is too small to see the ceiling, and that is an honest negative for our own thesis. |

⚠ **Do not quote any scaling result before T11 lands.** `meta.how_many_subjects` is *not* a scaling
curve — it measures the stability of the MMD estimate, and its values are MMD magnitudes, not
accuracies. That misreading was made once, on 2026-07-14, and retracted (see STATE.md history).

## 4. Standing limitations (state these up front, in the paper)

1. **14 datasets are 9 cohorts.** Every pooled statistic is reported both ways (k=14 and k=9).
2. **I² = 0.72** — the per-dataset effects are genuinely heterogeneous. Never quote the pooled r
   without the forest plot.
3. **senic reverses sign** and is currently unexplained (T2 is the attempt).
4. **The temporal axis rests on one lab** (the GrabMyo family).
5. **The difficulty target is a classical LOSO proxy.** Deep-model targets belong to another track
   and are out of scope here — stated as a scope decision, not hidden.
6. **Healthy subjects only.** A scope choice, not a gap.
7. **Every headline is a single seed (42).** T7 measures whether that matters.
8. **Two result folders (`module4`, `calibration`) predate the final code fixes.** Verified
   independent of every fix — their numbers stand — but they are not from the same code version as
   everything else.

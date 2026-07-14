# STATE — PROFILE (Paper 2: data-science characterisation of sEMG datasets)

**Updated:** 2026-07-14 01:20

**Status:** `BATCH 1 RUNNING · BATCH 2 READY`

- **BATCH 1 — T1…T8: DISPATCHED and running on the box** (4 terminals, 19 workers, started
  2026-07-13 ~23:55). Preflight passed, ground truth 48/48 at dispatch, and all 5 smoke experiments
  reproduced their local branches on the box. **Nothing to do until they finish** — then follow §4b
  exactly.
- **BATCH 2 — T9, T10, T11: BUILT, ground truth 62/62, smoke-tested on real data, NOT dispatched.**
  They must **not** run alongside Batch 1 (T9/T10 build new frames and will fight it for CPU). See
  §4d. Code needs copying to the box before they can run.

Whole suite: **11 experiments, 62/62 ground-truth checks, 233/233 core tests, validator 0 failures.**

**Smoke-test record (2026-07-13, all on `emaha_db4` unless noted). This is the evidence that each
experiment is correctly SET UP — not a result; one dataset is never a result.**

| Exp | Ran? | Time | What it produced | Sanity check |
|---|---|---|---|---|
| T1 | ✅ | 660 s | all 5 model families fitted (acc 0.318–0.326); rank agreement **0.865** | families agree on who is hard, as they should |
| T2 | ✅ | 672 s | ran; `build_pooled` correctly **declines a verdict** without senic present | the premise-void guard works |
| T3 | ✅ | 97 s | ladder complete | **reproduces frozen X4 to 4 decimals** (0.3247 / 0.3417 / 0.3284) |
| T4 | ✅ | 8 s | branch **B** | coarse beats fine (κ 0.364 vs 0.228) but a *random* merge gets 0.316 |
| T5 | ✅ | ~19 min | emaha_db1→db4, all 4 arms + subject-clustered CIs | **reproduces X9's below-chance failure**; alignment does not rescue it |
| T6 | ✅ | 2 s | branch **B** | centring benefit decays, turns negative at ~10× skew |
| T7 | ✅ | 117 s | branch **C** | r swings **−0.213 → +0.088** across seeds; sign unstable |
| T8 | ✅ | 5 s | branch **B** | 90 % of full strength needs ~100 s of unlabelled data |

**Read `CLAUDE.md` first.** It defines how this file may be edited. In particular: numbers in this
file must have been read out of a JSON *today*, and `results/LEDGER.xlsx` — not this file — is the
authority on every number.

---

## 1. Where the project actually stands, in one paragraph

The descriptive pipeline (Phase 1 + Phase 2), the X-suite (X1–X15) and the add-ons (A–E) have all
**run to completion on all 14 datasets** and are frozen in `results/legacy_v1/`. The validator passes
(0 failures). Four contributions are secured by that evidence and are ready to write up (§2). The
central claim — that a cheap, training-free statistic predicts which users a model will fail on — is
**real but weaker than the project's old documents said**: it is individually significant on only
2 of 14 datasets, and its strongest de-confounding test (does it predict a *non-linear* model's
failures, not just an LDA's?) **was never run**. The T-suite, built 2026-07-13, exists to close that
gap and six others. Until it runs, the paper's headline should be stated at the scope the evidence
supports, not at the scope the old documents claimed.

---

## 2. What is SECURED (quotable today, from the frozen tree)

Every number below was re-derived from the JSONs on 2026-07-13. Evidence paths are in
`RESEARCH_PREVIEW.md`.

| # | Claim | Number |
|---|---|---|
| **N1** | The Gaussian-KL mean/covariance split is **exactly invariant** to any global rescaling, so the raw-vs-normalised "which moment dominates" contrast used in this literature **measures nothing**. The `+1e-3·I` ridge is the only reason it ever appeared to. | invariance verified to 1e-11; the ridge breaks it by 8–187 % on 14/14 |
| **N2** | **Leakage audit.** Trial-grouped CV, plus a demonstration that H-divergence over 50 %-overlapping windows saturates at its maximum regardless of the true divergence. | within-subject d_H null now 0.00–0.28 (was ~1.97) |
| **N4** | **Within-subject separability systematically overstates cross-subject separability.** | 14/14 datasets, no exceptions. emaha_db1: 0.407 → 0.159 |
| **N5** | **Unsupervised per-subject mean-centring is actionable**, and — the surprise — **covariance alignment is not**. | centring helps 13/14 (+3.6 pp, up to +8.5); CORAL helps **0/14** and hurts grabmyo by 11.4 pp |
| **N6** | **Between-subject shift exceeds between-day shift**, 2.4–4.3×. | grabmyo 2.37, flow-dynamic 3.75, flow-static 3.49, senic 4.33 |
| — | 1 kHz suffices; halving the sampling rate costs 0–1.5 % of accuracy. | 9 testable datasets |
| — | One calibration repetition roughly doubles accuracy. | emaha_db1 0.249 → 0.493 |

## 3. What is NOT secured (and must not be written up as if it were)

| | Issue | The honest number |
|---|---|---|
| **N3/N7** | The difficulty predictor is **weaker than claimed**. | pooled r = **−0.399** [−0.579, −0.181], but individually significant on only **2 of 14** datasets (ninapro_db1, db2). I² = 0.72 — the datasets genuinely disagree. |
| | **senic reverses sign** and nobody knows why. | r = **+0.309** (no longer FDR-significant, but robustly positive in a *different* feature basis: +0.44, p = 0.007) |
| | The **model-agnosticism** of the predictor was never properly tested. | It predicts Random-Forest difficulty as well as LDA's (6/14 vs 5/14 significant) — a hint, not a result. **This is what T1 exists to settle.** |
| | The **ADL claim is wrong as stated.** | The docs said coarse FAABOS categories are more cross-subject robust. That compared 0.525 (5 classes, chance 0.20) against 0.253 (21 classes, chance 0.048) — an invalid comparison. **T4 does it properly.** |
| | **Cross-dataset transfer failed** and the alignment control was never run. | transfer at/below chance on 2 of 4 pairs; the r = −0.85 has n=4, p=0.15 → **uninterpretable**. **T5.** |
| | The imbalance caveat is **untestable on this panel**. | 8/14 datasets are perfectly balanced → the correlation is NaN. **T6 induces the imbalance instead.** |
| | **Seed robustness is unknown.** | every headline is a single draw at seed 42. **T7.** |

---

## 4. NEXT ACTION — dispatch the T-suite

Built and ground-truth green as of 2026-07-13 22:00. See `tsuite/` for each experiment's WHY,
pre-registered branches, and ground truth. Run per `CLAUDE.md` §6.

| | Experiment | Closes |
|---|---|---|
| **T1** | model-family target (LDA / RBF-SVM / RF / MLP / gradient boosting) | the biggest gap: is the predictor model-agnostic, or does it only track *linear* separability? **The CPU-only replacement for the retired X3.** |
| **T2** | senic root cause (4 hypotheses, 3 control datasets) | the unexplained sign reversal |
| **T3** | moment ladder (mean / scale / z-score / CORAL / whiten) | turns "centring helps, CORAL doesn't" into a *rule* |
| **T4** | ADL granularity, chance-corrected, with a random-merge control | the ADL claim, currently stated wrongly |
| **T5** | transfer after alignment | X9's null, without its missing control |
| **T6** | induced imbalance | X13's "untestable" |
| **T7** | seed robustness (4 seeds) | every headline is currently a single draw |
| **T8** | calibration budget in **seconds** | turns the statistic into a deployable protocol |

**Still to do after the T-suite:** the five missing figures (floor panel, window robustness, CORAL-vs-
centring bars, calibration curve, conformal coverage — none exist in `dsprofile/figures.py`); the
one-page affine-invariance proof appendix (N1 has numerics but no written proof); and a decision on
whether to re-run `module4`/`calibration` (2026-07-09/10, verified independent of every fix, so their
numbers stand — but they are the only two folders not produced by the current code version).

---

## 4a. THE ORDER OF OPERATIONS, end to end

```
   [NOW] Batch 1 (T1-T8) running on the box  ────────────────┐
                                                             │
   1. Batch 1 finishes                                       │
   2. Copy back results/v2/ + logs/            ──────────────┤  §4b
   3. build_ledger · validate · read the 8 pooled.json       │
   4. Update STATE.md + RESEARCH_PREVIEW.md                  │
                                                             │
   5. Copy tsuite/ + cli/ + the 3 .md files TO the box  ─────┤
   6. Run Batch 2 (T9, T10, T11) — 3 terminals               │  §4d
   7. Copy back results/v2/ + logs/ again                    │
   8. build_ledger · validate · read the 3 pooled.json       │
   9. Update STATE.md + RESEARCH_PREVIEW.md again            │
                                                             │
  10. Figures (5 missing panels) · N1 proof appendix         ┘  §4c
  11. Write-up
```

**Steps 2–4 and 7–9 are the SAME sequence (§4b below). Run it once per batch.**

## 4b. WHEN THE RESULTS COME BACK — the exact sequence (use for BOTH batches)

Do these in order. Do not skip step 2, and do not write a number into any document that you did not
read out of a JSON on the day you write it (CLAUDE.md §8d).

1. **Copy back** (AnyDesk) into this repo, overwriting:
   - `experiments-data/PROFILE/results/v2/`   ← the new results
   - `experiments-data/PROFILE/logs/`         ← the run logs + `tsuite_runs.jsonl` audit trail
   - Do **not** copy `results/_feature_cache/` (≈20 GB, regenerates itself).

2. **Regenerate the ledger and validate.** The ledger is the authority on every number; the markdown
   files are only prose about it.
   ```bash
   python -m cli.build_ledger                  # -> results/LEDGER.xlsx
   python -m cli.validate_results --root v2    # must exit 0
   ```

3. **Read the eight `pooled.json` verdicts** — `results/v2/<tag>/pooled.json`. Each one names the
   PRE-REGISTERED branch it landed on. **Write down the branch that actually fired, not the one we
   hoped for.** A branch B or C is a result, and it gets published.

4. **Check the panel-level guards before believing any headline:**
   - **T7 first.** If the sign of r flips across seeds on several datasets, then the per-dataset FDR
     table is noise and only the *pooled* effect may be quoted. Everything else is read in that light.
   - **T1's `mean_rank_agreement`.** If the model families disagree about *who* is hard, "subject
     difficulty" is not a well-defined property of the data and N3/N7 must be re-scoped — that
     outranks every other finding.
   - Every count in a verdict is reported **twice** (`n_datasets` and `n_cohorts`). The 14 datasets
     are 9 cohorts. Quote both, always.

5. **Update the three live documents, in this order:**
   - `STATE.md` — move each experiment from RUNNING to DONE with its date, branch, headline number,
     and evidence path; append a history entry in the §5 format.
   - `RESEARCH_PREVIEW.md` — add/amend each claim **with its evidence file**. If a result contradicts
     a claim already there, **the claim changes** — the experiment does not get re-run until it
     agrees.
   - Nothing else. `docs/archive/` is history and is never edited.

6. **Specific claims that are already on notice, pending these results:**
   - **The FAABOS/ADL claim** (`RESEARCH_PREVIEW` "the ADL claim"). T4's smoke already contradicts it.
     If T4 lands on branch B across the panel, the claim is **retracted** and replaced with the
     honest negative: coarsening makes the task easier, not more subject-robust.
   - **N3/N7's scope.** T1 decides whether the predictor is model-agnostic (headline), linear-only
     (scope correction), or ill-defined (re-scope).
   - **N5's precondition.** T6 decides whether "centre the mean" needs "…provided the new user's
     calibration data is roughly balanced" attached to it.
   - **Every single-seed number in the paper.** T7 decides whether they get a ± and a caveat.

---

## 4d. BATCH 2 — T9, T10, T11 (built 2026-07-14, ground truth 62/62, NOT yet dispatched)

**Do NOT run these while T1–T8 are still going.** They cannot corrupt anything (the cache keys
differ), but T9 and T10 each **build 14 new feature frames**, and that will fight the running jobs for
CPU. Run them as a second batch once the box is free.

| | Experiment | What it closes | Cost |
|---|---|---|---|
| **T9** | Which handcrafted feature families survive **cross-subject**? (Hudgins TD4 / extended TD / amplitude / frequency / Hjorth / entropy / our own basis / all) | The biggest hole in the paper, and the one closest to the group's ML track. `block_a` only ranked feature *reliability*; **nobody ever measured LOSO accuracy per family.** | builds the `complex` frames (the slow ones) |
| **T10** | How much accuracy does the **rest class** manufacture? | Every number here drops rest — correctly — but we never quantified what the field's rest-inclusive numbers are worth. Applicable on **7/14** datasets (only those ship a rest class). | builds 14 `rest0` frames |
| **T11** | Does collecting **more users** buy cross-subject accuracy? | **The scaling law nobody has run.** `meta.how_many_subjects` sounds like it and is NOT — it measures the stability of the MMD *estimate*, not accuracy. Reading it as accuracy is a mistake, and it was nearly made on 2026-07-13. | reads the cached frames; cheap |

**Run (after T1–T8 finish), 3 terminals:**
```bash
python -m tsuite.selftest                                  # must be 62/62
python -m tsuite.run --exp t9  --datasets all --jobs 7 2>&1 | tee logs/t9.log
python -m tsuite.run --exp t10 --datasets all --jobs 6 2>&1 | tee logs/t10.log
python -m tsuite.run --exp t11 --datasets all --jobs 6 2>&1 | tee logs/t11.log
```

**Smoke-tested on real data (2026-07-14):** T9 on emaha_db4 (165 s) → branch A; T10 on ninapro_db5
(48 s) → branch A, +1.3 pp raw inflation with gesture-only accuracy unchanged (+0.1 pp); T11 on
emaha_db4 (33 s) → branch C. T10 on emaha_db4 correctly self-reported `applicable: false` (that
dataset has no rest class).

**Two design flaws were caught by the ground truth / smoke before any real run — both recorded in the
code:**
1. **T10's obvious criterion was wrong.** "Raw accuracy rises but kappa does not" fails: an easy extra
   class raises kappa too (0.783 → 0.836), because the model genuinely classifies it. The invariant
   that isolates inflation is **gesture-only accuracy**, and every branch is now keyed on it.
2. **T11's row budget was not actually fixed** (320 rows at n=2 vs 2560 at n=16), so the curve would
   have confounded "more subjects" with "more data" — the exact confound the experiment exists to
   avoid. The budget is now whatever the *smallest* training size can supply, identical at every n.

## 4c. KNOWN OPEN ITEMS (nothing here blocks the run; none may be forgotten)

From the 2026-07-13 code review. The 2 critical and the 9 consequential majors are **fixed**; these
are what remains. Recorded so they cannot quietly disappear.

**Minor code issues (low impact, none affect a headline):**
- `kappa_chance` divides by zero if `n_classes == 1`; and K is the *global* class count while a LOSO
  test subject may not contain all K classes, so that fold's true chance level differs slightly.
- `t8._seconds_per_window(dataset)` ignores its argument (windowing is global) — the docstring claims
  it uses "the dataset's own" window length. Fix the docstring or drop the parameter.
- `subsample_train`'s floor of 2 rows/class means the "identical budget" guarantee bites unevenly on
  high-K vs low-K datasets when `2·K > cap_rows`.
- `cli/build_ledger.py` takes `first_run`/`last_run` from filesystem mtimes, which a file copy from
  the box rewrites. Better: stamp a timestamp into each JSON at run time.
- `t2` computes `H1_sign_flips_with_target` and never uses it.

**Deliverables still outstanding (after the T-suite lands):**
- **The five missing figures.** `dsprofile/figures.py` has 12 and none of them are: the floor-effect
  panel, window robustness, the CORAL-vs-centring bars (our best new result), the calibration curve,
  or the conformal-coverage plot. These must be *written*, not just run.
- **The N1 proof appendix.** The affine-invariance result has numerics on 14/14 but **no written
  proof**. It is our strongest contribution and it currently has no page.
- **`module4` / `calibration`** were produced 2026-07-09/10, before the final fixes. Verified
  independent of every fix (module4 uses scikit-learn's own NMI; calibration only borrows a basis
  helper), so their numbers stand — but they are the only two folders not from the current code
  version. Decide: re-run for one-code-version hygiene, or state the exception in the paper.

## 5. HISTORY (append-only — never edit an entry, only add)

- **2026-07-09 18:00 → 21:06 — Phase 1 first run.** Modules 1–5 on 14 datasets. *These numbers were
  later found to be leaked (overlapping-window CV; H-divergence saturating at its maximum).
  Superseded; archived outside the repo.*
- **2026-07-10 — Correctness audit.** 12 defects found by code audit, 3 more by adversarial review
  (2 of them in the fixes themselves). Reading the corrected results then exposed 1 blocker (F1:
  raw-scale PCA made `kl_removed_by_subject_center` = −597,703) and 3 statistical repairs (F3
  pseudo-replicated A4 p-values; F4 `cv_r2` = −278 from a 4-predictor fit on 9 points; F5 a
  near-chance channel criterion). All fixed and pinned by tests.
- **2026-07-11 — Add-ons A–E built.** Window-length robustness, per-subject recalibration,
  cross-session difficulty, predictor bake-off, permutation test. Also: the floor-effect claim in
  the then-current STATE.md was found to be **unsupported** — the committed `floor_effect.json`
  self-contradicted (grabmyo +0.95 vs ninapro_db2 −0.93) and its p-values were pseudo-replicated over
  29 *nested* rungs.
- **2026-07-11/12 — The X-suite (X1–X15) built** as a hardened package with a synthetic ground-truth
  selftest per module, and run on all 14 datasets.
- **2026-07-12 16:00–16:41 — Code fixes F1–F5 + F-dec back-ported** into `dsprofile/` (symmetric-
  uncertainty denominator; median-bandwidth MMD; magnitude-preserving shift aggregate; seed in the
  cache key; anti-aliased entropy decimation).
- **2026-07-12 22:15 → 2026-07-13 20:03 — THE CLEAN RUN.** Phase 1 + Phase 2 + X-suite + add-ons +
  X1 + 12 figures, re-run on the box in one consistent pass with the fixed code. `validate_results`:
  **0 failures**. This is the tree now frozen at `results/legacy_v1/` (see its `MANIFEST.md`).
- **2026-07-13 — FULL AUDIT of the frozen tree against the documents.** Findings:
  1. **The X-suite numbers never changed.** Verified by diffing the pre-fix snapshot against the
     post-fix one: X1, X4, X5, X6, X7, X8, X12 are *identical*. Reason: the X-suite carries its own
     maths (`paper_experiments/common.py` has always used the median-bandwidth MMD) and never used
     the buggy `dsprofile` functions. **Therefore the discrepancies between STATE.md and the results
     were never staleness — they were misreadings, present from the day they were written.**
  2. What the fixes *did* change: everything computed by `dsprofile`. FDR-significant datasets fell
     3 → **2**; senic's reversal fell from significant (q=0.006) to **not significant** (q=0.117);
     the SDI's significant datasets fell 4 → 2; the within-vs-cross gap improved to **14/14**.
  3. Five specific misquotes found and recorded in `results/legacy_v1/MANIFEST.md` §3 (X5 13/14 vs
     the real 10/14; X6 11/14 vs 10/14; X9 "3 of 4 below chance" vs the real 2 of 4; X15 "better
     calibrated" when the file says `false`; X11's sign-flipped prose).
  4. **X3 had been silently withdrawn** — its code and results deleted, `module5` rewritten to say
     "self-contained, no deep learning" — with no record of the decision anywhere.
- **2026-07-13 21:00 — Repository reorganised.** Loose scripts moved into `cli/` and `addons/`;
  results split into a frozen `legacy_v1/` and a live `v2/`; ten markdown files archived verbatim
  into `docs/archive/`; the superseded `floor_effect/` results and 8 stray test fixtures quarantined
  into `_superseded_DO_NOT_QUOTE/`. **Nothing was deleted.** All 247 existing tests still pass.
- **2026-07-13 21:30 — X3 formally retired, with a reason.** Deep learning belongs to another track
  and this repository is data-science-only. The scientific question behind X3 (does the statistic
  predict a *non-linear* model's failures?) is answered instead by **T1**, on CPU, with five learner
  families across all 14 datasets — a stronger test than X3's single deep model on a single dataset
  would have been.
- **2026-07-13 22:00 — T-suite (T1–T8) built.** Ground truth: **36/36 green**. The gate failed 8/36
  on the first attempt; all eight were flaws in the experiments' own synthetic constructions (an
  offset placed along the discriminative direction rather than a nuisance one; a "pure gain"
  corruption that also moved the mean; a "structureless" null whose random class centres in fact had
  structure), and all eight were caught **before any real data was touched**.
- **2026-07-13 22:10 — T3 smoke-tested on real data (emaha_db4).** Its `baseline` / `center` /
  `coral` rungs reproduce the frozen X4 numbers **to four decimals** (0.3247 / 0.3417 / 0.3284) —
  the new code is consistent with the frozen evidence. The new middle rungs already show something
  X4 could not see: on this dataset `zscore` (+2.6 pp) beats `center` (+1.7 pp).
- **2026-07-13 22:30 — INDEPENDENT CODE REVIEW of the T-suite, and the fixes.** A fresh reviewer read
  every T experiment against the shared maths and the protocol. It found **2 critical and 9 major**
  issues. All of the critical and the consequential majors are now fixed; the ground-truth gate is
  **45/45**. What it caught, and why each mattered:
  1. **CRITICAL — T1's pooled verdict could never have existed.** `fdr_bh` returns a *tuple*
     `(rejected, q)` and `pool_random_effects` returns the key `pooled_r`, not
     `pooled_r_random_effects`. Both were wrong, and `run.py` calls `build_pooled()` outside any
     try/except — so the flagship experiment would have crashed **at the finish line**, after five
     model families × 14 datasets of compute, with nothing to show. Fixed, and a `C.fdr_q` wrapper
     now makes the tuple mistake impossible.
  2. **CRITICAL — T4 leaked.** The coarse label mapping was built from a confusion matrix computed on
     a held-out subject's *true labels*, then scored with a LOSO loop in which that same subject took
     a turn as the test fold. Worse than the label leak: the merge was *selected* to collapse exactly
     the confusions the scoring model makes, so the headline quantity was biased upward **by
     construction**. Fixed by re-deriving the mapping inside every fold from training subjects only.
     **Proof the fix works:** on a synthetic with zero true structure, the confusion-merge's advantage
     over a random merge fell from **+0.057 to −0.010** kappa once the mapping was fold-nested.
  3. **The ground-truth gate never called `build_pooled()`** — which is *how* the T1 crash survived.
     `build_pooled` is where the pre-registered branch is decided; it is now exercised for all 8.
  4. Verdict logic that could fire on evidence it did not have: T3's branch A could declare "align the
     mean and nothing else" on **1/14** support; T2 would print "H1 CONFIRMED — senic's reversal is a
     target artifact" even when senic **never reversed** in the first place; T6 counted the *balanced
     control* as evidence that imbalance breaks centring; T8's "minimum budget" accepted a
     **sign-flipped** correlation and was a minimum-over-noise. All four now have explicit guard
     branches.
  5. T5's "raw" arm **was not raw** (each dataset was pre-z-scored with its own statistics, so a
     dataset-level alignment had already been applied before the comparison began), and its
     "beats chance" was a point estimate against 1/K on 50 %-overlapping windows. Now genuinely raw,
     with a subject-clustered bootstrap CI that must clear the *majority-class* rate.
  6. T8 budgeted the **reference pool** as well as the new user (no deployed system is data-poor on
     the training cohort) and computed `r` from the *mean of 10 draws* — a practitioner has one
     recording, not ten. Now the pool is full-size and `r` is per-draw, reported mean ± sd.
  7. **The 9-cohort structure was honoured nowhere.** "helps on 12/14 datasets" is really "7/9
     cohorts" and is a materially weaker claim. `count_both_ways()` added; T1 and T4 now report both.
  **Deferred, and recorded honestly:** FDR correction within each experiment's family of Wilcoxon
  tests (T3: 84 tests, T6: 70) is implemented (`C.helps_flags_fdr`) but not yet wired into T3/T6's
  counts; cohort reporting is not yet in T3/T6/T7/T8's pooled verdicts; T3/T6/T8 still lack a
  clean-data negative control. **None of these block the dispatch** — they change how results are
  *counted*, not whether the runs are valid — but they must land before anything is written up.
- **2026-07-13 22:40 — T4 smoke-tested on real data (emaha_db4), and it already contradicts the
  project's own ADL claim.** Branch **B**: coarse labels beat fine in chance-corrected kappa (0.364
  vs 0.228), but a **random** merge of identical shape does nearly as well (0.316) — a structure
  advantage of only +0.048, which does **not** clear the measured null bias of 0.08. On this dataset,
  coarsening is *easier*, not more subject-robust. One dataset is not a result, but it is exactly the
  outcome the corrected design was built to be able to detect.
- **2026-07-13 ~23:55 — BATCH 1 DISPATCHED on the box.** 4 terminals, 19 workers:
  `t1` (jobs 6) · `t3 t4` (5) · `t6 t8` (5) · `t2 t5` (3); `t7` to follow when a terminal frees.
  Preflight passed (20 cores, 74 cached fast frames reused, L1 auto-discovered). Ground truth 48/48
  on the box. The 5 smoke experiments reproduced their local branches exactly. Expected: T1 ≈ 3–6 h,
  the rest faster. Awaiting `results/v2/` + `logs/`.
- **2026-07-14 01:20 — BATCH 2 BUILT: T9, T10, T11. Ground truth 62/62.** These close the three
  content gaps found by auditing coverage (§4d): the feature-family question (the biggest hole, and
  the one closest to the group's own ML track), the rest-class inflation question, and the
  subject-scaling law. Smoke-tested on real data: T9 emaha_db4 165 s → branch A (entropy scores
  **−0.062 kappa** against cheap time-domain features); T10 ninapro_db5 48 s → branch A (rest inflates
  raw accuracy **+1.3 pp** while gesture-only accuracy moves **+0.1 pp**); T11 emaha_db4 33 s →
  branch C. T10 on emaha_db4 correctly self-reported `applicable: false` — that dataset has no rest
  class (only **7 of 14** do).
  **The ground truth caught two of my own design flaws before they touched real data, and the smoke
  caught a third:**
  1. T10's obvious criterion ("raw accuracy rises but kappa does not") is **wrong** — an easy extra
     class raises kappa too (0.783 → 0.836), because the model genuinely classifies it. Every branch
     is now keyed on **gesture-only accuracy**, the one quantity that isolates the inflation.
  2. T11's "fixed row budget" **was not fixed** (320 rows at n=2 vs 2560 at n=16), so the curve would
     have confounded "more subjects" with "more data" — the exact confound it exists to avoid.
  3. T10 initially fired "rest genuinely helps" on ninapro_db5 purely because kappa ticked up, while
     gesture accuracy was flat. Backwards. Branch B now requires gestures to actually improve.
- **⚠ 2026-07-14 — A CLAIM I MADE AND THEN RETRACTED, recorded so nobody repeats it.** While auditing
  coverage I read `meta.how_many_subjects` as a subject-scaling curve and reported that *"accuracy is
  FLAT in the number of training subjects — collecting more users buys nothing."* **That is false.**
  `how_many_subjects` (`dsprofile/meta.py:245`) measures the **stability of the inter-subject MMD
  estimate** as subjects are added to the matrix — a hygiene statistic about the estimator. Its values
  are MMD magnitudes, **not accuracies**. There is no subject-scaling law anywhere in the frozen
  results; **T11 is what actually answers the question, and it has not run yet.** Do not quote a
  scaling result until T11 lands.

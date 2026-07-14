# PROFILE — Phase-2 Plan (broad experiment suite covering the scientific questions)

Phase 1 (the 80%) is done and validated (`RESULTS_REVIEW.md`). Phase 2 adds our own novelty +
fixes the two methodology caveats + broadens coverage to the majority of the scientific questions.
Everything reuses the same infra (cached frames on the box, resume/skip, joblib parallel).

## Experiment suite (each maps to scientific questions; ★ = uses our own proposed method)

| ID | Experiment | Scientific Qs | Needs | Status |
|----|-----------|---------------|-------|--------|
| **E1 ★** | **SDI — Subject-Difficulty Index** (portable, LODO-validated) | A9 | module5 parquets (local) | **DONE + validated** |
| **E2** | **A4 fix** — within-subject cross-session vs cross-subject within-session | A4 | frames (box) | to build |
| **E3** | **Mean-vs-cov decomposition on RAW + normalised** features | A4, novelty | raw frame (box) | to build |
| **E4** | **Conditional-shift disparity matrix** (kNN label-disagreement) | A9 (both shifts) | frames (box) | to build |
| **E5** | **Feature-family analysis** (amplitude vs spectral vs complexity): separability, difficulty-prediction power, ranking stability across datasets | C1, C5, L4 | frames (box) | to build |
| **E6** | **Sampling-rate sufficiency** — decimate to 1000/500 Hz, re-measure separability | K2 | raw signal (box) | to build |
| **E7** | **Channel-reduction validation** — drop to mRMR minimal subset, measure separability retention | K1 | frames (box) | to build |
| **E8** | **Calibration curve** — self-LDA with k target reps → accuracy vs k | A8 | frames (box) | to build |
| **E9** | **FAABOS ADL-taxonomy profiling** (EMAHA hierarchical labels) | I2 | faabos frame (box) | to build |
| **E10** | **senic anomaly investigation** (why positive difficulty correlation) | methodology | senic frame + meta (box) | to build |

## Design decisions (locked, with reasoning)
- **E1 SDI** operates on **within-dataset-standardised** predictors (raw scales differ 1000×) and
  predicts *relative* within-dataset difficulty; validated **leave-one-dataset-out** (train 13,
  predict the 14th's ranking). Also test adding complexity/entropy summaries as extra predictors
  (does entropy improve the index beyond shift stats?). Report with/without the senic outlier.
- **Target** stays the self-LDA-LOSO proxy (Paper-2 self-contained); swap to Paper-1 DL LOSO later.
- **E3** raw-feature frame = a new cache variant (`normalize="none"` path) so we can show the mean
  term dominates on raw features (Yoneda) vs covariance on z-scored (why normalisation helps).
- **E5** families: amplitude/energy {MAV,RMS,WL,IEMG,SSI,DASDV,AAC,LOG,VAR,LOGRMS,NLE},
  spectral {MNF,MDF,SENT,MNP,TTP}, complexity {SampEn,FuzzyEn,fApEn,PermEn,HFD,Hjorth,MFL,skew,kurt}.
  Question: which family best predicts LOSO difficulty, and is the ranking stable across datasets?
- **E6/E7** are "sufficiency" checks: recompute Module-2 separability under decimation / channel
  subsets and report the accuracy-proxy retained.

## Build order (each smoke-tested locally, then run on the box)
1. E1 SDI — DONE (refine with entropy predictors + senic handling).
2. E2 + E3 + E4 — the distribution-shift completions (fix the review caveats). Highest value.
3. E5 feature-family — answers the ML-features questions (C/L).
4. E8 calibration curve, E7 channel reduction — deployment questions (A8/K1).
5. E6 sampling-rate, E9 FAABOS, E10 senic — coverage + the ADL angle.

## How it will run on the box (after copying updated code)
```bash
export SEMG_L1_ROOT=/home/honors/Ashraf_Ali_2025_Batch/data/L1
cd .../experiments-data/PROFILE
# Phase-1 caches on the box are reused; only new Phase-2 outputs are computed.
python run_phase2.py --exp all --datasets all --jobs 8     # (CLI built alongside the modules)
```
Resumable + parallel like Phase 1. E1 (SDI) needs only Phase-1 outputs (fast).

## Paper structure — all experiments map to 6 blocks (= paper sections)
- **Block A — Signal & features:** Module 1 + feature reliability (ICC across reps) + "does complexity
  add info beyond amplitude" (conditional MI).
- **Block B — Class structure:** Module 2 + universally-hard classes (cross-dataset) + subject-invariant
  vs idiosyncratic classes.
- **Block C — Distribution shift:** Module 3 + E2 A4-fix + E3 mean/cov-on-raw + E4 conditional disparity.
- **Block D — Channels & sensors:** Module 4 + E7 channel-reduction + E6 sampling-rate sufficiency.
- **Block E — Difficulty prediction (headline):** Module 5 + **E1 SDI** + E8 calibration curve.
- **Block F — Science of datasets (meta novelty):** what-makes-a-dataset-hard + dataset atlas +
  meta-analysis + how-many-subjects + transfer-compatibility.
- Woven in: FAABOS/ADL profiling (E9) + senic investigation (E10).

Decision (2026-07-09): Paper 2 is FULLY SELF-CONTAINED — BENCH-LOSO (DL) set aside; the difficulty
target is the self-LDA-LOSO proxy only.

## STATUS LOG
- **2026-07-09** — Phase-1 reviewed (14 datasets clean). Review caveats (A4, mean/cov, senic) logged.
- **2026-07-09** — Block E **E1 SDI** built + LODO-validated (mean ρ 0.29; ninapro_db1 ρ 0.82). `dsprofile/sdi.py`.
- **2026-07-09** — Block F **meta** built + RUN LOCALLY (`dsprofile/meta.py` → `results/meta/`):
  * what-makes-hard: kNN-LOO separability ρ=0.93, silhouette ρ=0.82 (>> #classes ρ=−0.61) drive dataset difficulty.
  * meta-analysis: pooled random-effects r=−0.39, 95%CI [−0.60,−0.14], k=14, I²=0.79 (senic outlier).
  * atlas: 14 datasets → 4 clusters (55% var in 2D); {myobit,senic} apart.
  * how-many-subjects: MMD estimate stabilises at ~22–30 subjects.
  * (transfer-compatibility still TODO — needs shared feature subset.)
- **NEXT (frame-dependent, for the box run):** E2 A4-fix, E3 mean/cov-raw, E4 conditional disparity
  (Block C); Block A feature-reliability + complexity-conditional-MI; Block B hard/invariant classes;
  E6/E7/E8/E9 (Blocks D/E); E10 senic. + `run_phase2.py` CLI. + atlas/meta figures.
- **2026-07-09 — ALL Phase-2 modules BUILT + TESTED (95/95 checks).** Files:
  `sdi.py` (E1), `meta.py` (Block F), `block_c.py` (E2/E3/E4), `block_a.py` (reliability + complexity-MI),
  `block_b.py` (per-class difficulty + subject-invariance), `block_d.py` (E7 channel-reduction + E6
  sampling-rate), `calibration.py` (E8), `faabos.py` (E9), `senic_probe.py` (E10), `transfer.py`
  (cross-dataset compatibility), `run_phase2.py` (CLI). windows.py gained `normalize`/`decimate` options
  + an O(n²)→O(1) fs-lookup fix. Tests: `test_math` 46, `test_phase2` 31, `test_blocks` 18 (synthetic
  frames validate math + I/O shapes). E1/meta/senic_probe verified on real local artifacts;
  senic anomaly = session-imbalance confound (loso & mmd both fall with session count).
  **Remaining: run `run_phase2.py` on the box; add atlas/meta/calibration figures; write-up.**
- **2026-07-10 — Added two DEPTH experiments to bulletproof the headline (built + tested, blocks 27/27):**
  * `robust_difficulty.py` (`--exp robust`): per-subject LOSO with 3 classifiers (LDA/SVM/RF) x 3 seeds
    -> do classifiers AGREE on who is hard, and does MMD-to-pool predict difficulty for EACH? Makes the
    difficulty notion classifier-agnostic (not an LDA artifact).
  * `actionability.py` (`--exp action`): SDI-guided calibration-budget allocation vs random vs oracle
    -> shows the cheap difficulty score is USEFUL (spend calibration on predicted-hard users first).
  Both wired into `run_phase2.py`; `--exp all` now includes them. Tests total 104 (46+31+27).

- **2026-07-10 — CORRECTNESS AUDIT of the completed results (see `CORRECTIONS.md`). ALL PRIOR
  NUMBERS SUPERSEDED.** Nothing crashed; the defects were in what the code computed.
  Headline defects: (1) E3's mean/cov split is *exactly* affine-invariant, so the raw-vs-z-scored
  contrast measured nothing — its numbers were `+1e-3*I` ridge artifacts; (2) every kNN accuracy
  leaked across 50%-overlapping windows of the same trial (synthetic: 1.000 vs a true 0.306);
  (3) `h_divergence` leaked the same way and SATURATED near its max of 2 — Phase-1's
  "H-div uniformly high (0.72-0.98)" was an artifact, not a finding; (4) the meta headline
  (`knn_loo` vs `loso_acc`, rho=0.93) was accuracy-predicts-accuracy, circular AND leaked;
  (5) `best_predictor` was post-hoc best-of-4 over 56 uncorrected tests, and "9/14 significant"
  included senic at the WRONG SIGN; (6) `combined_linear_r2` was in-sample; (7) E6 had no
  anti-alias filter and decimated 200 Hz sets to 44-50 Hz; (8) entropy ran on 25-50 sample
  windows (`ENT["min_samples"]` was defined and never used); (9) actionability's ~0.1-0.9 pp
  "win" sits under a ~1 pp ORACLE ceiling — it is a null result; (10) the transfer matrix
  z-scored each dataset before the MMD, deleting what it measured; (11) k=14 datasets are
  **9 cohorts**; (12) senic's confound verdict was never supported by its own probe.
  Fixed + pinned by `tests/test_corrections.py`. An adversarial review then found 3 more,
  two in the new code: E3's null floor was split by ROW not TRIAL (~14x too small),
  `h_divergence` was left unfixed, and my `_subsample_by_group` collapsed `knn_loso` to 1-2
  subjects. All fixed. Tests: 49+39+35+50 = 173 green.
  **SURVIVES:** A4 (inter-subject 2.4-3.6x inter-day), classifier-agnostic difficulty
  (LDA/SVM/RF agree, rho 0.47-0.94), ninapro_db1 r=-0.77 / db2 r=-0.73, low intrinsic dim.
  **OPEN:** the predictor works only where LOSO accuracy is near the floor (npdb1 12%, npdb2 14%)
  and fails where it is healthy (grabmyo 0.70 -> r=+0.03) — floor effect or real? Must resolve.
  **NEXT:** `invalidate_stale.py` -> `run_profile.py --module 12345` -> `run_phase2.py --exp all`
  -> `validate_results.py`.

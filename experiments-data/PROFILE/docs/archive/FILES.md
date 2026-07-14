# PROFILE (Paper 2 — data-science track) — FILE INDEX

Everything for the data-science paper lives under
**`experiments-data/PROFILE/`**. This is the map. (Generated 2026-07-10.)

Reading order for a newcomer: **`STATE.md`** → `CORRECTIONS.md` → `NEXT_STEPS.md` → `ROADMAP.md`.

---

## 1. Planning & narrative docs (`PROFILE/*.md`)

| File | Status | What it is |
|---|---|---|
| `STATE.md` | ✅ **AUTHORITATIVE** | Current state, results, fixes, next steps. Start here. |
| `CORRECTIONS.md` | ✅ current | The audit — every defect found + fixed, in order |
| `NEXT_STEPS.md` | ✅ current | Decision tree: results → write-up |
| `ROADMAP.md` | ✅ current (hook 3 struck) | Plan + 4 novelty pillars; has a 2026-07-10 warning banner |
| `EXTERNAL_RESEARCH_REVIEW.md` | ✅ current | Independent external review (cross-verified) + §14 detailed experiment program w/ novelty, code-fix specs, ground-truth checks; flags the floor-effect "RESOLVED" overclaim |
| `FILES.md` | ✅ this file | The index |
| `PLAN.md` | 📎 historical | First pre-code plan (5 modules) |
| `PHASE2_PLAN.md` | 📎 historical + status log | Phase-2 suite (E1–E10) + append-only log |
| `POST_RESULTS_PLAN.md` | ⚠️ partly superseded | Old post-run plan; see NEXT_STEPS instead |
| `RESULTS_REVIEW.md` | ⚠️ **SUPERSEDED — do not quote** | Phase-1 review; pre-audit numbers |

## 2. Code — `dsprofile/` package

- **Infra:** `config.py`, `windows.py`, `cv.py`, `stats.py`, `progress.py`,
  `features_extra.py`, `datacards.py`, `viz.py`, `figures.py`, `__init__.py`
- **Phase-1 modules:** `module1_signal.py`, `module2_separability.py`, `module3_shift.py`,
  `module4_channels.py`, `module5_difficulty.py`
- **Phase-2 blocks:** `block_a.py`, `block_b.py`, `block_c.py`, `block_d.py`, `calibration.py`,
  `robust_difficulty.py`, `actionability.py`, `faabos.py`, `senic_probe.py`, `sdi.py`,
  `meta.py`, `transfer.py`

## 3. Runners & standalone scripts (`PROFILE/*.py`)

| File | Purpose |
|---|---|
| `run_profile.py` | CLI for modules 1–5 |
| `run_phase2.py` | CLI for blocks + aggregates (`--exp all` / `--exp figs`) |
| `validate_results.py` | The gate — must exit 0 before write-up |
| `invalidate_stale.py` | Deletes exactly the stale outputs before a re-run |
| `floor_effect.py` | Floor-effect experiment — OLD/contested (single per-dataset trend; superseded by X1) |
| `floor_effect_x1.py` | Experiment **X1** — floor-effect DONE CORRECTLY (3 probes: matched class count / matched accuracy / dataset-clustered pooled model + synthetic ground truth). **BUILT — run on Ubuntu; `--selftest` first.** Settles the contested "RESOLVED" claim. |
| `exp_D_predictor_ranking.py` | Experiment D — which cheap statistic is best (DONE) |
| `exp_E_permutation.py` | Experiment E — permutation test (DONE) |
| `exp_A_window_length.py` | Experiment A — window-length robustness 100/250/500 ms ✅ DONE — complete, in `results/experiments/` |
| `exp_B_recalibration.py` | Experiment B — per-subject mean-recalibration → accuracy ✅ DONE — complete, in `results/experiments/` |
| `exp_C_cross_session.py` | Experiment C — cross-session difficulty prediction ✅ DONE — complete, in `results/experiments/` |
| `exp_common.py` | shared infra for A/B/C: atomic writes, resume/skip, shard-safe `--collect` |

Tests for A/B/C: `tests/test_addon_experiments.py` (41 ground-truth checks).

## 3b. Paper-2 experiment suite — PROVEN (buries the review's limitations)

Implements the `EXTERNAL_RESEARCH_REVIEW.md` §14 program. Every module ships a synthetic
**ground-truth** selftest (known-answer control); the real runs are trusted only if it is green.
- `paper_experiments/` — hardened package: `common.py` (math/stats/distances/transforms/classifiers/
  IO/synthetic-frames) + one module per experiment: `code_fixes` (F1–F4/F-dec), `x2_decoupling`,
  `x3_dl_target`, `x4_recalibration_coral`, `x5_deamplitude`, `x6_learned_repr`, `x7_mmd_sensitivity`,
  `x8_ood_baselines`, `x9_transfer`, `x10_senic`, `x11_meta_regression`, `x12_stability`,
  `x13_imbalance`, `x14_adaptive_lda`, `x15_conformal_difficulty` (**novel: coverage-guaranteed
  difficulty intervals**), and `selftest.py`.
  - Run all ground truth: `python -m paper_experiments.selftest` — **validated 53/53 + X1 PASS**.
- `floor_effect_x1.py` (X1, floor-effect done correctly; cohort-clustered) + `tests/test_floor_effect_x1.py`.
- `notebooks/` — 16 parallel-runnable notebooks (`00_SELFTEST_ALL` first, then `X1 … X15`) +
  `README.md` (parallel-run guide) + `_build_notebooks.py` (generator). Each reads the 250 ms cache
  read-only and writes `results/<tag>/` atomically → safe to launch all at once.
- `requirements.txt` — pinned deps (X15 reproducibility).

## 4. Tests (`tests/`) — 214 checks

`test_math.py` (49) · `test_phase2.py` (39) · `test_blocks.py` (35) · `test_corrections.py` (50) ·
`test_addon_experiments.py` (41) · `test_floor_effect_x1.py` (X1 ground-truth — BUILT, not yet run)

## 5. Results (`results/`)

- **Phase-1:** `module1/` (42) `module2/` `module3/` (28) `module4/` (28) `module5/` (28)
- **Phase-2 blocks (14 each):** `block_a/ block_b/ block_c/ block_d/ calibration/
  robust_difficulty/ actionability/ faabos/ senic_probe/`
- **Aggregates:** `module6_sdi/` (2) `meta/` (2) `transfer/` (2)
- **New experiments (all 5 add-ons COMPLETE):** `floor_effect/` (5 files, 4 datasets) ·
  `experiments/` — A/B/C (each: `summary` + 14 per-dataset) **+ D + E JSONs** (D/E were missing and
  were re-run + persisted 2026-07-11)
- **Cross-dataset table:** `results/cross_dataset_matrix.xlsx` — everything, one row per dataset
- **Cache (box-only):** `results/_feature_cache/` — the large fast/complex parquets; NOT synced here

## 6. Literature

- `paper-summaries/` — 14 structured summaries (`01`–`14`) + `INDEX.md`; the evidence base
- `data-science-related-papers/` — the source PDFs

## 7. Related files OUTSIDE PROFILE

- `../../STATE.md` — root repo state (the Paper-2 row points here)
- `../../.claude/sessions/SESSIONS.md` — session log with the audit entries
- `../../experiments-dl/BENCH-LOSO/` — sibling DL paper (Paper 1); linked only via
  Module 5's optional `load_paper1_loso` (swaps the difficulty target from LDA proxy to DL LOSO)

## 8. Datasets (read-only inputs, not part of PROFILE)

`../../data/L1/<dataset>/` — `signals.h5` + `manifest.parquet` + `STATE.md`, for the 14 datasets:
emaha_db1/db4/db5/db7, fors_emg, grabmyo (+2 flow variants), myobit,
ninapro_db1/db2/db4/db5, senic.

---

## 9. Cross-repo dependency footprint (checked 2026-07-10)

PROFILE is self-contained EXCEPT for a thin dependency on the shared L1 library:

| Needs | For |
|---|---|
| `../../semg/data/window_index.py` → `build_window_index()` | leakage-safe windowing (identical to Tracks 1/2) |
| `../../semg/splits/splitter.py` → `Normalizer(mode="global")` | train-only z-score |
| `../../data/L1/<dataset>/` | the 14 datasets (h5 + manifest) |

That is the ENTIRE external footprint (verified by grepping `dsprofile/` imports). To relocate
PROFILE you would need those two `semg` files + `data/L1/`. Nothing else.

## 10. Sibling tracks — related but NOT part of this paper

- **Track-1 (classical ML paper)** — repo ROOT: `Track1_Run.ipynb`, `EMAHA_Replication_Run.ipynb`,
  `Manual_Features_Run.ipynb`, `run.py`, `emaha_replication_results.xlsx`,
  `results/Track1_Results_Report.xlsx`. Uses `semg/features/` + `semg/train/track1.py`.
- **Track-2 (DL paper = "Paper 1")** — `../../experiments-dl/`: `BENCH-LOSO/` (active),
  `SEN-020/`, `AC-048/`. Linked to PROFILE only via Module 5's `load_paper1_loso`.
- **`../../semg/`** — the shared L1 library under all three tracks.
- **`../../scripts/`** — dataset-build utilities (`validate_l1.py`, `smoke_test.py`, …).

## 11. Repo governance & the BROAD literature review (context, not this paper)

- `../../CLAUDE.md`, `../../STATE.md`, `../../.claude/USER-REQUIREMENTS.md`,
  `../../.claude/user-background.md`, `../../.claude/sessions/SESSIONS.md`
- `../../.claude/literature-review/` + `../../literature-review-papers/` — the broad 35-column
  review feeding ALL tracks. Distinct from PROFILE's own `paper-summaries/` (the focused evidence
  base for THIS paper only).

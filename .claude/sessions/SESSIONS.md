# Session History (append-only — never edit previous entries)

<!-- Add new entries at the bottom. Format:
## Session N — YYYY-MM-DD
**Focus:** What was worked on
**Done:** What was accomplished
**Decisions:** Any decisions made (with rationale)
**Next:** What the next session should do
-->

## Session 1 — 2026-04-12
**Focus:** Bulk classification of the 19-venue IEEE literature snapshot in `literature-review-papers/conferences-and-journals-complete-list/` into a richer, multi-label taxonomy tied to the two research tracks and Q1–Q10.

**Done:**
- Authored `literature-review-papers/CLASSIFICATION-TASK.md` — single source of truth for sub-agents: researcher background, two tracks, Q1–Q10 link, 30-label `Primary_Category` vocabulary, `Secondary_Tags` multi-label rule, 4-tier `Relevance_Tier`, keyword dictionaries for every label (§4), output Excel schema (§7, 16 columns), and a reusable Python classifier skeleton (§10).
- Dispatched 8 parallel sub-agents (A–H), each given full inline context + folder assignment, covering all 19 folders with ~balanced workload.
- Every agent wrote its own reproducible classifier Python script in `literature-review-papers/` (`classify_1_nsre.py`, `classify_7_embc.py`, `classify_3_sensors.py`, `classify_9_access.py`, `classify_E_tbme_im.py`, `classify_F_bmi_icorr_iros.py`, `classify_G_icra_biocas_memea_hms.py`, `classify_H_small_venues.py`).
- Produced 19 `classified_papers.xlsx` files — one per venue folder — all with identical 16-column schema.
- Verified all 19 outputs exist; aggregated totals: **1,855 papers → 1,046 HIGH / 192 MEDIUM / 492 LOW / 125 SKIP**. Top primaries: FEATURE_EXTRACTION 249, REHAB_CLINICAL 193, DATASET 147, UNRELATED 125, OTHER_EMG 123, ML_CLASSICAL 114, DL_CNN 93, PREPROCESSING_FILTERING 70, DL_RNN 69, DL_HYBRID 65, DL_GNN_GAN_OTHER 65, HD_EMG 62, DL_TRANSFORMER 58, REVIEW_SURVEY 34, CROSS_SUBJECT_LOSO 16 (primary), TRANSFER_LEARNING 13, CROSS_DATASET 1.

**Decisions:**
- Multi-label taxonomy with priority ladder (UNRELATED → REVIEW → DATASET → DL_HYBRID > DL_TRANSFORMER > DL_GNN_GAN_OTHER > DL_RNN > DL_CNN → FOUNDATION → ML_CLASSICAL → FEATURE/PREPROC/AUG/DIMRED → LOSO/TRANSFER/CROSS → apps → fallback) rather than hand-classification. Rationale: reproducible, auditable, re-runnable when taxonomy changes; at ~1,855 papers hand-classification by LLM would be inconsistent and costly.
- Keyword-hit-based classifier written in Python with `Keywords_Hints` column preserving the literal substrings that triggered each label, so researcher can audit every single classification.
- HD-EMG safeguard: primary HD_EMG is capped at LOW unless LOSO or TRANSFER_LEARNING secondaries also fire → MEDIUM. Matches user's "HD-EMG is not much useful unless it generalizes" instruction.
- Source files left untouched; prior `Category` / `Relevance_Reason` columns from an earlier pass in NSRE/TBME/Sensors xlsx were ignored and a richer taxonomy written from scratch.

**Known false-positive modes (flagged for manual review):**
- `DATASET` bucket over-triggered by bare `ninapro` token — many method-papers that merely benchmark on NinaPro get DATASET primary. Pronounced in IEEE Sensors (31) and I&M (16). Fix: tighten §4.7 to require phrases like `we present|we release|publicly released`.
- `DATA_AUGMENTATION` catches the word "augmentation" in "prosthesis augmentation" contexts (robotic assistance, not data aug). ~5 rows in NSRE.
- `5_BMI` folder had non-EMG papers leak through the IEEE Xplore pre-filter (e.g. `EMGANet` is a breast-ultrasound net). Classifier reflects the abstract truthfully; the venue itself wasn't clean.
- `12_IEEE_Instrumentation_Measurement` CSV has mojibake (`Human�Machine`) — preserved verbatim.

**Next:**
- User should open a HIGH-tier view in any venue and spot-check the keyword hits vs. the `Justification` column to decide whether to tighten the regexes.
- Tighten `DATASET` detection (§4.7 in CLASSIFICATION-TASK.md) and re-run the 8 `classify_*.py` scripts.
- Filter `Secondary_Tags` (not Primary_Category) for CROSS_SUBJECT_LOSO / TRANSFER_LEARNING / CROSS_DATASET across all 19 venues to build the Q6/Q7/Q10 evidence pool.
- Use `Primary_Category = REVIEW_SURVEY` (34 papers) as initial deep-review reading list.
- Begin Phase 2 deep reviews on the highest-yield venues: NSRE (206 HIGH), EMBC (181), IEEE Access (119), I&M (86), Sensors (136), BMI (71), MeMeA (25, mostly classical-ML-friendly), HMS (25).
- When Phase 2 starts, seed `papers.yaml` with P### IDs and write `reviews/PXXX.md` files per the template in `literature-review-papers/CLAUDE.md`.

## Session 2 — 2026-04-14
**Focus:** Consolidate the 19 per-venue `classified_papers.xlsx` outputs into a single master Excel that adds (a) abstract-derived engineering summaries, (b) re-classification into the researcher's custom category ordering, and (c) per-category must-read lists ranked by descending publication year.

**Done:**
- Authored `literature-review-papers/consolidate_summary.py` — single-file pipeline that loads all 19 venue files (1,855 rows), builds an extractive 3-sentence abstract summary per paper (problem-cue + method-cue + result-cue heuristics) plus a per-category engineering-problem template, re-buckets every paper into 16 custom categories in the researcher's requested ordering, and scores each row for must-read priority (tier × year × venue × ADL/EMAHA/LOSO bonus, with HD-EMG penalty).
- Tightened the dataset-release detector after first pass: original "publicly available" pattern over-fired on papers that merely USED a public dataset; switched to a "we/this paper + release-verb (collect/record/present/release/...) + dataset/database/corpus/benchmark within 80 chars" window match. Result: NEW_DATASETS bucket dropped from 47 (noisy) → 4 (too strict) → 30 (well-calibrated; includes EMAHA-DB1 P0123, MovePort P0703, SeNic P0526, AVE Speech P0017).
- Wrote `literature-review-papers/MASTER_SUMMARY.xlsx` — 21 sheets: `0_Index`, `All_Papers` (1,855 rows × 21 cols including PaperID, Custom_Category, Engineering_Problem, Abstract_Summary, MustRead_Score), `Category_Counts`, `Venue_Counts`, 16 `MR_<CATEGORY>` sheets (top-80 each, year-desc + score-desc sort), and `All_MustRead_Top200`.
- Wrote `literature-review-papers/MUST_READ_INDEX.md` — top-10 markdown index per category for quick browsing.

**Custom-category histogram (1,855 papers):**
OTHER 739 / FEATURE_EXTRACTION 365 / PREPROCESSING 129 / TRANSFORMERS 115 / DEEP_LEARNING 107 / TRANSFER_LEARNING 91 / ML_CLASSICAL 61 / DATA_AUGMENTATION 47 / CROSS_SUBJECT_LOSO 46 / REVIEW_SURVEY 43 / FOUNDATION_LLM 35 / NEW_DATASETS 30 / OTHER_ARCH 28 / MAMBA_SSM 11 / DISTILLATION 6 / CROSS_DATASET 2.

**Decisions:**
- Used a deterministic Python summarizer rather than per-paper LLM summarisation. Rationale: 1,855 papers makes per-paper LLM cost-prohibitive and slow; the extractive "problem + method + result" sentence picker plus a category-derived engineering-problem template gives consistent, auditable summaries the researcher can grep across.
- Each paper appears in exactly one Custom_Category (first match wins, in researcher's stated priority order). Multi-tag membership is still recoverable from `Secondary_Tags` (preserved verbatim in `All_Papers`). Rationale: per-category must-read sheets need a single home for each paper, otherwise the same paper would dominate multiple sheets.
- Kept the upstream `Primary_Category` column unchanged so the upstream classifier output remains independently auditable; added `Custom_Category` as a separate column.
- Capped each MR_<CATEGORY> sheet at 80 rows. Rationale: the user asked for "must-reads", not full per-category dumps — and the full corpus is in `All_Papers` for any deeper filter.
- Ordering within each MR sheet: `Year_Int desc, MustRead_Score desc, Relevance_Tier asc` — newest year wins ties, then highest priority signal, then HIGH-tier preferred.

**Known limitations of this consolidation:**
- The `OTHER` bucket (739 papers) is everything outside the researcher's custom rank — mostly REHAB_CLINICAL (186), OTHER_EMG (117), UNRELATED (76), PROSTHETIC_CONTROL (64), EXOSKELETON_HMI (60), HD_EMG (59), FORCE_TORQUE_ESTIMATION (45), HARDWARE_ELECTRODE (35), MUSCLE_SYNERGY_DECOMP (35), FATIGUE (25), DIMENSIONALITY_REDUCTION (16). These are intentionally not surfaced as must-reads under the researcher's current ordering; they remain available in `All_Papers`.
- Extractive summaries inherit any abstract-quality issues (e.g. mojibake `Human�Machine` from venue 12 CSV). The original `Abstract` column is preserved verbatim for verification.
- Category re-bucketing is regex-driven; some borderline papers (e.g. methodology papers that mention "publicly available" datasets they used) may sit in adjacent categories. The audit trail is `Primary_Category + Secondary_Tags + Justification` columns.
- Top-of-category MR sheets contain a few false positives (e.g. P1705 wind-energy survey landed in MR_REVIEW_SURVEY because the upstream classifier flagged the title word "Survey"). Researcher should treat the MR sheets as a sorted candidate pool, not a vetted reading list.

**Next:**
- Researcher to skim `MR_NEW_DATASETS` first (30 papers) — confirm whether EMAHA-DB1, MovePort, SeNic, AVE Speech are the right anchors for the dataset catalog (`datasets/CATALOG.md`).
- Read `MR_REVIEW_SURVEY` (43 papers) to anchor the broader literature view, as planned in Phase 2.
- For Track 1 (classical ML): use `MR_FEATURE_EXTRACTION` + `MR_ML_CLASSICAL` + `MR_PREPROCESSING` as the candidate pool.
- For Track 2 (DL): use `MR_DEEP_LEARNING` + `MR_TRANSFORMERS` + `MR_MAMBA_SSM` + `MR_OTHER_ARCH` + `MR_FOUNDATION_LLM`.
- For cross-cutting goals (LOSO / transfer / cross-dataset): use `MR_CROSS_SUBJECT_LOSO` + `MR_TRANSFER_LEARNING` + `MR_CROSS_DATASET` + `MR_DISTILLATION` + `MR_DATA_AUGMENTATION`.
- Re-run `consolidate_summary.py` after any upstream classifier change — script is idempotent and pulls fresh from the 19 venue xlsx files.

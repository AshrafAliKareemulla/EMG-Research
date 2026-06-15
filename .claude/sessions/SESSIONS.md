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

## Session 3 — 2026-05-28
**Focus:** Deep literature review of all 57 score-3 abstract-rated papers in the IEEE EMBC folder (7_EMBC).

**Done:**
- Confirmed 57 papers with Q9=3 from EMBC_abstract_screening.xlsx (341 total; score breakdown: 5→21, 4→54, 3→57, 2→65, 1→127, 0→17).
- All 57 papers had valid MinerU markdown files — no PDF fallbacks needed.
- Set up `7_EMBC/_shared/` folder; copied and configured `build_deep_review_excel.py` (PREFIX=EMBC, OUT_FILENAME=EMBC_deep_review.xlsx).
- Ran 3 waves of 2 parallel background agents each (6 agents total):
  - Wave 1A: EMBC-005 to EMBC-070 (10 papers) ✅
  - Wave 1B: EMBC-077 to EMBC-127 (10 papers) ✅
  - Wave 2A: EMBC-128 to EMBC-160 (10 papers) ✅
  - Wave 2B: EMBC-168 to EMBC-208 (10 papers) ✅
  - Wave 3A: EMBC-211 to EMBC-287 (10 papers) ✅
  - Wave 3B: EMBC-302 to EMBC-336 (7 papers) ✅
- All 57 JSONs passed full structural validation (39 keys, no blanks, no extras).
- Built EMBC_deep_review.xlsx at the folder root (57 rows, 38 columns).

**Key inconsistencies flagged across waves:**
- EMBC-211: Abstract states rank-5 accuracy=99.6%; Section III states 96.2%.
- EMBC-234: r² boxplot values differ from text medians (OCR approximation).
- EMBC-151: Abstract has impossible std dev > mean for a correlation coefficient (paper typo).
- EMBC-331: Table I shows 7 subject rows but paper states N=8 subjects collected.
- EMBC-107: Internally contradictory sentence mixing aquatic vs on-land accuracy values.

**High-value follow-ups identified:**
- EMBC-029: Online TAC Test framework; offline-online accuracy gap directly relevant to Track-1 methodology.
- EMBC-042: Public code at https://github.com/adwi592/GPT-EMG-Analyser/ — LLM-automated feature extraction.
- EMBC-135: HYSER public dataset; MU spatial channel selection applicable to EMAHA.
- EMBC-153: emg2vec SSL pretraining framework — highly applicable to Track-2 ADL pipeline.
- EMBC-177: Public dataset+code on GitHub; paper cites EMAHA-DB1 directly.
- EMBC-192: LocoD public dataset + code; probability-rejection post-processing reusable for Track-1 LDA.
- EMBC-250: 3D CNN for 14-DoF HD-sEMG convolutive decoding.
- EMBC-287: LSTM on public NinaPro datasets.
- EMBC-331: PyTorch code at https://github.com/mufengjun260/MCSHRI; 7 ML + 2 DL baseline comparison.

**Notable findings for our ADL research:**
- EMBC-302: Electrode ID permutation catastrophic (drops to ~20% accuracy) — need ID verification in EMAHA sessions.
- EMBC-336: LOGD (Logarithm Detector) is top-tier time-domain feature alongside WFL/MAV/RMS — underused in literature, should add to Track-1 feature set.
- EMBC-127: LOSO regression drops 33pp vs within-subject — strong quantitative cross-subject degradation evidence.
- EMBC-202: Forearm EMG degrades less across days than wrist EMG — validates EMAHA forearm placement.

**Next:**
- Deep review of score-4 and score-5 EMBC papers when requested (54 + 21 = 75 papers).
- Consider building a cross-venue comparison Excel once other folders are also reviewed.

---

## Session 3 — 2026-05-29 — EMBC score-4/5 deep review

**Goal:** Deep-review all abstract-score 4 and 5 papers in `7_EMBC` (score-3 already done); cross-check the existing 3★ work first.

**Done:**
- Cross-checked the 57 pre-existing score-3 JSONs: all structurally valid (39 keys, no empties), avg ~23K content chars. Clean.
- Deep-reviewed **20 score-5** + **54 score-4** = 74 papers via parallel background subagents (per §6 recipe; smaller 4-paper batches mid-run due to recurring socket/API drops + one account session-limit pause). Every JSON validated structurally + content-spot-checked.
- **EMBC-050** (ar=11252918) SKIPPED — no mineru markdown AND no PDF; logged in `_shared/doubts.txt` (unrecoverable per §6.7).
- Full-folder sweep: **131 JSONs, 0 structural problems** (s5 20/20, s4 54/54, s3 57/57).
- Rebuilt `EMBC_deep_review.xlsx` (133 rows × 38 cols, banner "131 papers (abstract score >= 3)").

**Quality:** new JSONs run 27–44K content chars; col_18 ~1.7–2.4K, col_20 ~1.0–1.5K. Content spot-checks (EMBC-237, 337) verified numbers trace to source tables.

**Notable inconsistencies flagged (gold per protocol):**
- EMBC-239 ViT-HGR: abstract pairs 84.62% with 78,210 params, but Table II shows that accuracy = 340,866-param model.
- EMBC-045: abstract 0.85/0.87 accuracy unsupported — Table I shows 0.49–0.65.
- EMBC-228: fabricated "online accuracy 99.3%" row (Discussion admits no online test); leaky split.
- EMBC-304: "inter-subject" claim but CV is by-trial within subject; circular GGS segmentation ground truth.
- EMBC-266 / EMBC-298 / EMBC-330: optimization-on-eval-metric leakage / window-overlap leakage / train-on-validation typo.

**High-value for our ADL work:** EMBC-101 (new public EMAHA-DB7 + pace/position robustness benchmark), EMBC-108 (cross-subject variance-transfer for LOSO calibration), EMBC-096 (cross-subject disentanglement on Hyser), EMBC-145 (genuine LOSO over 27 subjects), EMBC-261/278 (Track-1 feature/classifier benchmarks), EMBC-260 (mixup augmentation), EMBC-008 (public MyoBM-Net code).

**Next:**
- Move to the next venue's deep review when requested (NSRE/Sensors/Access/I&M/BMI remain).

---

## Session — 2026-05-29 — Folder 20 (Dataset_IEEE_Command_Search) deep review COMPLETE

**Mode:** User explicitly required main-thread, one-paper-at-a-time review (NO subagents), full depth, quality over speed.

**Done:** Deep-reviewed all 18 abstract-score-3/4/5 papers in `20_Dataset_IEEE_Command_Search` (prefix DAT). Set up `_shared/` (copied JSON template + build script), wrote DAT-001/003/004/005/006/007/009/010/011/012/013/014/015/016/017/020/021/022 JSONs (39 keys each, all structurally validated, 14.5–26.4K content chars; scaled to paper substance). Built `DatasetSearch_deep_review.xlsx` (18 rows × 38 cols, sorted 5s→4s→3s).

**Worklist (score order):** s5 = DAT-001,004,006,007,010,012,014,015,017 ; s4 = DAT-003,005,011,013,016,020,022 ; s3 = DAT-009,021. (Excluded s1/s2: DAT-002,008,018,019.)

**Highest-value for our ADL research:**
- **DAT-015 = EMAHA-DB1 — OUR PRIMARY DATASET.** Baselines to beat: cubic-SVM (SVM3) + F5 → 75.39% (22-class) / + F2 → 83.21% (FAABOS), within-subject; EMGHandNet (CNN+BiLSTM) UNDERperforms SVM3 (73.34%). **Central open problem: LOSO collapses to 58.59% (FAABOS) — the inter-subject gap to attack.**
- Public robustness benchmarks usable alongside EMAHA: **DAT-014 SeNic** (8-ch, 36 subj, 5 non-ideal factors, quantified electrode-shift angles), **DAT-016 MyoBit** (16-ch semi-dense, 9 non-ideal factors, +IMU), **DAT-017** (fully open wrist-sEMG + toolbox + SD/CD/CS benchmark + domain adaptation).
- Method donors: DAT-007 (Scheme&Englehart feature robustness: at ≤6 ch add WAMP to Hudgins TD), DAT-006 (window/overlap/kernel: 75% overlap + kernel-7; window length matters more at low channel count), DAT-022 (≥5 reps/position for CNN position-invariance; unbalanced data HURTS LDA), DAT-005 (~1/3 data / ≈2 reps fine-tuning suffices), DAT-020 (electrode-geometry augmentations: all-opposite + channel-switch help; mirror/malfunction hurt), DAT-013 (DCGAN aug + DTW/FFT-MSE realism), DAT-011 (MIA detect-then-denoise; synthetic-MIA generator), DAT-010 (COZDAL frequency-split + CBAM), DAT-021 (MovePort multimodal EMG+IMU+MoCap+IPS).

**Inconsistencies flagged (gold per protocol):**
- DAT-010 COZDAL: calls pooled-all-subjects-in-train-and-test "subject-independent" (it is the opposite) → headline 95.3/98.8% are within-subject upper bounds.
- DAT-009: per-sample classification + random 50/50 SAMPLE split → severe leakage; ~98% accuracy is an artifact (cautionary example).
- DAT-004: Ninapro gesture count 19 vs 17; latency 3.3 vs 2.45 ms; 2D-CNN size < 1D-CNN despite 2× params.
- DAT-012: 5 vs 6 output classes; 26-dim vs 4×8=32 features; implausible 96.52% on 52-class DB1.
- DAT-006: figure overlap-label reversal; delay tables mislabeled "seconds" (are ms); duplicated DB3 rows.
- DAT-007: ALL result figures garbled in MinerU (impossible negative errors) — relied on prose findings; flagged PDF fallback as route to exact numbers.

**Extraction notes:** MinerU text/tables clean across all 18; recurring issue = garbled figure/heatmap/scatter extractions (DAT-006/007/014/016/017/021) — used prose + clean tables, ignored garbled figures, all flagged in col_35. No PDF fallback needed (text sufficed for the findings).

**Memory:** Corrected `dataset_senic_structure.md` — "Angle xlsx" = electrode-SHIFT angles (not joint kinematics; SeNic has none); p0–p10 are 11 shift POSITIONS (not postures); h30–h35 = fatigue-enhanced cohort.

**Next:** next venue deep review when requested (NSRE/Sensors/Access/I&M/BMI remain).

---

## Session — Dataset Infrastructure (L1 pipeline) — 2026-06-15, ~23:45

Built L1 pipeline + ingested ninapro_db1, ninapro_db2, emaha_db1 (canonical `signals.h5`+`manifest.parquet`); DB3 empty. Reusable loader/splitter in `semg/`. Fixed normalization leakage (`Normalizer` modes, `global` default) + added adapter versioning/device metadata. Smoke test ALL PASSED on GPU. Details: design doc `semg-datasets/semg-dataset-setup.md` (§15) + `data/L1/<ds>/STATE.md`. Next: Track 1 (reproduce EMAHA SVM / LOSO 58.59%) or Track 2 (1D-CNN LOSO).

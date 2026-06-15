# STATE.md

**Phase:** 1.5 — Literature Review (consolidation + must-read ranking complete)
**Updated:** 2026-05-29
**Session count:** 3

## Area Status (one line each — details in area status files)

| Area | Status | Detail file |
|------|--------|-------------|
| Literature review | Phase 2 in progress. EMBC deep review COMPLETE (131 JSONs). **Folder 20 (Dataset_IEEE_Command_Search) deep review COMPLETE 2026-05-29: all 18 score-3/4/5 papers (9×s5 + 7×s4 + 2×s3) reviewed by main thread one-by-one — DAT-001..022 JSONs in _shared/ (all 39-key validated, 14.5–26.4K chars) + DatasetSearch_deep_review.xlsx built (18 rows × 38 cols). Key papers: DAT-015=EMAHA-DB1 (our primary dataset; SVM3+F5 75.39%/FAABOS 83.21%, LOSO 58.59% = central gap), DAT-014=SeNic + DAT-016=MyoBit + DAT-017 (public robustness benchmarks).** Next venue pending. | `literature-review-papers/STATUS.md` |
| Datasets | L1 ingest in progress. **NinaPro DB1** (✓ 294 MB, 28,161 trials), **DB2** (✓ 10 GB, 23,640 trials), **EMAHA-DB1** (✓ 2.35 GB, 15,736 trials; primary ADL dataset, 22-class + FAABOS + orig train/test split preserved) converted to canonical L1 via `semg/adapters/`. DB3 empty (amputee set — re-download). Reusable loader/splitter (`semg/data/`, `semg/splits/`) built; **smoke test PASSED locally on EMAHA** (loader+LOSO+normalizer). See per-dataset `data/L1/<name>/STATE.md`; design in `semg-datasets/semg-dataset-setup.md`. | `datasets/CATALOG.md` |
| ML experiments | Not started | `experiments/ml-classical/STATUS.md` |
| DL experiments | Not started | `experiments/dl-advanced/STATUS.md` |
| LOSO | Not started | `experiments/loso/STATUS.md` |
| Cross-domain | Not started | `experiments/cross-domain/STATUS.md` |

## Next Steps

1. Open `literature-review-papers/MASTER_SUMMARY.xlsx` — start with `0_Index` sheet, then `MR_NEW_DATASETS` (30 papers) to seed `datasets/CATALOG.md`.
2. Read the `MR_REVIEW_SURVEY` sheet (43 papers) to anchor the broader literature view.
3. Track 1 reading pool: `MR_FEATURE_EXTRACTION` + `MR_ML_CLASSICAL` + `MR_PREPROCESSING` + `MR_DATA_AUGMENTATION`.
4. Track 2 reading pool: `MR_DEEP_LEARNING` + `MR_TRANSFORMERS` + `MR_MAMBA_SSM` + `MR_OTHER_ARCH` + `MR_FOUNDATION_LLM`.
5. Cross-cutting (Q6/Q7/Q10) reading pool: `MR_CROSS_SUBJECT_LOSO` + `MR_TRANSFER_LEARNING` + `MR_CROSS_DATASET` + `MR_DISTILLATION`.
6. (Optional) Tighten upstream §4.7 / §4.5 regexes in `literature-review-papers/CLASSIFICATION-TASK.md` and re-run the 8 `classify_*.py` scripts; then re-run `literature-review-papers/consolidate_summary.py` to refresh `MASTER_SUMMARY.xlsx`.
7. Begin Phase 2 deep reviews on the highest-yield venues (NSRE 206 HIGH, EMBC 181, Sensors 136, Access 119, I&M 86, BMI 71).

## Blockers

- None (pre-filtered paper corpus is now fully classified and ready for Phase 2 deep review).

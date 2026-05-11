# STATE.md

**Phase:** 1.5 — Literature Review (consolidation + must-read ranking complete)
**Updated:** 2026-04-14
**Session count:** 2

## Area Status (one line each — details in area status files)

| Area | Status | Detail file |
|------|--------|-------------|
| Literature review | Phase 1.5 done: 1,855 papers consolidated → MASTER_SUMMARY.xlsx with engineering summaries + 16 must-read sheets | `literature-review-papers/STATUS.md` |
| Datasets | Not started | `datasets/CATALOG.md` |
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

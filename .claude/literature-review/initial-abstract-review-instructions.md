# Initial Abstract Review — Operating Instructions

> **Scope.** Phase 1 abstract screening of an IEEE Xplore CSV export for one conference/journal folder, ending with a single colour-coded Excel sheet ranked by must-read score. This is **not** the deep paper review (that lives in `.claude/literature-review/literature-review-questions.md`).
>
> **Status (2026-05-12).** Done for folders **9 → 19 + 20** (see existing `*_abstract_screening.xlsx` files and `1_combined_abstract_screening.xlsx`). **Pending for folders 1, 2, 3, 4, 5, 6, 7, 8** — see §6 for folder-specific quirks.

---

## 1. Purpose of this phase

For every paper in a venue's IEEE Xplore CSV, decide — from the abstract alone — whether the paper is worth a full read for this project's sEMG / ADL research goals. The output is one row per paper with nine answers (Q1–Q9, definitions in §4) and a 0–5 must-read score that drives Phase 2.

Researcher context to keep in mind while screening (don't restate it in cells):

- BTech-Honors researcher, sEMG signal classification for ADL.
- **Track 1:** classical ML — all preprocessing × all feature extraction × all classifiers (incl. PCA / SFS / tsfresh).
- **Track 2:** DL — CNN, BiLSTM, Transformers, hybrids, transfer learning, distillation, wavelet/Fourier+DL, LLM fine-tuning, raw and feature-based.
- Datasets in use: **EMAHA (primary), NinaPro, BioPatRec**. Hardware budget: single **24 GB A5000** GPU.
- Cross-cutting goals: LOSO, cross-dataset, ADL-Net (one universal model), ML-vs-DL decision rule.
- Full goal list is in `.claude/USER-REQUIREMENTS.md`; the 10 engineering questions are in `literature-review-papers/engineering-questions.md`. Read those once before starting; do **not** quote them in every cell.

---

## 2. Pipeline at a glance

```
IEEE Xplore CSV  ──prep_screening.py──►  abstract_screening/batch_inputs/batch_N.json
                                                       │
                                          (LLM sub-agent per paper, see §4)
                                                       ▼
                                         abstract_screening/agent_returns/<paper_id>.json
                                                       │
                                       ──build_screening_excel.py──►
                                                       ▼
                                <Venue>_abstract_screening.xlsx   (one sheet, sorted, colour-coded)
```

Two helper scripts already live in `literature-review-papers/conferences-and-journals-complete-list/_shared/`:

| Script | Role |
|---|---|
| `prep_screening.py <folder> <prefix> <batch_size>` | Read the venue's CSV, assign Paper IDs (`<prefix>-001`, `<prefix>-002`, …), and split into JSON batches under `<folder>/abstract_screening/batch_inputs/`. Also creates `agent_returns/` empty. |
| `build_screening_excel.py <folder> <out_xlsx_name>` | Load every `<paper_id>.json` from `agent_returns/`, sort, colour-fill by score, and write the final Excel. |

Both scripts are idempotent — re-running `prep_screening.py` only overwrites batch files; re-running `build_screening_excel.py` rebuilds the Excel from whatever JSONs are present (partial runs are fine).

Sub-agents are the moving part in the middle: one agent per paper produces one strict-schema JSON file (§4).

---

## 3. Folder layout (during and after screening)

```
<N>_<Venue>/
├── export<date>.csv                          # IEEE Xplore export — the source of truth
├── papers/                                   # PDFs (not needed for abstract screening)
├── abstract_screening/                       # ◄ created by prep_screening.py
│   ├── batch_inputs/
│   │   ├── batch_1.json                      # array of {paper_id, title, year, authors, doi, abstract}
│   │   ├── batch_2.json
│   │   └── ...
│   └── agent_returns/
│       ├── <prefix>-001.json                 # one per paper, schema in §4
│       ├── <prefix>-002.json
│       └── ...
└── <Venue>_abstract_screening.xlsx           # ◄ final deliverable, written by build_screening_excel.py
```

The `abstract_screening/` directory is **transient** — once the Excel is written and reviewed, the intermediate JSONs can be cleaned up (folders 9–19 already had this done; only the XLSX remains).

---

## 4. The nine screening questions (Q1–Q9)

Each sub-agent must return **exactly one JSON object per paper** with these 12 keys (sourced verbatim from `build_screening_excel.py`):

| Key | Type | Meaning |
|---|---|---|
| `paper_id` | string | The ID assigned by `prep_screening.py` (e.g. `NSRE-042`). |
| `title` | string | Copy verbatim from the input batch. |
| `year` | string | Copy verbatim. |
| `q1_emg_surface` | `"Yes"`/`"No"`/`"Unclear"` | Is the paper about EMG, and specifically **surface** EMG? (Intramuscular / needle EMG → No.) |
| `q2_introduces_dataset` | `"Yes"`/`"No"`/`"Unclear"` | Does this paper **introduce a new sEMG dataset**? Merely benchmarking on NinaPro/EMAHA/etc. → No. Require an explicit "we present / we release / publicly released" cue. |
| `q3_scientific_objective` | string (1–2 sentences) | What is the author trying to prove scientifically? Cross-subject? Transfer learning? New feature representation? New architecture? Write the conclusion you draw from the abstract, in plain language. |
| `q4_paper_type_and_accuracy` | string | Is it a **survey / accuracy improvement / new method / new dataset / clinical study**? If an accuracy number appears in the abstract, record it (e.g. "Accuracy paper; 92.4% on X dataset"). |
| `q5_related_to_our_research` | `"Yes"`/`"No"`/`"Partial"` | Is this useful for our sEMG ADL classification work (Track 1 or Track 2 or any cross-cutting goal)? If `No`, fill `q6` with a one-line skip reason and stop reasoning further. |
| `q6_what_we_can_infer` | string (1–3 sentences) | If `q5 ≠ No`: list the concrete things we can borrow — preprocessing tricks, feature sets, model designs, training tricks, evaluation protocols, ablations. If `q5 = No`: `"N/A — paper not related"`. |
| `q7_mixed_modality` | string | What modalities does the paper use? `"sEMG-only"`, `"sEMG + IMU"`, `"sEMG + EEG"`, `"sEMG + force/kinematics"`, `"ECG + EMG hybrid"`, etc. Flag mixed modalities because they tend to be less directly useful. |
| `q8_skip_reason` | string or `null` | If the paper should be skipped, a short reason here (e.g. "Pure ECG paper; no sEMG content"). Otherwise `null`. |
| `q9_score` | integer 0–5 | Must-read score: `5 = must read`, `4 = strong yes`, `3 = read selectively`, `2 = skim only`, `1 = barely relevant`, `0 = completely unrelated, skip`. See §5 for the rubric. |

### Hard rules for the screening agent

1. **Use the abstract only.** Do not open the PDF. Do not fetch external context. If the abstract is missing or empty, set `q1` = `"Unclear"`, `q9` = `0`, and explain in `q8_skip_reason`.
2. **One JSON file per paper.** Filename = `<paper_id>.json` in `agent_returns/`. No wrapping `[...]` array, no markdown code fences — but `build_screening_excel.py` tolerates triple-backtick wrapping if it slips in.
3. **Use straight ASCII quotes.** Avoid smart quotes, em-dashes that get mojibake'd, and non-UTF8 characters. Keep one paper's JSON under 32k characters per field (Excel cell cap; the script truncates anything longer).
4. **Be terse but specific.** `q3`, `q6`, and `q8` should each be 1–3 sentences. Don't restate the abstract.
5. **Don't invent.** If the abstract is silent on accuracy, say so in `q4` (`"Accuracy not reported in abstract"`). If modality is unclear, say `"sEMG-only (assumed; not explicitly stated)"`.
6. **Skip ≠ Score 0.** A paper with `q5 = No` may still score 1–2 if there's any tangential relevance (e.g. a robotics paper that uses EMG as one of many control signals). Score 0 is reserved for *completely* off-topic papers that wandered into the venue's EMG keyword search (e.g. "EMGANet" for breast-ultrasound — a known leak in folder 5_BMI).

### Example — score 5 (must-read)

```json
{
  "paper_id": "AC-015",
  "title": "Temporal Forecasting of Sit-to-Stand Motion for Assistive Device Control Based on Muscle Synergy Prediction",
  "year": "2026",
  "q1_emg_surface": "Yes",
  "q2_introduces_dataset": "No",
  "q3_scientific_objective": "Forecast sit-to-stand muscle synergy with a multilayer LSTM DNN on sEMG to anticipate motion intent for assistive chair control.",
  "q4_paper_type_and_accuracy": "Accuracy / forecasting paper; 87.97% average accuracy at 300 ms forecast horizon.",
  "q5_related_to_our_research": "Yes",
  "q6_what_we_can_infer": "Multi-resolution LSTM for sEMG temporal forecasting, muscle-synergy decomposition as input feature, anticipation-time vs accuracy trade-off — directly relevant to Track 2 BiLSTM/Transformer designs and ADL motion-intent decoding.",
  "q7_mixed_modality": "sEMG-only",
  "q8_skip_reason": null,
  "q9_score": 5
}
```

### Example — score 0 (skip)

```json
{
  "paper_id": "AC-012",
  "title": "DynaMI: Dynamic Membership Inference via Adaptive Manifold Perturbations",
  "year": "2026",
  "q1_emg_surface": "No",
  "q2_introduces_dataset": "No",
  "q3_scientific_objective": "Propose a privacy-attack framework for ML membership inference using adaptive manifold projection.",
  "q4_paper_type_and_accuracy": "Generic ML privacy/security framework; no biosignal content.",
  "q5_related_to_our_research": "No",
  "q6_what_we_can_infer": "N/A — paper not related",
  "q7_mixed_modality": "No biosignals at all — generic ML privacy paper",
  "q8_skip_reason": "Pure ML privacy / membership-inference paper unrelated to sEMG or biosignals.",
  "q9_score": 0
}
```

---

## 5. The 0–5 must-read rubric

Anchor each score on the dominant signal in the abstract, not on a feeling:

| Score | When to give it |
|---|---|
| **5** | Directly on-topic for ADL classification, LOSO, transfer learning, new sEMG dataset, or a model architecture we plan to try (CNN/BiLSTM/Transformer/hybrid/Mamba). Recent (≥ 2023) is a tiebreaker but not required. |
| **4** | Strongly relevant — sEMG gesture/ADL classification with a feature, preprocessing, or training trick we could borrow, even if dataset is non-ADL (NinaPro/CapgMyo). |
| **3** | Related sEMG work that is *adjacent* — e.g. force/torque estimation, prosthetic control, HD-EMG decomposition — useful as background but not core. |
| **2** | sEMG paper but in a context we don't pursue (single-channel custom hardware, clinical-only, gait/EEG fusion). Skim if time permits. |
| **1** | Tangential — paper mentions EMG only in passing, or studies a single non-ADL muscle activity in isolation. |
| **0** | Off-topic (non-EMG, pure ECG/EEG, ML methods paper that mentions "EMG" as a keyword without engaging with it). Document the reason in `q8_skip_reason`. |

For consistency, prefer the lower score when in doubt — it's easier to promote a 3 to a 4 during Phase 2 than to demote a falsely-confident 5.

---

## 6. End-to-end procedure for the 8 pending folders

Pending folders, source CSV, and suggested ID prefix:

| # | Folder | Source CSV | Rows | Suggested prefix | Suggested output filename |
|---|---|---|---:|---|---|
| 1 | `1_IEEE_NSRE` | `export2026.05.11-15.03.24.csv` | 447 | `NSRE` | `NSRE_abstract_screening.xlsx` |
| 2 | `2_IEEE_BE` | `export2026.05.11-15.01.24.csv` | 420 | `BE`   | `BE_abstract_screening.xlsx` |
| 3 | `3_IEEE_Sensors` | `export2026.05.11-14.58.36.csv` | 306 | `SEN`  | `IEEE_Sensors_abstract_screening.xlsx` |
| 4 | `4_ICASSP` | `2021_2026_export2026.04.11-16.49.43.csv` | 28 | `ICASSP` | `ICASSP_abstract_screening.xlsx` |
| 5 | `5_BMI` | `2021_2026_export2026.04.11-16.56.05.csv` | 88 | `BMI`    | `BMI_abstract_screening.xlsx` |
| 6 | `6_MLSP` | `export2026.04.11-16.59.35.csv` | 3 | `MLSP` | `MLSP_abstract_screening.xlsx` |
| 7 | `7_EMBC` | `export2026.04.11-17.04.46.csv` | 341 | `EMBC` | `EMBC_abstract_screening.xlsx` |
| 8 | `8_IJCNN` | `export2026.04.11-17.07.34.csv` | 14 | `IJCNN` | `IJCNN_abstract_screening.xlsx` |

### Folder-specific quirks (read before running)

- **Folder 1 — NSRE.** This CSV was **re-pulled on 2026-05-11** with a broader filter; it now contains 348 NSRE-journal papers plus 84 rows from rehab-robotics conferences (ICORR / WRRC / i-CREATe). Rows from ICORR were already screened in folder 16 (`ICORR_abstract_screening.xlsx`). Decide before running: re-screen everything in this CSV with NSRE prefixes (simplest, ~84 duplicates of work), or pre-filter the CSV to drop rows whose `Publication Title` contains `ICORR` / `WRRC` / `i-CREATe` (cleaner, but requires a one-off filter step).
- **Folder 2 — `2_IEEE_BE`.** Despite the name, this is **not** TBME-only. The May-2026 re-pull spans 60+ biomedical-engineering venues — IEEE TBME (118), JBHI (89), BioRob, BioCAS, IECBES, BMEiCON, ICABME, etc. (420 rows total). The older 2026-04 screening (`STATUS.md`) covered only 107 TBME rows. Confirm with the user whether the goal is to screen the whole 420-row pull or just the TBME subset.
- **Folder 3 — IEEE Sensors.** Also a May-2026 re-pull (306 rows). 217 from IEEE Sensors Journal, 29 from Sensors Letters, rest from sister sensors conferences. Older screening pass had 190 rows.
- **Folder 4 — ICASSP.** Only 28 rows — one batch is enough.
- **Folder 5 — BMI.** Known to have non-EMG papers leaking through the IEEE Xplore EMG filter (e.g. `EMGANet` is a breast-ultrasound network). Expect a higher-than-usual share of `q9_score = 0`; record the reason in `q8_skip_reason` so the user can audit.
- **Folder 6 — MLSP.** Only 3 rows. Run as a single batch.
- **Folder 7 — EMBC.** Largest pending folder (341 rows). Split into ~14–17 batches of ~20–25 papers each so each agent stays comfortably within its context budget.
- **Folder 8 — IJCNN.** 14 rows — one batch.

### Step-by-step

1. **Prep.** From any working directory:
   ```powershell
   cd "E:\sEMG Research Enhanced\literature-review-papers\conferences-and-journals-complete-list\_shared"
   python prep_screening.py <folder_name> <prefix> <batch_size>
   ```
   Use a batch size that keeps each agent under ~25 papers (the existing screening used batches of 13–25). Example for NSRE: `python prep_screening.py 1_IEEE_NSRE NSRE 22` → ~21 batches.

2. **Screen.** For each batch, dispatch one sub-agent with these inputs:
   - The full batch JSON (paper_id, title, year, authors, doi, abstract).
   - This file (`initial-abstract-review-instructions.md`) as the prompt.
   - Explicit instruction to write one JSON file per paper into `<folder>/abstract_screening/agent_returns/<paper_id>.json`.

   Agents may be dispatched in parallel — they don't share state, and `build_screening_excel.py` happily picks up whichever files exist. The existing approach was 8 parallel sub-agents covering all 19 folders in one go (see `.claude/sessions/SESSIONS.md` Session 1).

3. **Build the Excel.**
   ```powershell
   cd "E:\sEMG Research Enhanced\literature-review-papers\conferences-and-journals-complete-list\_shared"
   python build_screening_excel.py <folder_name> <out_xlsx_name>
   ```
   The script:
   - Loads every JSON from `agent_returns/`.
   - Sorts: `score desc → year desc → paper_id asc`.
   - Colour-fills each row by score: **5 = green** (`C6EFCE`), **4 = light blue** (`DDEBF7`), **3 = light yellow** (`FFF2CC`), **2 = light orange** (`FCE4D6`), **1 = light pink** (`F4CCCC`), **0 = grey** (`C9C9C9`).
   - Writes 12 columns: `Paper ID | Title | Year | Q1: surface EMG? | Q2: new dataset? | Q3: scientific objective | Q4: type + accuracy | Q5: related? | Q6: what we can infer | Q7: modality mix | Q8: skip reason | Q9: score 0-5`.
   - Freezes the header row, sets row height 80, wraps long cells, and prints a score histogram.

4. **Sanity-check.** Open the XLSX, scan the score histogram printed by the script, spot-check a handful of rows in each score bucket. Tune any obvious agent misfires (e.g. by hand-editing a JSON and re-running step 3).

5. **Update state when done.** Per `CLAUDE.md` rules:
   - Append a session entry to `.claude/sessions/SESSIONS.md` (Focus / Done / Decisions / Next).
   - Update `literature-review-papers/STATUS.md` with the new per-venue counts.
   - Update `STATE.md` if the Phase number / next-step list shifts.
   - When all 8 folders are done, regenerate the cross-venue rollup `1_combined_abstract_screening.xlsx` to include the new venues (the existing combined file already covers 9–19 + 20; add 1–8 to it).

---

## 7. After all 8 folders are screened — cross-venue rollup

Once each folder has its own `<Venue>_abstract_screening.xlsx`, regenerate the consolidated view used for cross-venue triage:

- `1_combined_abstract_screening.xlsx` already has the schema we need — sheets: `All_Combined`, `Score_Counts`, `Venue_x_Score`, and one `Score_<N>` sheet per score bucket (0–5). Each `All_Combined` row carries `Source_Folder`, `Venue`, `Paper ID`, `Title`, `Year`, `Q1…Q9`.
- The natural extension is a small script under `_shared/` (e.g. `build_combined.py`) that walks every `<N>_<Venue>/<Venue>_abstract_screening.xlsx`, concatenates the rows with the right `Source_Folder` / `Venue` tag, and rewrites the combined workbook. If such a script doesn't exist yet, write it during the rollup step and commit it next to the other `_shared/` scripts.

This consolidated workbook is what feeds Phase 2 (deep review) and the must-read ranking in `MASTER_SUMMARY.xlsx`.

---

## 8. Boundaries — what this phase is NOT

- **Not a deep review.** Don't open PDFs, don't extract methods, don't fill the 32-column deep-review template. That belongs to `.claude/literature-review/literature-review-questions.md` and is only triggered for `q9_score ≥ 4` papers (and selectively for 3s).
- **Not a regex-based bulk-classification pass.** That separate pipeline (`literature-review-papers/classify_*.py` + `CLASSIFICATION-TASK.md` + `consolidate_summary.py`) produced `MASTER_SUMMARY.xlsx`. It runs against the same CSVs but assigns multi-label taxonomy categories rather than the 9-question screening output. The two outputs are complementary and live side-by-side; **do not conflate them**.
- **Not a relevance verdict that ends the paper.** Score 0 / 1 papers stay in the workbook (greyed/pink). They are recorded so the audit trail is complete, not deleted.

---

## 9. Open questions for the user (resolve before kicking off)

These are real ambiguities the data itself doesn't answer — flag them up front rather than guessing:

1. **Folder 1 (NSRE) — drop ICORR/WRRC/i-CREATe rows?** The May-2026 NSRE CSV pulled in 84 conference rows including ICORR papers that were already screened in folder 16. Options:
   - (a) Screen everything in the CSV with `NSRE-###` prefixes (simplest, accepts ~84 duplicates).
   - (b) Filter the CSV first so the screening covers only NSRE-journal rows (348 papers), and treat ICORR/WRRC/i-CREATe as already covered.
   - (c) Split: re-emit ICORR rows as additional papers into folder 16's screening sheet.

2. **Folder 2 (`2_IEEE_BE`) — scope?** The May-2026 CSV (420 rows) spans 60+ biomedical-engineering venues, not just TBME. Old screening (Session 1) covered only 107 TBME rows. Confirm whether we screen the whole 420 (with prefix `BE`) or restrict to TBME + JBHI + one or two flagship sister journals.

3. **Folder 3 (Sensors) — scope?** Same shape as Folder 2: 306 rows including IEEE Sensors Letters and several BSN/SENSORS conferences. Same call needed.

4. **Folder 5 (BMI) — manual pre-filter for venue leaks?** Around 5–10 rows are known non-EMG (medical-imaging networks named `*EMG*Net`). Should the agent flag these via `q9 = 0` only, or should they be removed from the CSV before screening?

5. **Naming convention for `<prefix>`.** Suggested values are in §6's table; confirm if any clashes with existing prefixes (current ones: `AC, IE, HMS, IM, CYB, ICRA, IROS, ICORR, BIO, SMC, MEM, DS` for folders 9–20). The suggested set (`NSRE, BE, SEN, ICASSP, BMI, MLSP, EMBC, IJCNN`) does not collide.

6. **Output filename.** Folder 3 already uses just `IEEE_Sensors_abstract_screening.xlsx`; folders 4–8 should follow the venue-friendly pattern (`ICASSP_abstract_screening.xlsx`, etc.) — confirm.

7. **Run sub-agents serially or in parallel, and how many at once?** Session 1 dispatched 8 parallel sub-agents for the full 19 folders. For the 8 pending folders the natural granularity is 8 parallel sub-agents, one per folder; very large folders (NSRE 447, BE 420, EMBC 341) may benefit from further per-batch parallelism. Confirm preferred concurrency.

---

## 10. Quick-reference cheat sheet

```text
1. cd _shared
2. python prep_screening.py <folder> <prefix> <batch_size>      # makes batch_inputs/*.json + agent_returns/
3. (sub-agents)   one JSON file per paper into agent_returns/   # schema: 12 keys, §4
4. python build_screening_excel.py <folder> <Venue>_abstract_screening.xlsx
5. open XLSX, eyeball histogram, spot-check rows
6. update STATUS.md, STATE.md, SESSIONS.md
7. after all 8 folders done → regenerate 1_combined_abstract_screening.xlsx
```

Anything outside the green path above (PDF reads, full-text extraction, deep-review fills, cross-paper synthesis) is **out of scope** for this phase.

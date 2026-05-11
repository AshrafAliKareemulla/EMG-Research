# Initial Abstract Review — Operating Instructions

> **Scope.** Phase 1 abstract screening of an IEEE Xplore CSV export for one conference/journal folder, ending with a single colour-coded Excel sheet ranked by must-read score. This is **not** the deep paper review (that lives in `.claude/literature-review/literature-review-questions.md`).

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

Two helper scripts live in `literature-review-papers/conferences-and-journals-complete-list/_shared/`:

| Script | Role |
|---|---|
| `prep_screening.py <folder> <prefix> <batch_size>` | Read the venue's CSV (it greps `export*.csv` inside `<folder>`), assign Paper IDs (`<prefix>-001`, `<prefix>-002`, …), and split into JSON batches under `<folder>/abstract_screening/batch_inputs/`. Also creates an empty `agent_returns/`. |
| `build_screening_excel.py <folder> <out_xlsx_name>` | Load every `<paper_id>.json` from `agent_returns/`, sort, colour-fill by score, and write the final Excel. |

Both scripts are idempotent — re-running `prep_screening.py` overwrites batch files; re-running `build_screening_excel.py` rebuilds the Excel from whatever JSONs are present (partial runs are fine).

Sub-agents are the moving part in the middle: one agent per paper produces one strict-schema JSON file (§4).

---

## 3. Folder layout (during and after screening)

```
<N>_<Venue>/
├── export<date>.csv                          # IEEE Xplore export — the source of truth
├── papers/                                   # PDFs (not needed for abstract screening)
├── abstract_screening/                       # ◄ created by prep_screening.py
│   ├── batch_inputs/
│   │   ├── batch_1.json                      # array of {paper_id, title, year, authors, doi, abstract, pdf_link}
│   │   ├── batch_2.json
│   │   └── ...
│   ├── pdf_links.json                        # paper_id -> pdf_link, side-channel for the builder
│   └── agent_returns/
│       ├── <prefix>-001.json                 # one per paper, schema in §4
│       ├── <prefix>-002.json
│       └── ...
└── <Venue>_abstract_screening.xlsx           # ◄ final deliverable, written by build_screening_excel.py
```

The `abstract_screening/` directory is **transient** — once the Excel is written and reviewed, the intermediate JSONs can be cleaned up.

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
6. **Skip ≠ Score 0.** A paper with `q5 = No` may still score 1–2 if there's any tangential relevance (e.g. a robotics paper that uses EMG as one of many control signals). Score 0 is reserved for *completely* off-topic papers that wandered into the venue's EMG keyword search.

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

## 6. End-to-end procedure

1. **Prep batches.** From any working directory:
   ```powershell
   cd "E:\sEMG Research Enhanced\literature-review-papers\conferences-and-journals-complete-list\_shared"
   python prep_screening.py <folder_name> <prefix> <batch_size>
   ```
   Pick a `<prefix>` that doesn't collide with existing ones (current set: `AC, IE, HMS, IM, CYB, ICRA, IROS, ICORR, BIO, SMC, MEM, DS`). Pick a `<batch_size>` that keeps each agent under ~25 papers (the existing screening used batches of 13–25). For a folder with 341 rows, e.g. `python prep_screening.py 7_EMBC EMBC 22` → ~16 batches.

2. **Screen.** For each batch, dispatch one sub-agent with these inputs:
   - The full batch JSON (paper_id, title, year, authors, doi, abstract).
   - This file (`initial-abstract-review-instructions.md`) as the prompt.
   - Explicit instruction to write one JSON file per paper into `<folder>/abstract_screening/agent_returns/<paper_id>.json`.

   Agents may be dispatched in parallel — they don't share state, and `build_screening_excel.py` happily picks up whichever files exist.

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

   Use the existing filename convention: `<Venue>_abstract_screening.xlsx` written at the folder root.

4. **Sanity-check.** Open the XLSX, scan the score histogram printed by the script, spot-check a handful of rows in each score bucket. Tune any obvious agent misfires (e.g. by hand-editing a JSON and re-running step 3).

5. **Update state when done.** Per `CLAUDE.md` rules:
   - Append a session entry to `.claude/sessions/SESSIONS.md` (Focus / Done / Decisions / Next).
   - Update `literature-review-papers/STATUS.md` with the new per-venue counts.
   - Update `STATE.md` if the Phase number / next-step list shifts.

---

## 7. After all folders are screened — cross-venue rollup

Once each folder has its own `<Venue>_abstract_screening.xlsx`, regenerate the consolidated view used for cross-venue triage:

- `1_combined_abstract_screening.xlsx` already has the schema we need — sheets: `All_Combined`, `Score_Counts`, `Venue_x_Score`, and one `Score_<N>` sheet per score bucket (0–5). Each `All_Combined` row carries `Source_Folder`, `Venue`, `Paper ID`, `Title`, `Year`, `Q1…Q9`.
- The natural extension is a small script under `_shared/` (e.g. `build_combined.py`) that walks every `<N>_<Venue>/<Venue>_abstract_screening.xlsx`, concatenates the rows with the right `Source_Folder` / `Venue` tag, and rewrites the combined workbook. If such a script doesn't exist yet, write it during the rollup step and commit it next to the other `_shared/` scripts.

This consolidated workbook feeds Phase 2 (deep review) and the must-read ranking in `MASTER_SUMMARY.xlsx`.

---

## 8. Boundaries — what this phase is NOT

- **Not a deep review.** Don't open PDFs, don't extract methods, don't fill the 32-column deep-review template. That belongs to `.claude/literature-review/literature-review-questions.md` and is only triggered for `q9_score ≥ 4` papers (and selectively for 3s).
- **Not a regex-based bulk-classification pass.** That separate pipeline (`literature-review-papers/classify_*.py` + `CLASSIFICATION-TASK.md` + `consolidate_summary.py`) produced `MASTER_SUMMARY.xlsx`. It runs against the same CSVs but assigns multi-label taxonomy categories rather than the 9-question screening output. The two outputs are complementary and live side-by-side; **do not conflate them**.
- **Not a relevance verdict that ends the paper.** Score 0 / 1 papers stay in the workbook (greyed/pink). They are recorded so the audit trail is complete, not deleted.

---

## 9. Quick-reference cheat sheet

```text
1. cd _shared
2. python prep_screening.py <folder> <prefix> <batch_size>      # makes batch_inputs/*.json + agent_returns/
3. (sub-agents)   one JSON file per paper into agent_returns/   # schema: 12 keys, §4
4. python build_screening_excel.py <folder> <Venue>_abstract_screening.xlsx
5. open XLSX, eyeball histogram, spot-check rows
6. update STATUS.md, STATE.md, SESSIONS.md
```

Anything outside the green path above (PDF reads, full-text extraction, deep-review fills, cross-paper synthesis) is **out of scope** for this phase.

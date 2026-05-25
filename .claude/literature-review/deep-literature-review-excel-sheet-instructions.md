# Deep Literature Review — JSON & Excel Sheet Instructions

**Purpose.** This file is the operating manual for the *output side* of the sEMG deep
literature review: which papers get reviewed, how each paper's review is stored as a
JSON file, and how all the JSON files are assembled into one formatted Excel workbook
per conference/journal folder.

It is the companion to `.claude/literature-review/deep-literature-review-instructions.md`
— that file tells you **what to read in a paper and what to write in each of the 35
columns**; *this* file tells you **the screening rule, the JSON file format, the folder
layout, and the exact Excel design**. Read both. Do not deviate from either.

### Ready-made files (do not re-derive these)

Two ready-to-use files live in the **central shared scripts folder**
`literature-review-papers/conferences-and-journals-complete-list/_shared/` — the single
top-level `_shared/` directory that sits next to the venue folders and also holds the
Phase-1 screening scripts (`prep_screening.py`, `build_screening_excel.py`, …). This is
**not** a per-venue `_shared/`. **Copy them from there — do not reconstruct the structure
from scratch.**

| File (in `conferences-and-journals-complete-list/_shared/`) | What it is | How to use |
|------|-----------|-----------|
| `deep-review-json-template.json` | The exact 39-key JSON skeleton (4 metadata keys + `col_01`…`col_35`), in the correct order, with a placeholder describing what each value should contain. | Copy it to the venue folder's `_shared/<PREFIX>-NNN.json` for each paper and fill every value. |
| `build_deep_review_excel.reference.py` | The canonical Excel builder with all colours, groups, freeze panes, fonts, widths and layout already coded. | Copy it into the venue folder's `_shared/` as `build_deep_review_excel.py`, edit only the 3 marked config lines, run it. |

So a new session should **not** spend time on JSON structure or Excel styling — those
are fixed and provided. Spend the time on the literature review itself. This markdown
explains *why* each piece is shaped the way it is; the two files above *are* the shapes.

> **Two different `_shared/` folders — do not confuse them.** (1) The **central**
> `conferences-and-journals-complete-list/_shared/` holds the *common, cross-venue*
> scripts and templates (the two files above + the Phase-1 screening scripts). (2) Each
> **venue** folder has its own `<NN>_IEEE_<Venue>/_shared/` holding *that venue's*
> per-paper JSON files and its copy of the build script. Common reusable files live only
> in the central `_shared/`; per-venue outputs live only in the venue `_shared/`.

---

## 0. The pipeline at a glance

```
Abstract screening (.xlsx, Q9 score)
        |
        |  keep papers scored 3, 4, 5
        v
MinerU markdown extraction  (mineru_output/ieee_<n>.md)   <-- primary source
        |  (fall back to papers/ieee_<n>.pdf only if the markdown is corrupted)
        v
Deep review per paper  ->  one 35-column JSON file  (_shared/<PREFIX>-NNN.json)
        |
        |  after every paper in the folder is reviewed
        v
build_deep_review_excel.py  ->  one formatted .xlsx at the folder root
```

One conference/journal folder is processed start-to-finish before moving to the next.

---

## 1. Folder layout (per conference/journal)

Every venue folder under
`literature-review-papers/conferences-and-journals-complete-list/` (e.g.
`12_IEEE_Instrumentation_Measurement`) has this structure:

```
<NN>_IEEE_<Venue>/
├── papers/                         # the PDF papers,  named  ieee_<arnumber>.pdf
├── mineru_output/                  # MinerU markdown,  named  ieee_<arnumber>.md
├── _shared/                        # ALL JSON files + the build script live here
│   ├── <PREFIX>-001.json
│   ├── <PREFIX>-002.json
│   ├── ...
│   └── build_deep_review_excel.py
├── <Venue>_abstract_screening.xlsx # the abstract-screening sheet (input)
├── export<date>.csv                # the venue's paper export (titles, PDF links)
└── <Venue>_deep_review.xlsx        # FINAL output, written at the folder root
```

**Strict rule about `_shared/`.** All JSON files and the build script go *inside*
`_shared/`. The folder **root** contains only: `papers/`, `mineru_output/`,
`_shared/`, the abstract-screening `.xlsx`, the export `.csv`, and the single final
`<Venue>_deep_review.xlsx`. Nothing else. Delete any stale/old deep-review Excel files
before writing the new one.

### Paper-ID prefix per folder

Each folder uses a short prefix for its paper IDs:

| Folder | Prefix | Example ID |
|--------|--------|-----------|
| 10_IEEE_Industrial_Electronics | `IE` | `IE-003` |
| 11_IEEE_Human_Machine_Systems | `HMS` | `HMS-014` |
| 12_IEEE_Instrumentation_Measurement | `IM` | `IM-076` |

The prefix is followed by a zero-padded 3-digit number: `<PREFIX>-NNN`.

### File-naming convention

- PDF papers: `ieee_<arnumber>.pdf`  (e.g. `ieee_10135136.pdf`)
- MinerU markdown: `ieee_<arnumber>.md`  (same `<arnumber>` as the PDF)
- The numeric part is the IEEE `arnumber` (article number from the IEEE Xplore link).
- The `<PREFIX>-NNN` paper ID is a *separate* sequential index; map it to an
  `arnumber` via the export `.csv` (CSV row order → `<PREFIX>-NNN`).
- Occasionally a markdown file is named after the paper's title instead of
  `ieee_<arnumber>.md` — confirm against the `mineru_file` you record.

---

## 2. Step 1 — Abstract screening (which papers to deep-review)

Abstract screening is the **first pass** and is already done. Its result lives in the
`<Venue>_abstract_screening.xlsx` file at the folder root.

- Each row in the screening sheet is one paper.
- The screening sheet contains a **relevance score in the `Q9` column**
  (column index 11, i.e. the 12th column, 0-based). This score is `1`–`5`.
- The score expresses how relevant the paper is to our research focus
  (sEMG signal processing for gesture/movement/intent recognition, prosthetic
  control, rehabilitation, and especially **Activities of Daily Living (ADL)**
  classification — see `CLAUDE.md` and `USER-REQUIREMENTS.md`).

### Screening score meaning

| Q9 score | Meaning | Deep review? |
|----------|---------|--------------|
| **5** | Most relevant — directly on-topic (ADL / sEMG classification / core method) | **YES** |
| **4** | Relevant — clearly useful to the review | **YES** |
| **3** | Moderately relevant — partially useful / adjacent | **YES** |
| 2 | Marginal — only loosely related | No — skip |
| 1 | Not relevant | No — skip |

### The selection rule

> **Deep-review every paper with an abstract score of 3, 4, or 5.**
> Papers scored 1 or 2 are excluded — do not write JSON files for them.

Process the selected papers **in score order: all the 5s first, then the 4s, then the
3s.** (The Excel build later sorts them the same way.)

To build the worklist for a folder:
1. Open `<Venue>_abstract_screening.xlsx`, read the `Q9` column (index 11).
2. Keep every paper with `Q9 ∈ {3, 4, 5}`.
3. Read the export `.csv` to map each kept paper to its title (CSV column index 0)
   and PDF link (CSV column index 15), and to its `arnumber` (CSV row → `<PREFIX>-NNN`).
4. The PDF link can also be taken from the abstract-screening sheet.

---

## 3. Step 2 — Source material (MinerU first, PDF fallback)

The extraction pipeline is **two-step**:

1. **Primary source — MinerU markdown.** Use the markdown file in `mineru_output/`
   (`ieee_<arnumber>.md`) as the starting point for every paper. MinerU was chosen
   after comparing it against PaddleOCR and opendataloader; it gives the best
   structured output (text, tables, math equations, references). Images are ignored —
   only the title/text around an image is extracted. For this review images are
   assumed not essential.
2. **Fallback — the PDF.** Read the PDF in `papers/` directly **only** when the
   markdown is unusable: not readable / corrupted, wrong or partial extraction, or the
   text flatly contradicts the numbers in the tables throughout the paper. Minor issues
   (numbering, math glyphs, unicode, OCR noise in figure-derived tables) are fine and
   should be ignored.

Whatever you do, record the source quality honestly in **column 35
(`Paper Extraction Quality`)** — note any OCR-garbled tables/figures, and state clearly
if you fell back to the PDF.

**Read the whole paper, twice.** Abstract, introduction, related work, methods, every
experimental subsection, ablations, discussion, limitations, appendices, datasets,
tables, math. Do not fill columns from the abstract alone. When a markdown file is
truncated/paginated, read every page before writing.

---

## 4. Step 3 — The per-paper JSON file

For **each** selected paper, produce **exactly one** JSON file in the folder's
`_shared/` directory, named `<PREFIX>-NNN.json` (e.g. `IM-076.json`).

**One record per paper.** Never split a paper into multiple JSON files — not per
dataset, not per model. Multiple datasets/models are listed as bullets *inside* the
relevant column value.

### 4.1 JSON structure — exact keys

> **Use the ready-made skeleton**
> `literature-review-papers/conferences-and-journals-complete-list/_shared/deep-review-json-template.json`
> — copy it and fill in the values. The structure below is the same thing, shown here
> for reference.

The JSON is a flat object (string keys → string values). It has **39 keys**: 4 metadata
keys followed by the 35 template-column keys, in this order:

```json
{
  "paper_id":            "IM-076",
  "abstract_score":      "5",
  "mineru_file":         "ieee_10135136.md",
  "high_value_followup": "...short flag note, see 4.3...",

  "col_01_paper_title":               "...",
  "col_02_pdf_link":                  "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10135136",
  "col_03_aim_contribution":          "...",
  "col_04_datasets_used":             "...",
  "col_05_subjects_acquisition":      "...",
  "col_06_preprocessing":             "...",
  "col_07_segmentation_windowing":    "...",
  "col_08_handcrafted_features":      "...",
  "col_09_learned_features_encoder":  "...",
  "col_10_feature_selection_dimreduction": "...",
  "col_11_data_sufficiency_augmentation":  "...",
  "col_12_task_type":                 "...",
  "col_13_methodology_training":      "...",
  "col_14_evaluation_protocol":       "...",
  "col_15_novelty_positioning":       "...",
  "col_16_main_results":              "...",
  "col_17_reproducibility":           "...",
  "col_18_limitations_gaps_improvements": "...",
  "col_19_reusable_techniques":       "...",
  "col_20_game_changer_knobs":        "...",
  "col_21_generalization":            "...",
  "col_22_ml_vs_dl":                  "...",
  "col_23_features":                  "...",
  "col_24_architecture_training":     "...",
  "col_25_data_augmentation_synthetic":"...",
  "col_26_ssl_foundation":            "...",
  "col_27_robustness":                "...",
  "col_28_realtime_deployment":       "...",
  "col_29_adl_specific":              "...",
  "col_30_clinical_validity":         "...",
  "col_31_channels_sampling_sensor":  "...",
  "col_32_methodological_honesty":    "...",
  "col_33_multimodal":                "...",
  "col_34_other_scientific_questions":"...",
  "col_35_paper_extraction":          "..."
}
```

### 4.2 The 35 column keys → meaning

The 35 `col_NN_*` keys correspond exactly to the 35-column template in
`deep-literature-review-instructions.md`. Their meaning, in three groups:

**Group 1 — Core descriptive (columns 1–17), factual extraction:**

| Key | Column |
|-----|--------|
| `col_01_paper_title` | Paper Title |
| `col_02_pdf_link` | Paper PDF Link |
| `col_03_aim_contribution` | Aim & Contribution Type |
| `col_04_datasets_used` | Datasets Used |
| `col_05_subjects_acquisition` | Subjects & Acquisition |
| `col_06_preprocessing` | Preprocessing Pipeline |
| `col_07_segmentation_windowing` | Segmentation & Windowing |
| `col_08_handcrafted_features` | Handcrafted / Manual Features |
| `col_09_learned_features_encoder` | Learned Features / Encoder |
| `col_10_feature_selection_dimreduction` | Feature Selection / Dimensionality Reduction |
| `col_11_data_sufficiency_augmentation` | Data Sufficiency & Augmentation |
| `col_12_task_type` | Task Type |
| `col_13_methodology_training` | Methodology & Training |
| `col_14_evaluation_protocol` | Evaluation Protocol |
| `col_15_novelty_positioning` | Novelty Positioning |
| `col_16_main_results` | Main Results |
| `col_17_reproducibility` | Reproducibility |

**Group 2 — Critical analysis (columns 18–20), the agent's own reasoning:**

| Key | Column |
|-----|--------|
| `col_18_limitations_gaps_improvements` | Limitations, Gaps & Proposed Improvements |
| `col_19_reusable_techniques` | Reusable Techniques for Our Research |
| `col_20_game_changer_knobs` | Game-Changer Hyperparameters / Experimental Knobs |

**Group 3 — Scientific-question tracking (columns 21–34) + extraction (column 35):**

| Key | Column |
|-----|--------|
| `col_21_generalization` | Generalization & Subject-Independence |
| `col_22_ml_vs_dl` | ML vs DL Trade-offs |
| `col_23_features` | Features |
| `col_24_architecture_training` | Architecture & Training |
| `col_25_data_augmentation_synthetic` | Data, Augmentation, Synthetic Generation |
| `col_26_ssl_foundation` | Self-Supervised & Foundation Models |
| `col_27_robustness` | Robustness |
| `col_28_realtime_deployment` | Real-Time, Deployment, Online Control |
| `col_29_adl_specific` | ADL-Specific |
| `col_30_clinical_validity` | Clinical Validity |
| `col_31_channels_sampling_sensor` | Channels, Sampling, Sensor Design |
| `col_32_methodological_honesty` | Methodological Honesty |
| `col_33_multimodal` | Multimodal |
| `col_34_other_scientific_questions` | Any Other Scientific Questions |
| `col_35_paper_extraction` | Paper Extraction Quality |

For the **content** of each column — exactly what to capture, the sub-questions,
the `Y/N/Partial` answer format for the scientific-question columns — follow
`deep-literature-review-instructions.md` precisely. Key content rules carried over:

- Every field gets a concrete answer or the literal string `N/A — not discussed`.
  **No blank values, ever.** Do not invent; do not paraphrase the abstract.
- Record exact numbers with units; if a number is missing, write `Not stated`.
- For the scientific-question columns (21–34) use the per-sub-question
  `Q-ID: Y/N/Partial — one-line evidence` style; if a theme is untouched, write
  `Question not addressed.`
- Columns 18–20 are opinionated analysis, not extraction — be specific and actionable.
- Distinguish "authors claim X" from "X is true"; flag inconsistencies with
  `⚠ Inconsistency:`.

### 4.3 The 4 metadata keys

| Key | What to put |
|-----|-------------|
| `paper_id` | The `<PREFIX>-NNN` ID, e.g. `"IM-076"`. |
| `abstract_score` | The Q9 abstract-screening score as a string: `"5"`, `"4"`, or `"3"`. |
| `mineru_file` | The markdown filename actually used, e.g. `"ieee_10135136.md"`. |
| `high_value_followup` | A short note flagging whether this paper is a high-value follow-up — e.g. public code/dataset, a concrete improvement idea, or directly relevant to our ADL work. If the paper has a public repo and an identifiable improvement, raise the `🚩 HIGH-VALUE FOLLOW-UP` flag here. Always fill it (never blank). |

### 4.4 JSON quality rules

- The file **must be valid JSON** (UTF-8, double-quoted keys and string values).
- All 39 keys present, in the order above. **No extra keys**, no missing keys.
- **No empty string values** — every column carries real content or
  `N/A — not discussed` / `Question not addressed.`
- Values are plain strings. Use bullet-style text *inside* a string (e.g.
  `• NinaPro DB1 ...  • CapgMyo ...`) when listing multiple datasets/models.
- Keep direct quotes from the paper short (under ~15 words) and cite section/table.

### 4.5 Mandatory cross-check after every paper

After writing each JSON, **validate it** before moving on. Confirm:

1. It parses as valid JSON.
2. All 4 metadata keys + all 35 `col_01..col_35` keys are present.
3. No extra/stray keys; no empty values.
4. `paper_id` and `abstract_score` are correct for this paper.

A quick batch check over the whole `_shared/` folder:

```python
import json, glob
meta = ['paper_id','abstract_score','mineru_file','high_value_followup']
for f in sorted(glob.glob('<PREFIX>-*.json')):
    d = json.load(open(f, encoding='utf-8'))
    cols = sorted(int(k.split('_')[1]) for k in d if k.startswith('col_'))
    miss = [m for m in meta if m not in d] + [f'col_{i}' for i in range(1,36) if i not in cols]
    extra = [k for k in d if not k.startswith('col_') and k not in meta]
    empty = [k for k in d if isinstance(d[k],str) and not d[k].strip()]
    print(f, 'OK' if not (miss or extra or empty) else f'PROBLEM miss={miss} extra={extra} empty={empty}')
```

---

## 5. Step 4 — Build the Excel workbook

When **every** selected paper in the folder has a JSON file, assemble the workbook.

### 5.1 The build script

> **Use the ready-made script**
> `literature-review-papers/conferences-and-journals-complete-list/_shared/build_deep_review_excel.reference.py`
> — copy it into the venue folder's `_shared/` directory, rename it to
> `build_deep_review_excel.py`, edit only the 3 marked config lines, and run it. Do
> not rewrite it from scratch; the colours/layout below are already coded in it.

`_shared/build_deep_review_excel.py` collects every `<PREFIX>-*.json` in `_shared/`
and writes one formatted `.xlsx` to the **folder root**. It uses only `openpyxl`.
Re-running it is idempotent — it rebuilds from whatever JSON files currently exist.

Per folder, the script is adapted in exactly **three places** (the marked
`>>> PER-FOLDER CONFIG <<<` block in the reference script):

1. The glob pattern: `<PREFIX>-*.json` (e.g. `IM-*.json`).
2. The output filename: `<Venue>_deep_review.xlsx` (e.g.
   `InstrumentationMeasurement_deep_review.xlsx`), written to the folder root.
3. The banner text on row 1 (see 5.4).

Run it from inside `_shared/`:

```
python build_deep_review_excel.py
```

### 5.2 Column schema (38 spreadsheet columns)

The sheet has **38 columns** = 2 metadata columns + the 35 template columns + 1
follow-up flag column. Each is a `(json_key, display_header, group)` tuple. Column
order:

1. `paper_id` → "Paper ID" — group `meta`
2. `abstract_score` → "Abstract\nScore" — group `meta`
3.–37. `col_01_*` … `col_35_*` → "1. Paper Title" … "35. Paper Extraction Quality"
4. `high_value_followup` → "HIGH-VALUE Follow-up Flag" — group `flag`

(`mineru_file` is stored in the JSON but is **not** shown as an Excel column.)

The five header-colour **groups**:

| Group | Columns | Meaning |
|-------|---------|---------|
| `meta` | Paper ID, Abstract Score | metadata |
| `core` | template columns 1–17 | core descriptive |
| `crit` | template columns 18–20 | critical analysis |
| `sciq` | template columns 21–34 | scientific-question tracking |
| `extr` | template column 35 | extraction quality |
| `flag` | HIGH-VALUE Follow-up Flag | follow-up flag |

### 5.3 Visual design (exact specification)

**Header colours** (`PatternFill` solid, white bold text):

| Group | Hex |
|-------|-----|
| `meta` | `1F2A44` (near-navy) |
| `core` | `1F4E78` (deep blue) |
| `crit` | `B45309` (burnt amber) |
| `sciq` | `5B2C87` (deep purple) |
| `extr` | `374151` (slate grey) |
| `flag` | `9A3412` (dark rust) |

**Row 1 — title banner.** Merged across all 38 columns, fill `0F1B2D`, white bold
13 pt text, left-aligned, row height 26.

**Row 2 — header row.** One cell per column, group colour, white bold 10 pt,
centred, wrapped text, thin borders, row height 50.

**Body rows (row 3 onward).** One paper per row, font Calibri 9 pt, colour `20242B`,
left/top aligned, wrapped text, thin borders (`C8CDD6`), row height **340**.

- **Alternating band:** even body rows fill `FFFFFF` (`BAND_A`), odd rows `F4F6F9`
  (`BAND_B`).
- **Score-tinted ID cells:** the `paper_id` and `abstract_score` cells are tinted by
  the abstract score — `5` → green `C6EFCE`, `4` → light blue `DDEBF7`,
  `3` → light yellow `FFF2CC`; bold, centred.
- **Aim column highlight:** column 3 (`col_03_aim_contribution`) is tinted
  `FFF8E1` (`AIM_TINT`) on every body row — it is the anchored, always-visible column.
- **Paper Title** cells (`col_01`) are bold.
- **PDF link** cells (`col_02`) that start with `http` become real hyperlinks,
  blue `1155CC`, underlined.

**Column widths:** Paper ID = 11, Abstract Score = 9, Paper Title = 34,
PDF Link = 30, every other column = 52 (`DEFAULT_WIDTH`).

**Freeze panes = `F3`.** This keeps the banner + header rows (rows 1–2) and the first
five columns — Paper ID, Abstract Score, Paper Title, PDF Link, **and the Aim &
Contribution column** — visible while scrolling right. (The Aim column is the 5th
column, E; freezing at F3 anchors A–E.)

**Auto-filter** is applied over the header row, spanning the full table.
**Gridlines are turned off** (`showGridLines = False`) — the cell borders carry the
structure.

### 5.4 Row ordering and banner

- Records are **sorted by abstract score descending, then paper ID ascending** —
  all 5s, then 4s, then 3s; within a score, IDs in order.
- The row-1 banner reads:
  `sEMG ADL Deep Literature Review  -  IEEE <Venue> (folder <NN>)  -  <N> papers (abstract score >= 3)`
  where `<N>` is the number of JSON files / papers reviewed.

### 5.5 Output

One workbook, sheet titled "Deep Review", saved as
`<Venue>_deep_review.xlsx` at the folder root. Delete any previous deep-review Excel
file in that folder first.

---

## 6. Hard rules (do not violate)

1. **Screen by score.** Deep-review exactly the papers with abstract score 3, 4 or 5.
   Process 5s, then 4s, then 3s.
2. **MinerU first.** Use the `mineru_output/` markdown; fall back to the PDF only when
   the markdown is genuinely unusable, and say so in column 35.
3. **Read the whole paper twice** before writing anything.
4. **One JSON per paper**, in `_shared/`, named `<PREFIX>-NNN.json`, with all 39 keys,
   no blanks, no extra keys, valid JSON.
5. **Cross-check every JSON** right after writing it (section 4.5).
6. **`_shared/` holds all JSONs + the build script**; the folder root holds only the
   inputs and the single final `<Venue>_deep_review.xlsx`. Remove stale Excels.
7. **The Excel design is fixed** — the colours, groups, freeze pane `F3`, score tints,
   row height 340, banner, auto-filter, hidden gridlines are all as specified above.
   Match the existing folders exactly; only the glob prefix, output filename, and
   banner text change between folders.
8. **Do not split a paper** across multiple JSON files; list multiple datasets/models
   as bullets inside a column value.
9. Follow `deep-literature-review-instructions.md` for the content of every column.
```

---

## 7. Working pace, cross-checking, and asking for help (MANDATORY)

This section is operational guidance about *how you work through the papers*, not about
file formats. These rules carry the same weight as the hard rules in Section 6 — they
exist because the literature review is **judgment-intensive and high-stakes**: every
JSON ends up in the final Excel that the user will use for their research write-up. A
fast-but-sloppy review is worse than no review at all.

### 7.1 Take proper time. Do not rush.

- **Time is not a constraint.** Treat each paper as if it is the only paper you will
  review today. There is no schedule pressure, no quota to hit, no token budget to
  protect by truncating analysis. Deliver one careful, fully cross-checked JSON per
  paper before moving to the next.
- **Read each paper twice end-to-end** before writing anything (already in Section 3,
  re-stated here because it is the most-violated rule). The first pass is for
  understanding the paper as a whole — what the authors claim, what they actually did,
  how the results connect to the methodology. The second pass is for capturing exact
  numbers, table values, equation forms, hyperparameters, and statistical-test
  outcomes that go into individual columns.
- **For long or complex papers, read three times.** The third pass is targeted —
  re-read just the methods + results + tables before drafting columns 13–17, and
  re-read just the discussion + limitations before drafting columns 18–20.
- **Re-read targeted sections during writing.** When drafting a specific column, scan
  the paper again for the relevant section instead of working from memory. This is
  not optional — memory is unreliable across 35 columns.
- **No batch / parallel paper review.** Never start reading paper N+1 while still
  writing the JSON for paper N. Finish, cross-check, and confirm one paper before
  picking up the next.

### 7.2 Reason, reiterate, analyse — then write

The three Group-2 analysis columns (18–20) and the 14 Group-3 scientific-question
columns (21–34) are the high-value content of the entire review. For each:

1. **Reason** — given what the paper actually did, what is the honest answer to the
   sub-question? Distinguish "authors claim X" from "evidence shows X".
2. **Reiterate** — read the relevant evidence in the paper again. Confirm the number
   you are about to write matches the table or sentence it came from.
3. **Analyse** — for columns 18–20 especially, ask: which limitation actually
   matters? What specific experiment would address it? What would change a reviewer's
   mind? Vague critiques like "small sample" are not enough — say *what specifically
   should be done about it and why it matters*.

Only after all three steps for a given column should you write the value into the
JSON. Skipping straight from "I read the paper" to "I am writing" is exactly the
failure mode the user has been correcting.

### 7.3 Mandatory cross-check after every paper — full procedure

Section 4.5 lists the validation script. That script catches *structural* errors. It
does **not** catch content errors. Do both of the following after every JSON:

**Step A — structural validation (automatic):** run the Python snippet from Section 4.5.
Confirm: valid JSON, 39 keys present, no extras, no empties. If any check fails, fix
before moving on.

**Step B — content cross-check (manual, per paper):** re-open the source mineru
markdown and verify, value by value:

1. **`paper_id` and `abstract_score`** match the abstract-screening sheet.
2. **`mineru_file`** matches the actual file you read.
3. **`col_01_paper_title`** matches the paper's title exactly.
4. **`col_02_pdf_link`** matches the IEEE Xplore PDF link.
5. **`col_04_datasets_used`** — every subject count, channel count, sampling rate,
   class count is checked against the paper's "Materials" / "Dataset" section.
6. **`col_06_preprocessing`** — every filter cutoff, order, window length, overlap is
   checked against the paper's "Preprocessing" / "Methods" section.
7. **`col_13_methodology_training`** — every hyperparameter (LR, batch, epochs,
   architecture layer sizes) is checked against the paper's "Training" / "Network
   Architecture" section.
8. **`col_14_evaluation_protocol`** — the train/test split, CV scheme, and any
   subject-disjoint vs random claim is checked against the paper's "Evaluation"
   section. This column is where leakage gets missed.
9. **`col_16_main_results`** — every accuracy / RMSE / R² / F1 number is checked
   against the paper's Tables and Result figures. Cross-check both the headline mean
   and the standard deviation. Note any inconsistencies between the paper's prose and
   its tables in `col_35`.
10. **`col_17_reproducibility`** — code/data/weights/seeds claims are checked against
    the paper's actual statements (look for explicit URLs / "publicly available" /
    "code released").
11. **`col_18` to `col_20`** — re-read your own analysis and ask: is each criticism
    grounded in something the paper actually does or does not say? Remove any
    speculation not anchored to the paper.
12. **`col_21` to `col_34`** — for every `Y/N/Partial` answer, confirm the one-line
    evidence quote is grounded in the paper.
13. **`col_35`** — describe the actual extraction quality faithfully. Flag any garbled
    tables, malformed figures, missing equations.

State out loud (in your message) when you have completed both Step A and Step B for a
paper. Only then move to the next paper.

### 7.4 Numbers and quotes — verify before you write

- **Every reported number must trace to a specific table, equation, or sentence in
  the paper.** If you cannot point to where a number comes from, you must not put it
  in the JSON.
- **Cross-check Tables against prose.** Papers often have small inconsistencies (e.g.
  Table II mean stride accuracy 96.7 % vs per-subject mean of ~94 %). When you find
  such an inconsistency, flag it explicitly in `col_35` with `⚠ Inconsistency:` and
  in `col_18` if it materially affects the headline claim.
- **Short quotes (under ~15 words) are encouraged when the wording is itself the
  point.** Use them sparingly, and always with the section/table reference.
- **Distinguish `Authors:` vs `Inferred:` prefixes** in your analysis columns when it
  would otherwise be ambiguous whether a claim is the paper's or yours.

### 7.5 When to stop and ask the user — IMMEDIATELY

**Whenever the process gets stuck, you hit an API error, or you have doubts or
questions (small doubt or big one doesn't matter), ask the user immediately.**

Concretely, stop and ask the user — do not silently work around — in *any* of these
situations:

- **API error / tool failure.** Tool call returns an error you cannot resolve in one
  retry. Report the exact error message and what you were trying to do.
- **Missing or corrupted source file.** The mineru markdown for a paper does not
  exist, is empty, or is so corrupted that even PDF fallback would not recover it.
- **PDF fallback needed but you cannot read the PDF.** State clearly that you tried
  the markdown, why it was unusable, and that you need guidance on whether to (a)
  attempt to read the PDF, (b) skip the paper with a note, or (c) get the file
  re-extracted.
- **Score-vs-content mismatch you cannot resolve.** A paper is scored 5 but on
  reading it is actually a 3 (or vice versa) — flag this to the user before deciding
  how to treat it. Do not silently downgrade or upgrade.
- **Ambiguity in the paper itself** that materially changes a column value (e.g.
  "is the train/test split subject-disjoint or random?"; "does Table II's mean
  contradict the per-subject values?"). Ask before guessing.
- **Class count or task type ambiguous.** If methods say 8 classes but results table
  shows 4, ask.
- **Folder structure / IDs ambiguous.** If the paper-ID-to-arnumber mapping does not
  produce a unique mineru file, ask before guessing.
- **You're about to depart from these instructions.** Any planned deviation from
  Sections 1–6 above (different column count, alternate JSON schema, different Excel
  styling, splitting one paper into two JSONs, etc.) must be approved by the user
  first.
- **Small doubts count.** Even a one-line "I'm not sure if this number is RMSE in ms
  or in degrees" is worth asking about. A 30-second clarification beats a wrong
  JSON value buried in a 14-paper batch.

**How to ask:** state the paper ID, the specific column or section affected, what
you have already checked, and the candidate options you see. Keep it to a few short
lines. Do not turn a question into a long defensive memo — short, specific, and
honest.

**Do not ask the user as a stalling tactic.** If the answer is clearly already in the
paper or in these instructions, find it instead of asking. The threshold is: "is
there a non-trivial chance I am about to record something wrong in the JSON?" — if
yes, ask.

### 7.6 Summary

The user's standing instructions for this work:

1. **Take proper time. Time is not the bottleneck. Quality is.**
2. **Read each paper at least twice, three times if complex.**
3. **Reason, reiterate, analyse — then write.**
4. **Cross-check every JSON (structural Step A + content Step B) before moving on.**
5. **Verify every number against its source table or sentence.**
6. **Flag inconsistencies explicitly in `col_35`.**
7. **Ask the user immediately on any API error, missing file, ambiguity, or doubt
   (small or big — doesn't matter).**

If you find yourself moving fast because the paper "seems simple", slow down. The
simple-looking papers are the ones where details get missed.


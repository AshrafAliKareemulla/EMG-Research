# STATE — FORS-EMG (L1 canonical dataset)

> Read before touching `fors_emg`. Records the raw data, decisions, conversion, and **why**.
> Design doc: `semg-datasets/semg-dataset-setup.md`. Paper: "FORS-EMG: A Novel sEMG Dataset
> for Hand Gesture Recognition Across Multiple Forearm Orientations" (Rumman et al.),
> `literature-review-papers/dataset-papers/mineru_output/FORS EMG.md`. Kaggle:
> https://www.kaggle.com/datasets/ummerummanchaity/fors-emg-a-novel-semg-dataset
> FORS-EMG is the first dataset we onboard whose headline factor is **forearm orientation**.

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~436 MB) + `manifest.parquet` (3420 trials) produced by `semg/adapters/fors_emg.py`.
`scripts/validate_l1.py --datasets fors_emg` → ALL PASS. Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **trial**, shape `(8, 8000)` `float32`, `lzf`,
  chunked `(8, ~1 s)` = `(8, 985)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — one row per trial (3420 rows). **Need both files together.**

Raw source (L0, read-only): `semg-datasets/fors/zip_file_contents/FORS-EMG/Subject<n>/<Orientation>/<Gesture>-<trial>.mat`.

## 2. What FORS-EMG is (the raw dataset)
- **19 healthy subjects** (16M/3F, age 22–40, Bangladeshi), **single session**.
- **8 channels**, MFI bar electrodes (gel): **CH1–4 near the ELBOW, CH5–8 on the MID-FOREARM**
  (CH5–8 aligned to the same muscles as CH1–4). A common ground sat on the dorsal hand.
- **985 Hz**, 10-bit, stored in **physical units (V)** (verified range ≈ ±2.5 V). RAW sEMG.
- **12 hand gestures** (3 single-finger, 5 multi-finger, 4 wrist), each repeated **5 trials**
  in each of **3 forearm orientations** (pronation, rest, supination). 8 s per clip.
- One `.mat` = ONE trial, holding a single variable `value` of shape **(8, 8000)** =
  (channels, time). 19 × 3 × 12 × 5 = **3420** files.

### Gesture map (filename stem → label, acronym, group; order = gestures_name.txt)
| label | stem | acr | group | | label | stem | acr | group |
|---|---|---|---|---|---|---|---|---|
| 1 | Thumb_UP | TU | finger | | 7 | Hand_Close | HC | finger |
| 2 | Index | IDX | finger | | 8 | Hand_Open | HO | finger |
| 3 | Right_Angle | RA | finger | | 9 | Wrist_Extension | WE | wrist |
| 4 | Peace | PCE | finger | | 10 | Wrist_Flexion | WF | wrist |
| 5 | Index_Little | IL | finger | | 11 | Ulner_Deviation | UD | wrist |
| 6 | Thumb_Little | TL | finger | | 12 | Radial_Deviation | RD | wrist |
("Ulner" is the dataset's own spelling of Ulnar.)

## 3. Decisions — and WHY (user-confirmed)
1. **Orientation = explicit `orientation` column; `session` stays 1.** FORS is single-day,
   so mapping orientation onto `session` would be dishonest and would make the generic
   splitter offer a misleading "cross-day" protocol. The paper's headline protocol
   (train on **rest**, test on **all** orientations) is a FORS-specific manifest query on
   `orientation` ∈ {pronation, rest, supination}. (Considered & rejected: orientation→session.)
2. **Labels 1..12; 0 reserved.** FORS has **no rest class**, so label 0 never appears. Order
   follows the dataset's gestures_name.txt. Keeping 0 = universal rest/none sentinel keeps
   cross-dataset label logic consistent with DB1/DB2/DB4/DB5/EMAHA.
3. **`gesture_group` column (finger/wrist)** added (dataset-specific) to support the paper's
   finger-vs-wrist analysis. finger=2280 rows, wrist=1140.
4. **Store the FULL 8 s clip natively.** Each clip has rest padding (active region ≈5–7 s),
   but FORS provides **NO per-sample onset markers** → active-region extraction is a
   downstream/load-time concern (energy onset detection). We never fabricate an onset. ⚠
   Windowing the full clip will label start/end rest windows as the gesture — handle with
   onset detection or centre-window selection downstream (see §6).
5. **No transpose.** `value` is already `(8, 8000)` = (channels, time); the paper's Table II
   "8000×8" is a doc/data discrepancy — the data wins. (Adapter still defends with a
   shape check that would transpose a `(8000, 8)` array if it ever appeared.)
6. Store native (8 ch, 985 Hz, raw); `is_envelope=False`. Resample/filter at load time.

## 4. Manifest schema (columns)
REQUIRED 18 (per design doc §8.1) + dataset-specific `native_label`, `orientation`,
`gesture_group`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_label,
orientation, gesture_group, fs, n_channels, n_samples, electrode_layout, recording_device,
electrode_type, subject_type, is_envelope, adapter_version, ingested_utc, domain`.
Values: `fs=985`, `n_channels=8`, `electrode_layout="mfi_elbow_midforearm_8"`,
`recording_device="Custom 8-channel MFI sEMG system"`, `electrode_type="MFI_bar_gel"`,
`subject_type="healthy"`, `session=1`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name` = gesture stem (e.g. `Hand_Close`); `repetition` = trial 1..5.
`trial_key` = `S01/rest/HC/t3` (subject / orientation / acronym / trial).
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO or cross-orientation claim.

## 5. Validation (ACTUAL ingest, 2026-06-16) — clean
- 19 subjects ✓; orientations {pronation, rest, supination} ✓; labels 1..12, 12 gestures,
  **rest_present=False as expected** ✓; total trials **3420 = 19×3×12×5** ✓;
  per (subject,orientation) = 60 trials ✓; all `n_samples`=8000 ✓; all `trial_key`s unique ✓;
  no NaN; schema + dtypes pass; LOSO subject-disjoint ✓; global normalizer train-only ✓.
- `scripts/validate_l1.py --datasets fors_emg` → PASS (full suite, all 6 datasets PASS).

## 6. What this enables / limits
- **Enables (the headline):** **cross-orientation generalization** — train `orientation=='rest'`,
  test pronation+supination (the paper's protocol; LDA+SNTDF F1 = 88.58%). Also LOSO,
  within-subject, finger-vs-wrist (`gesture_group`), and **elbow-vs-mid-forearm channel**
  studies (CH1–4 vs CH5–8 — a load-time channel-subset choice, NOT a manifest column).
- **Limits:**
  - single session → no cross-day; healthy only → no clinical transfer.
  - **No onset markers / rest padding inside each clip** → naive full-clip windowing mislabels
    rest windows as the gesture; downstream must do onset detection or pick the active centre.
  - **985 Hz** is an unusual rate; for cross-dataset combine, resample at load time.
  - Label space (12 finger/wrist gestures) does NOT match NinaPro/EMAHA taxonomies →
    cross-dataset supervised use needs an explicit label map (design doc §5.7).

## 7. How to re-run
```
python -m semg.adapters.fors_emg \
  --raw "E:/sEMG Research Enhanced/semg-datasets/fors/zip_file_contents/FORS-EMG" \
  --out "E:/sEMG Research Enhanced/data/L1/fors_emg"
```
Reads one `.mat` at a time (low RAM); ~436 MB h5.

## 8. Add-on files (not ingested, kept in L0)
`Database Info/`: `readme.txt`, `gestures_name.txt`, `gestures_sequence.jpg`,
`Orientations.png`, `Subject Details.csv` (names/age/height/disease — subject 9 has
diabetes), `Signal Acquisition.pdf` (device + electrode placement).

# STATE — SeNic (L1 canonical dataset)

> Read before touching `senic`. Records the raw data, decisions, conversion, and **why**.
> Design doc: `semg-datasets/semg-dataset-setup.md`. Paper: "SeNic: An Open Source Dataset
> for sEMG-Based Gesture Recognition in Non-Ideal Conditions", Zhu et al., IEEE TNSRE 2022,
> `literature-review-papers/dataset-papers/mineru_output/SeNic.md`. GitHub:
> https://github.com/BoZhuBo/SeNic
> SeNic is our **first genuinely multi-session dataset** (real cross-day) and the first
> built around **electrode-shift** robustness.

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` + `manifest.parquet` (24,486 trials) produced by `semg/adapters/senic.py`.
`scripts/validate_l1.py --datasets senic` → PASS (see §5). Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **trial**, shape `(8, T)` `float32`, `lzf`, chunked
  `(8, ~1 s)` = `(8, 200)`. Attrs: `fs`, `label`, `subject`. T varies (~1200–1800 ≈ 6–9 s).
- `manifest.parquet` — one row per trial (24,486 rows). **Need both files together.**

Raw source (L0, read-only): `semg-datasets/senic/h<n>/<session>/emg_p<i>_r<j>_<gesture>.csv`
plus per-session `h<n>/Angle_h<n>_<session>.xlsx` and root `SubjectsInfo.xlsx`.

## 2. What SeNic is (the raw dataset)
- **36 healthy subjects** (h0–h35; 24.6±2.2 yr; 11 female — note: repo readme corrects h35
  to female vs the paper's "male"). **Myo armband, 8 channels, 200 Hz**, raw Myo 8-bit sEMG.
- **7 gestures** (NO rest class): `fist, pinch_middlefinger, two, open_hand,
  pinch_forefinger, varus` (Add of Wrist), `eversion` (Abd of Wrist). Each clip ~6–9 s
  (2 s rest + 4–7 s gesture hold).
- Per session: **231 trials = 11 positions (p0–p10) × 7 gestures × 3 reps (r0–r2)**.
- **Variable sessions/subject:** h0–h5 = 10, h6–h13 = 3, h14–h35 = 1.
- **Total = 24,486 trials** (verified; every session has exactly 231).
- One CSV = one trial: **headerless, 8 columns (channels)**, raw values.
- **Five non-ideal factors** by design: electrode shift, individual difference, inter-day,
  muscle fatigue, arm posture.

### The `position` index has TWO meanings (critical)
- **h0–h29:** p0–p10 are **electrode-shift positions** — the armband is rotated around the
  forearm; each channel's absolute ruler angle (0–360°) is recorded in
  `Angle_h{subj}_{sess}.xlsx` (11 rows × 8 CH columns). p0 = initial/reference (~0° shift).
- **h30–h35 (fatigue subjects):** p0–p10 are **fatigue STAGES** — electrodes stay fixed,
  subjects do dumbbell wrist exercises between blocks. **No Angle file exists** for these.

## 3. Decisions — and WHY (user-confirmed)
1. **1-based numbering for repo consistency.** Native `h0..h35` → `subject` 1..36 (native id
   kept in `native_subject` = "h0".."h35"). Native session `0..` → `session` 1..N. Native
   rep `r0..r2` → `repetition` 1..3. **`position` kept 0..10** (a within-session block index;
   p0 = reference). (Considered & rejected: keep everything native 0-based — breaks the 1..N
   convention every other dataset + the validator use.)
2. **Electrode-shift angles parsed into the manifest** (SeNic's scientific core):
   `ch1_angle_deg..ch8_angle_deg` = each channel's absolute ruler degrees at that position;
   `shift_deg` = mean rotation vs position 0 (mod 360). **NaN for h30–h35** (no Angle file).
3. **Factor columns:** `description` (RI=rotate inward / RO=outward / RM=small-angle /
   **FA**=fatigue-enhanced, per subject from SubjectsInfo.xlsx) + boolean `is_fatigue`
   (True for h30–h35). ⚠ The xlsx spells fatigue **`FA`** while the paper's Table I says
   `FE` — the adapter accepts both so `is_fatigue` is robust to that discrepancy.
4. **Labels 1..7 (no rest class); 0 reserved** as the universal rest/none sentinel
   (consistent with FORS). Gesture order = paper Fig. 3(b) excl. rest (fist=1 … eversion=7).
5. **Store the FULL clip natively.** Each clip has 2 s rest padding and **no onset markers**
   → active-region/onset detection is a downstream/load-time concern (like FORS). We never
   fabricate an onset.
6. Store native (8 ch, 200 Hz, raw); `is_envelope=False`. CSV `(T,8)` → stored `(8,T)`.
   HDF5 chunk `(8, ~1 s)` + `lzf`.
7. **`session` is real here.** SeNic is multi-session → the generic splitter offers genuine
   **cross-day** protocols for h0–h13 (`has_multiple_sessions` = True).

## 4. Manifest schema (columns)
REQUIRED 18 (design doc §8.1) + dataset-specific
`native_label, native_subject, position, shift_deg, ch1_angle_deg..ch8_angle_deg,
description, is_fatigue`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_label,
native_subject, position, shift_deg, ch1_angle_deg, ch2_angle_deg, ch3_angle_deg,
ch4_angle_deg, ch5_angle_deg, ch6_angle_deg, ch7_angle_deg, ch8_angle_deg, description,
is_fatigue, fs, n_channels, n_samples, electrode_layout, recording_device, electrode_type,
subject_type, is_envelope, adapter_version, ingested_utc, domain`.
Values: `fs=200`, `n_channels=8`, `electrode_layout="myo_8"`,
`recording_device="Thalmic Myo armband"`, `electrode_type="myo_dry_stainless"`,
`subject_type="healthy"`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name` = gesture stem (e.g. `fist`); `repetition` = 1..3.
`trial_key` = `S01/sess1/p00/r1/fist` (subject / session / position / rep / gesture).
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO claim;
`mode="per_session"` is meaningful here for cross-day work (design doc §5.5).

## 5. Validation (ACTUAL ingest, 2026-06-16)
- subjects 1..36 ✓; sessions/subject {10:6, 3:8, 1:22} ✓; labels 1..7, 7 gestures,
  **rest_present=False as expected** ✓; positions 0..10 ✓; total trials **24,486** ✓;
  per (subject,session) = 231 ✓; all `trial_key`s unique ✓; schema + dtypes pass;
  LOSO subject-disjoint ✓; **multi-session → cross-day available** ✓; global normalizer
  train-only ✓.
- `shift_deg`/angle columns: non-null for h0–h29 (shift subjects), NaN for h30–h35 (fatigue).
- `description`: RI/RO/RM/FE counts as expected; `is_fatigue`=True only for h30–h35.
- `scripts/validate_l1.py --datasets senic` → PASS (full suite: all datasets PASS).
> (This section is filled from the actual ingest+validator run; see SESSIONS.md entry.)

## 6. What this enables / limits
- **Enables (headline):**
  - **Cross-day / inter-day** (h0–h5 over 10 sessions, h6–h13 over 3) — our first real
    cross-session dataset; query `session`.
  - **Electrode-shift robustness** — train on p0 (reference), test on shifted p1–p10; the
    actual shift magnitude is `shift_deg` / per-channel `chX_angle_deg` (queryable, not just
    an index). This is SeNic's unique contribution.
  - **Muscle fatigue** — h30–h35, `is_fatigue=True`, `position` = fatigue stage.
  - LOSO / within-subject; consumer-grade Myo @200 Hz (pairs naturally with NinaPro DB5,
    also Myo-class, for "consumer armband" studies).
- **Limits:**
  - **Arm posture (SBE/uSBE/DAN)** varies across h6–h13 sessions, but the dataset files do
    **NOT record the per-session posture order** — only described in the paper prose. So
    posture is **not encoded per-trial** (documented limitation; never guessed). If the
    posture↔session mapping is obtained from the authors, add a `posture` column then.
  - **`position` is overloaded** (shift for h0–h29, fatigue stage for h30–h35) — always read
    it together with `is_fatigue`/`description`. `shift_deg` is NaN for fatigue subjects.
  - **No onset markers / 2 s rest padding** inside each clip → naive full-clip windowing
    mislabels rest windows as the gesture; downstream needs onset detection or active centre.
  - **200 Hz** caps usable EMG band (~0–100 Hz, Myo); no power-line noise (removed by Myo HW).
  - Healthy only; rotational shift only (no longitudinal); Myo 8-bit low dynamic range.
  - Label space (7 Myo gestures) does NOT match NinaPro/EMAHA/FORS → cross-dataset
    supervised use needs an explicit label map (design doc §5.7).

## 7. How to re-run
```
python -m semg.adapters.senic \
  --raw "E:/sEMG Research Enhanced/semg-datasets/senic" \
  --out "E:/sEMG Research Enhanced/data/L1/senic"
```
Reads 24,486 CSVs + 100 Angle .xlsx (one .mat-free; pandas). Takes a few minutes.

## 8. Add-on files (kept in L0, not ingested as signals)
`SubjectsInfo.xlsx` (per-subject demographics + Description RI/RO/RM/FE + Sessions count) —
used by the adapter for `description`. `Angle_h{subj}_{sess}.xlsx` (h0–h29) — parsed into
the shift-angle columns. `readme.txt` (folder/naming spec + h35-gender correction).

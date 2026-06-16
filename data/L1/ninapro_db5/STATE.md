# STATE — NinaPro DB5 (L1 canonical dataset)

> Read this before touching `ninapro_db5`. Records the raw data, decisions, conversion,
> and **why**. Companion design doc: `semg-datasets/semg-dataset-setup.md`. See the DB1
> STATE.md for the shared offset rationale and the DB4 STATE.md for the sibling
> (Pizzolato et al. 2017) dataset; this file emphasises where DB5 **differs**.
> Official: https://ninapro.hevs.ch/instructions/DB5.html

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~208 MB) + `manifest.parquet` (6270 trials) produced and cross-checked.
DB5 is perfectly regular (no missing/fragmented reps, unlike its sibling DB4). Ready to copy.

## 1. What this folder is (after ingest)
Canonical **L1** form produced by `semg/adapters/ninapro_db5.py`.
- `signals.h5` — one HDF5 dataset per **trial**, shape `(16, T)` `float32`, `lzf`,
  chunked `(16, ~1 s)` = `(16, 200)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — one row per trial. **Need both files together.**

Raw source (L0, read-only): `D:\Complete EMG Dataset\ninapro_db5\downloaded_data\s<n>\s<n>\S<n>_E<ex>_A1.mat`
(doubly-nested `s<n>/s<n>/` folder, `S<n>_E<ex>_A1` name order — like DB2/DB4).

## 2. What NinaPro DB5 is (the raw dataset)
- **10 intact (healthy) subjects**, all right-handed (age 22–34, height 156–187 cm,
  weight 52–83 kg), single session.
- **52 hand movements + rest**, **6 repetitions** each, split across 3 exercise files
  (E1=12 finger movements, E2=17 isometric/wrist, E3=23 grasping/functional).
- **16 channels = 2× Thalmic Myo armbands.** Cols 1–8 = first armband equally spaced
  around the forearm at the radio-humeral joint; cols 9–16 = second armband, nearer the
  elbow and **tilted 22.5° clockwise**.
- **200 Hz** (Myo's native low rate — the lowest of all our datasets), **RAW Myo sEMG**
  (verified: signed 8-bit, range −128..127) — NOT an envelope.
- Files **also** contain `acc` (1× 3-axis Myo IMU) and `glove` (22-ch Cyberglove).

### Raw variables and how we used them
| Variable | Meaning | Used? |
|---|---|---|
| `emg` (T×16) | raw Myo sEMG | ✅ stored (16×T) |
| `restimulus` | movement label, **time-refined, per-exercise local** | ✅ labels + boundaries |
| `rerepetition` | repetition, time-refined | ✅ split the 6 reps |
| `acc` (T×3) | Myo 3-axis accelerometer | ❌ ignored (re-ingestable for future EMG+IMU fusion) |
| `glove` (T×22) | Cyberglove kinematics | ❌ ignored (EMG classification, not regression) |
| `stimulus`/`repetition` | raw versions | ❌ ignored |
| `exercise` (scalar) | **scrambled — do NOT use (see §3.3)** | ❌ exercise taken from filename |

## 3. Decisions — and the DB5-specific ones in **bold**
1. Use `restimulus`/`rerepetition` (time-refined). Same rationale as DB1/DB2/DB4.
2. **Per-exercise LOCAL labeling, like DB1/DB4 (UNLIKE DB2).** Verified E1→1..12, E2→1..17,
   E3→1..23. Global 1..52 via offset `{E1:+0, E2:+12, E3:+29}` — verified to yield exactly
   52 distinct movements/subject. Stored `native_label`(local) + `exercise`.
3. **(⚠ gotcha) Exercise number is read from the FILENAME, not the internal `exercise`
   scalar.** Scrambled in DB5: E1 file reports `exercise=3`, E2→`1`, E3→`2`. Adapter
   parses `S<n>_E<ex>_A1.mat` and ignores `m['exercise']`. (Same quirk as DB4.)
4. **Store EMG only; ignore `acc` and `glove`.** Per design doc §5.3/§5.7: L1 holds the
   canonical EMG signal so every dataset keeps the identical one-trial-per-HDF5-dataset
   shape and the single generic loader works uniformly. `acc`/`glove` stay recoverable
   from the untouched L0 raw if a future EMG+IMU / EMG→kinematics experiment wants them.
5. Keep rest as label 0 (one trial per rest block). One trial per contiguous
   `(restimulus, rerepetition)` run.
6. **`is_envelope=False`** — raw Myo, so a downstream pipeline *may* bandpass/rectify it.
   (But note: at 200 Hz the usable EMG band is already limited — see §6 caveat.)
7. Store native (200 Hz, 16 ch); resample/channel-select only at load time.
8. HDF5 chunk `(16, ~1 s)` + `lzf`.

## 4. Manifest schema (columns)
Identical schema/order to DB1: `trial_key, dataset_id, subject, session, repetition,
label, label_name, native_label, exercise, fs, n_channels, n_samples, electrode_layout,
recording_device, electrode_type, subject_type, is_envelope, adapter_version,
ingested_utc, domain`.
DB5 values: `fs=200`, `n_channels=16`, `electrode_layout="double_myo_16"`,
`recording_device="Thalmic Myo armband x2"`, `electrode_type="myo_dry_stainless"`,
`subject_type="healthy"`, `session=1`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name = "E<ex>_M<nn>"` (DB1 style). `trial_key` (movements) =
`S01/E1/mov01/rep01/s<start>` (start-sample suffix kept identical to the DB4 sibling for
parallelism; DB5 has no fragmented reps so it isn't strictly needed here); rest =
`S01/E1/rest/seg001`.
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO claim (design doc §5.5).

## 5. Validation (ACTUAL ingest, 2026-06-16) — DB5 is clean
- subjects `1..10` ✓; labels `0..52` (53 incl. rest) ✓; all `trial_key`s unique ✓;
  every subject has all 52 movements ✓; no NaN ✓; no WARN/offset collision ✓;
  schema columns + dtypes identical to DB1 ✓.
- **movement trials = 3120 = 10 × 52 × 6 exactly** (3120 distinct combos, 0 missing,
  0 fragmented reps). rest trials = 3150. total = 6270.
- Unlike DB4, DB5's GLR relabeling produced a fully regular 6-reps-per-movement grid.

## 6. What this enables / limits
- **Enables:** LOSO, within-subject k-fold. **Same 52-movement taxonomy as DB1 & DB4** →
  DB1/DB4/DB5 form the clean same-protocol cross-dataset trio. DB5 is the **low-rate
  (200 Hz), consumer-grade Myo, 16-channel** corner — ideal for "does ≥1 kHz matter?"
  (vs DB4's 2 kHz, same movements) and "consumer armband vs lab electrodes" studies.
- **Limits:** single session → no cross-day. Healthy only. **200 Hz caps the EMG bandwidth
  (~0–100 Hz usable by Nyquist)** — much of the 20–450 Hz sEMG band is absent, so absolute
  accuracy is expected lower than DB4; this is a *feature* for the sampling-rate study, not
  a bug, but downstream filters must respect the 100 Hz ceiling.
- **Cross-dataset channel handling:** 16 ch (double Myo) vs DB1's 10 / DB4's 12 / DB2's 12
  → use dataset-specific stem + shared trunk (design doc §15.3 / §5.7).

## 7. How to re-run
```
python -m semg.adapters.ninapro_db5 \
  --raw "D:/Complete EMG Dataset/ninapro_db5/downloaded_data" \
  --out "E:/sEMG Research Enhanced/data/L1/ninapro_db5"
```
Reads one `.mat` at a time. 200 Hz × 16 ch is small → expect a modest (tens-of-MB) h5.

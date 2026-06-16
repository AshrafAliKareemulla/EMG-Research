# STATE — NinaPro DB4 (L1 canonical dataset)

> Read this before touching `ninapro_db4`. Records the raw data, decisions, conversion,
> and **why**. Companion design doc: `semg-datasets/semg-dataset-setup.md`. See the DB1
> STATE.md for the shared offset rationale; this file emphasises where DB4 **differs**.
> Source papers: Atzori et al. 2014 (Ninapro P1) for protocol conventions; Pizzolato et
> al. 2017 ("Comparison of six EMG acquisition setups") introduced DB4/DB5 specifically.
> Official: https://ninapro.hevs.ch/instructions/DB4.html

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~2585 MB) + `manifest.parquet` (6276 trials) produced and cross-checked.
Validation in §5 reflects ACTUAL ingest results. Files are ready to copy to the GPU box.

## 1. What this folder is (after ingest)
Canonical **L1** form produced by `semg/adapters/ninapro_db4.py`.
- `signals.h5` — one HDF5 dataset per **trial**, shape `(12, T)` `float32`, `lzf`,
  chunked `(12, ~1 s)` = `(12, 2000)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — one row per trial. **Need both files together.**

Raw source (L0, read-only): `D:\Complete EMG Dataset\ninapro_db4\downloaded_data\s<n>\s<n>\S<n>_E<ex>_A1.mat`
(note the **doubly-nested** `s<n>/s<n>/` folder, and `S<n>_E<ex>_A1` name order — like DB2).

## 2. What NinaPro DB4 is (the raw dataset)
- **10 intact (healthy) subjects**, single session.
- **52 hand movements + rest**, **6 repetitions** each, split across 3 exercise files
  (E1=12 finger movements, E2=17 isometric/wrist, E3=23 grasping/functional).
- **12 channels**, **Cometa** electrodes (8 equally spaced around forearm at the
  radio-humeral joint; 9–10 flexor/extensor digitorum superficialis; 11–12 biceps/triceps).
- **2000 Hz**, **RAW bipolar sEMG** (verified: signed, ~±18k–30k ADC counts) — NOT an envelope.
- Files contain **no glove and no acc** (unlike DB5) — only `emg` + labels + subject metadata.

### Raw variables and how we used them
| Variable | Meaning | Used? |
|---|---|---|
| `emg` (T×12) | raw sEMG | ✅ stored (12×T) |
| `restimulus` | movement label, **time-refined, per-exercise local** | ✅ labels + boundaries |
| `rerepetition` | repetition, time-refined | ✅ split the 6 reps |
| `stimulus`/`repetition` | raw versions | ❌ ignored |
| `exercise` (scalar) | **scrambled — do NOT use (see §3.2)** | ❌ exercise taken from filename |
| `subject`,`age`,`gender`,`height`,`weight`,`laterality`,`circumference`,`sensor`,`frequency` | subject metadata | partial (subject from filename) |

## 3. Decisions — and the DB4-specific ones in **bold**
1. Use `restimulus`/`rerepetition` (time-refined). Same rationale as DB1/DB2.
2. **Per-exercise LOCAL labeling, like DB1 (UNLIKE DB2).** Verified ranges E1→1..12,
   E2→1..17, E3→1..23 (each file restarts at 1). Global 1..52 via offset
   `{E1:+0, E2:+12, E3:+29}` — verified to yield exactly 52 distinct movements/subject,
   range 1..52. Stored `native_label`(local) + `exercise` for traceability.
3. **(⚠ gotcha) Exercise number is read from the FILENAME, not the internal `exercise`
   scalar.** The internal field is scrambled in DB4: the E1 file reports `exercise=2`,
   E2 reports `1`, E3 reports `3`. Trusting it would corrupt the label offset. The adapter
   parses `S<n>_E<ex>_A1.mat` with a regex and ignores `m['exercise']`. (DB1's adapter
   trusted the internal field; that would be a BUG here — this is exactly why each dataset
   is verified rather than cloned blindly.)
4. Keep rest as label 0 (one trial per rest block). Same as DB1/DB2.
5. One trial per contiguous `(restimulus, rerepetition)` run.
6. **`is_envelope=False`** — DB4 is raw, so a downstream pipeline *should* bandpass/rectify it.
7. Store native (2 kHz, 12 ch); resample/channel-select only at load time.
8. HDF5 chunk `(12, ~1 s)` + `lzf` (read-speed tuning, window-config independent).

## 4. Manifest schema (columns)
Identical schema/order to DB1: `trial_key, dataset_id, subject, session, repetition,
label, label_name, native_label, exercise, fs, n_channels, n_samples, electrode_layout,
recording_device, electrode_type, subject_type, is_envelope, adapter_version,
ingested_utc, domain`.
DB4 values: `fs=2000`, `n_channels=12`, `electrode_layout="cometa_forearm_12"`,
`recording_device="Cometa miniWave wireless"`, `electrode_type="cometa_active_dry"`,
`subject_type="healthy"`, `session=1`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name = "E<ex>_M<nn>"` (local id, DB1 style). `trial_key` (movements) =
`S01/E1/mov01/rep01/s<start>` — **the trailing `/s<start>` (run start sample) is appended
so that fragmented reps get unique keys** (see §5 anomaly); rest = `S01/E1/rest/seg001`.
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO claim (design doc §5.5).

## 5. Validation (ACTUAL ingest, 2026-06-16) + a real DB4 data anomaly
- subjects `1..10` ✓; labels `0..52` (53 incl. rest) ✓; all `trial_key`s unique ✓;
  every subject has all 52 movements ✓; no NaN in EMG ✓; no offset collision (no WARN) ✓;
  schema columns + dtypes identical to DB1 ✓.
- **movement trials = 3123 rows across 3117 distinct (subject,movement,rep) combos**
  (NOT a clean 3120). rest trials = 3153. total 6276.
- **⚠ DB4 relabeling anomaly (verified against raw, NOT a bug, NOT a download problem).**
  NinaPro's GLR refinement that produces `restimulus`/`rerepetition` **drops 3 weak
  repetitions** — it re-labels their movement samples as rest (`restimulus=0`):
    - **S4, mov 3 (E1 "Middle finger flexion"), rep 6** — raw `stimulus` has rep 6
      (10,889 samples); refined relabeled them rest. Refined keeps reps 1–5.
    - **S4, mov 10 (E1 "Thumb abduction"), rep 3** — raw has rep 3 (10,697 samples);
      refined keeps reps 1,2,4,5,6.
    - **S6, mov 10 (E1 "Thumb abduction"), rep 6** — raw has rep 6 (10,696 samples);
      refined keeps reps 1–5.
  The raw `stimulus`/`repetition` still contain all 6 reps; only the *refined* labels lose
  them. We use refined labels (NinaPro's documented recommendation, same as DB1/DB2), so
  these 3 reps are intentionally absent. **No reinstall needed — the official DB4 download
  has exactly this; verified all 30 .mat files present and intact.**
- **Rep fragmentation:** 5 reps are split into 2 non-contiguous bursts by the relabeling
  (e.g. S4 E1 mov3 rep3) → stored as 2 trials each (unique via `/s<start>` key), faithful
  to timing (no silent merge). This is why row count (3123) > combo count (3117).
- Folder cleanliness: one harmless macOS `.DS_Store` in `s3/s3/` (ignored by the glob).

## 6. What this enables / limits
- **Enables:** LOSO, within-subject k-fold. Same 52-movement protocol as DB1 → a natural
  **same-protocol, different-hardware** pair with DB1 (Otto Bock envelope 100 Hz ⇄ Cometa
  raw 2 kHz): clean ground for "does device/rate matter?" and cross-dataset transfer where
  the label taxonomy actually matches (rare across Ninapro DBs — DB4↔DB1↔DB5 share it).
- **Limits:** single session → no cross-day. Healthy only → no clinical transfer alone.

## 7. Cross-dataset note
DB1, DB4, DB5 all use the **same 52-movement / 3-exercise taxonomy** → their global 1..52
labels are directly comparable (unlike DB2's 49-movement set). This makes DB1/DB4/DB5 the
clean supervised cross-dataset trio; DB2 needs an explicit label map (design doc §5.7).

## 8. How to re-run
```
python -m semg.adapters.ninapro_db4 \
  --raw "D:/Complete EMG Dataset/ninapro_db4/downloaded_data" \
  --out "E:/sEMG Research Enhanced/data/L1/ninapro_db4"
```
Reads one `.mat` at a time (low RAM). Raw EMG is ~2 kHz × 12 ch → expect a few-hundred-MB h5.

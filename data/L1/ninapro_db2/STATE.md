# STATE — NinaPro DB2 (L1 canonical dataset)

> Read this before touching `ninapro_db2`. Records the raw data, decisions, conversion,
> and **why** — so a future session can reason about it without re-deriving everything.
> Companion design doc: `semg-datasets/semg-dataset-setup.md`. See also the DB1 STATE.md
> for the shared rationale; this file emphasises where DB2 **differs** from DB1.

## 1. What this folder is
Canonical **L1** form of NinaPro DB2 produced by `semg/adapters/ninapro_db2.py`.

- `signals.h5` (~10 GB) — one HDF5 dataset per **trial**, shape `(12, T)` `float32`, `lzf`-compressed, chunked `(12, ~1 s)` = `(12, 2000)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` (23,640 rows) — one row per trial. **Need both files together.**

Raw source (L0, read-only): `D:\Complete EMG Dataset\ninaprodb2\downloaded_data\DB2_s<n>\DB2_s<n>\S<n>_E<ex>_A1.mat` (note the **nested** folder and `S<n>_E<ex>_A1` name order — different from DB1's `S<n>_A1_E<ex>`).

## 2. What NinaPro DB2 is (the raw dataset)
- **40 intact (healthy) subjects**, single session.
- **49 hand movements + rest**, **6 repetitions** each, split across 3 exercise files (E1: movements 1–17, E2: 18–40, E3: 41–49).
- **12 channels**, **Delsys Trigno** (8 around forearm + flexor/extensor digitorum + biceps/triceps), **2000 Hz**.
- **RAW bipolar sEMG** (verified: zero-mean, ~±0.005) — NOT an envelope.

### Raw variables and how we used them
| Variable | Meaning | Used? |
|---|---|---|
| `emg` (T×12) | raw sEMG | ✅ stored (12×T) |
| `restimulus` | movement label, **time-refined, already global** | ✅ labels + boundaries |
| `rerepetition` | repetition, time-refined | ✅ split the 6 reps |
| `stimulus`/`repetition` | raw versions | ❌ ignored |
| `glove` (T×22) | Cyberglove | ❌ ignored |
| `acc` (T×36) | 12× 3-axis accelerometer (IMU) | ❌ ignored (kept for possible future multimodal work) |
| `inclin` (T×2) | inclinometer | ❌ ignored |
| `force`/`forcecal` | force (exercise 3) | ❌ ignored |

## 3. Decisions — and the DB2-specific ones in **bold**
1. Use `restimulus`/`rerepetition` (time-refined). Same rationale as DB1.
2. **NO label offset.** Unlike DB1, DB2's `restimulus` is **already globally numbered across exercises** (verified: E1 max=17, E2 max=40, E3 max=49). So `label = restimulus` directly. (This is exactly why we verify each dataset rather than copy DB1's logic.)
3. Keep rest as label 0 (one trial per rest block). Same as DB1.
4. One trial per contiguous `(restimulus, rerepetition)` run.
5. **Ignore `acc`/`glove`/`inclin`/`force`** — only `emg` is stored. (`acc` is genuine IMU and could matter for a future EMG+IMU fusion experiment; re-runnable then.)
6. **`is_envelope=False`** — DB2 is raw, so a downstream pipeline *should* bandpass/rectify it (opposite of DB1). This flag is the single most important difference to remember.
7. Store native (2 kHz, 12 ch); resample/channel-select only at load time.
8. HDF5 chunk `(12, ~1 s)` + `lzf` (read-speed tuning, window-config independent).

## 4. Manifest schema (columns)
Identical schema to DB1: `trial_key, dataset_id, subject, session, repetition, label, label_name, native_label, exercise, fs, n_channels, n_samples, electrode_layout, recording_device, electrode_type, subject_type, is_envelope, adapter_version, ingested_utc, domain`.
(adapter_version 1.0.0; recording_device "Delsys Trigno wireless"; electrode_type "delsys_active_dry". NOTE: these 4 metadata columns were added to the DB2 manifest **by `scripts/patch_db2_manifest.py`** — the 10 GB `signals.h5` was NOT rewritten since its content is unchanged. The adapter is the forward source of truth.)
**Normalization:** `Normalizer` `mode="global"` default for cross-subject/LOSO (design doc §5.5).

DB2 values: `fs=2000`, `n_channels=12`, `electrode_layout="delsys_trigno_12"`, `subject_type="healthy"`, `session=1`, `is_envelope=False`. `label_name = "M<nn>"`. `trial_key` = `S01/E1/mov18/rep03` or `S01/E2/rest/seg007`.

## 5. Validation that passed
- Movement trials = **11,760 = 40 × 49 × 6** ✓
- Subject 1 has **49 distinct movements** ✓
- Labels `0..49` (50 incl. rest), all `trial_key`s unique ✓
- 11,880 rest trials. Total 23,640 trials.

## 6. What this enables / limits
- **Enables:** LOSO, within-subject k-fold. Larger/raw/high-rate → good for DL, and for the "does ≥1 kHz matter?" study (can be downsampled at load time, comparable to DB1's 100 Hz).
- **Limits:** single session → no cross-day on DB2 (handled by `session` field = 1). Healthy only.
- **Cross-dataset with DB1:** possible but heterogeneous — different channels (10 vs 12), rate (100 Hz envelope vs 2 kHz raw), and **label spaces do not match** (different movement sets/ids). For combined supervised training an explicit label map is required; SSL-on-union is the clean default (design doc §5.7).

## 7. Scalability note (why ~10 GB is fine)
DB2's `signals.h5` is **34× larger than DB1's** (294 MB → 10 GB). This is purely a disk fact. Ingest read one `.mat` at a time (peak RAM ≈ one subject file). Training uses the lazy `WindowDataset`, so DB2 consumes the **same RAM/VRAM as DB1** — only one batch reaches the GPU. The 2 kHz rate means more samples per window (e.g. 500 ms = 1000 samples vs DB1's 50), which affects per-batch VRAM, handled by batch size / mixed precision — not by dataset size.

## 8. How to re-run
```
python -m semg.adapters.ninapro_db2 \
  --raw "D:/Complete EMG Dataset/ninaprodb2/downloaded_data" \
  --out "E:/sEMG Research Enhanced/data/L1/ninapro_db2"
```
Takes a few minutes (10 GB write). Was run in background.

## 9. Related: NinaPro DB3 (NOT ingested)
DB3 (`D:\Complete EMG Dataset\ninaprodb3`) is **empty** — `downloaded_data/` and `process_data/` contain no files. DB3 is the **amputee** dataset (11 transradial amputees) and is the one that would unlock healthy→amputee clinical-transfer experiments. **Re-download when possible**, then write a DB3 adapter (will need `subject_type="amputee"` and per-subject remaining-muscle notes).

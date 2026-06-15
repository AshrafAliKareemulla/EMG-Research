# STATE — NinaPro DB1 (L1 canonical dataset)

> Read this before touching `ninapro_db1`. It records what the raw data is, what we
> decided, how we converted it, and **why** — so a future session can reason about it
> without re-deriving everything. Companion design doc: `semg-datasets/semg-dataset-setup.md`.

## 1. What this folder is
Canonical **L1** form of NinaPro DB1 produced by the adapter
`semg/adapters/ninapro_db1.py` from the raw `.mat` files.

- `signals.h5` (~294 MB) — one HDF5 dataset per **trial**, shape `(n_channels, time)` = `(10, T)`, `float32`, `lzf`-compressed, chunked `(10, ~1 s)`. Per-dataset attrs: `fs`, `label`, `subject`.
- `manifest.parquet` (28,161 rows) — one row per trial; the queryable catalog. **You need both files together** — h5 is signal only; manifest holds the metadata.

Raw source (L0, read-only, untouched): `D:\Complete EMG Dataset\ninaprodb1\downloaded_data\s1..s27\S<n>_A1_E<ex>.mat`

## 2. What NinaPro DB1 is (the raw dataset)
- **27 intact (healthy) subjects**, single session (`A1`).
- **52 hand movements + rest**, **10 repetitions** each, split across **3 exercise files** per subject (E1=12 finger movements, E2=17 isometric/wrist, E3=23 grasping/functional).
- **10 channels**, **Otto Bock MyoBock 13E200** electrodes, **100 Hz**.
- ⚠️ **The EMG is an ENVELOPE, not raw sEMG.** Otto Bock electrodes output a rectified/smoothed amplitude (verified: all values non-negative, min 0.0 / max 4.66 / mean 0.06).
- Each `.mat` is one continuous 100 Hz stream where `restimulus` holds a movement id during a movement and `0` during rest between movements.

### Raw variables and how we used them
| Variable | Meaning | Used? |
|---|---|---|
| `emg` (T×10) | the sEMG envelope | ✅ stored (transposed to 10×T) |
| `restimulus` | movement label, **time-refined** | ✅ labels + trial boundaries |
| `rerepetition` | repetition index, **time-refined** | ✅ to split the 10 reps |
| `stimulus` / `repetition` | raw (un-refined) versions | ❌ ignored (refined is cleaner — NinaPro's own guidance) |
| `glove` (T×22) | Cyberglove kinematics | ❌ ignored (we do sEMG classification, not regression) |
| `exercise`, `subject` | scalars | ✅ for label offset + manifest |

## 3. Decisions we made and WHY
1. **Use `restimulus`/`rerepetition`, not `stimulus`/`repetition`.** The refined versions have movement boundaries corrected a-posteriori to match the real movement — cleaner trials. This is NinaPro's documented recommendation.
2. **Global 1–52 label space via per-exercise offset** (`E1 +0`, `E2 +12`, `E3 +29`). DB1 numbers movements *per exercise* (each file restarts at 1), so without the offset three different movements would collapse to the same label. Stored `native_label` + `exercise` too, so any global label is traceable back to its source.
3. **Keep rest as its own class (label 0).** Most flexible — can be dropped later by a manifest filter, but can't be recovered if discarded now. Needed for any future null-class / rejection study.
4. **One trial per contiguous `(restimulus, rerepetition)` run.** Each movement-rep and each rest block becomes its own trial. No EMG samples are lost — movement + rest blocks together cover the whole recording.
5. **Ignore the Cyberglove.** Focus is EMG classification. Re-runnable later if EMG→kinematics regression is ever wanted.
6. **Store native, transform later.** No resampling, no filtering, no channel selection at ingest — those are load-time experiment variables (see design doc §5.1). The `is_envelope=True` flag tells a downstream pipeline **NOT** to re-rectify/bandpass DB1 (the hardware already did it).
7. **HDF5 chunk `(10, ~1 s)` + `lzf`.** Read-speed tuning aligned to typical window reads; independent of window/overlap choice. Light compression decompresses fast under DataLoader workers (gzip avoided).

## 4. Manifest schema (columns)
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_label, exercise, fs, n_channels, n_samples, electrode_layout, recording_device, electrode_type, subject_type, is_envelope, adapter_version, ingested_utc, domain`
(adapter_version 1.0.0; recording_device "Otto Bock MyoBock 13E200"; electrode_type "otto_bock_active_envelope". The `recording_device`/`electrode_type`/`adapter_version`/`ingested_utc` columns were added per the external review — see design doc §15.)
**Normalization:** use `Normalizer` `mode="global"` for any cross-subject/LOSO claim (default); per-subject only within-subject. See design doc §5.5.

DB1-specific values: `fs=100`, `n_channels=10`, `electrode_layout="ottobock_forearm_10"`, `subject_type="healthy"`, `session=1` (single session), `is_envelope=True`.
`trial_key` format: `S01/E1/mov03/rep05` (movements) and `S01/E1/rest/seg012` (rest blocks).

## 5. Validation that passed (run on ingest)
- Movement trials = **14,040 = 27 × 52 × 10** ✓
- Subject 1 has **52 distinct movements** ✓ (offset works)
- Labels `0..52` (53 incl. rest), all `trial_key`s unique ✓
- 14,121 rest trials. Total 28,161 trials.

## 6. What this enables / limits
- **Enables:** LOSO and within-subject k-fold (just manifest queries on `subject`).
- **Limits:** **single session → cross-day/cross-session NOT possible on DB1.** Handled generically by the `session` field (all = 1); the splitter simply won't offer cross-day for DB1.
- Healthy only → no clinical (amputee) transfer from DB1 alone.

## 7. How to re-run
```
python -m semg.adapters.ninapro_db1 \
  --raw "D:/Complete EMG Dataset/ninaprodb1/downloaded_data" \
  --out "E:/sEMG Research Enhanced/data/L1/ninapro_db1"
```
Ingest reads one `.mat` at a time (low RAM); takes ~30 s.

## 8. Open / future notes
- Movement `label_name` is generic (`E1_M03`) — real DB1 movement names (e.g. "thumb up") can be mapped in later from the NinaPro movement table if needed.
- Rest segmentation = one trial per rest block (not merged) — faithful to timing.

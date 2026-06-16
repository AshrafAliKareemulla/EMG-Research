# STATE — GrabMyo (original) (L1 canonical dataset)

> Read before touching `grabmyo`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> Paper: Pradhan, He, Jiang, "Multi-day dataset of forearm and wrist electromyogram for hand
> gesture recognition and biometrics", Sci. Data 2022 — `mineru_output/GrabMyo.md`.
> PhysioNet: https://physionet.org/content/grabmyo/1.1.0/ · IEEE Dataport:
> https://ieee-dataport.org/documents/gesture-recognition-and-biometrics-electromyography-grabmyo-dataset
> Part of the GrabMyo family: see also `grabmyo_flow_static` and `grabmyo_flow_dynamic`.

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~17.4 GB) + `manifest.parquet` (**15,351 trials — COMPLETE**) by
`semg/adapters/grabmyo.py`. `scripts/validate_l1.py --datasets grabmyo` → PASS.
**Session3/participant9 was re-downloaded on 2026-06-16** and the 8 previously-missing trials
were SHA256-verified against the official sums and restored, so the dataset is now the full
43×3×17×7 = 15,351 (0 skipped). Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **trial**, shape `(28, 10240)` `float32` (physical mV),
  `lzf`, chunked `(28, 2048)` = (channels, ~1 s). Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — one row per trial (15,343 rows).

Raw source (L0): PhysioNet WFDB tree `grabmyo/1.1.0/Session{s}/session{s}_participant{p}/
session{s}_participant{p}_gesture{k}_trial{l}.{dat,hea}`. Read with the dependency-free
`semg/adapters/_wfdb_read.py` (no `wfdb` package needed).

## 2. What GrabMyo is (the raw dataset)
- **43 healthy subjects** (23M/20F, 38 right-handed, age 26.4±2.9), **3 sessions on days 1, 8,
  29** (genuine multi-day / cross-day; un-uniform spacing).
- **17 gestures** (16 hand/wrist + **REST**), **7 trials** each. 2048 Hz, **5 s = 10240 samples**.
- **28 monopolar EMG channels = 16 forearm (F1–F16) + 12 wrist (W1–W12)**, two rings each
  (proximal/distal, 2 cm apart). EMGUSB2+ amplifier, gain 500, Ambu pre-gelled electrodes.
- WFDB `.dat` has **32 columns** = 28 EMG + **4 unused/grounded U1–U4** (gain ~1.3–1.5M,
  read as exactly 0 mV) → the 4 U channels are **dropped**.
- Signals stored in **physical units (mV)** via the per-channel `.hea` gain/baseline.
  Paper applied 10–500 Hz Butterworth + 60 Hz notch (filtering state of stored data is the
  authors'; we treat further filtering as a load-time choice).

### Channel order in stored (28, T): F1..F16 (forearm), then W1..W12 (wrist)
Forearm-only = channels 0–15; wrist-only = channels 16–27. Site selection is a **load-time
channel-subset choice**, not a manifest column. (Wrist channels W1–W12 here match the
12-channel `grabmyo_flow_static` set, enabling original↔flow wrist comparisons.)

## 3. Decisions — and WHY (user-confirmed)
1. **Three separate L1 datasets** for the family (this 28ch `grabmyo`; `grabmyo_flow_static`
   12ch wrist extension; `grabmyo_flow_dynamic` transitions) — different channel counts /
   class sets / durations preclude one shared h5.
2. **Labels: REST (native gesture 17) → label 0** (universal rest/none sentinel); gestures
   1–16 keep labels 1–16. `native_label` preserves the dataset's 1–17.
3. **`session` = day index 1/2/3** → genuine cross-day protocols (multi-session, like SeNic).
4. Store native (28 ch, 2048 Hz, physical mV); `is_envelope=False`; resample/filter/
   forearm-vs-wrist channel selection at load time.
5. HDF5 chunk `(28, ~1 s)` + `lzf`.

## 4. Manifest schema (columns)
REQUIRED 18 (design doc §8.1) + `native_label`, `native_subject`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_label,
native_subject, fs, n_channels, n_samples, electrode_layout, recording_device,
electrode_type, subject_type, is_envelope, adapter_version, ingested_utc, domain`.
Values: `fs=2048`, `n_channels=28`, `electrode_layout="grabmyo_forearm16_wrist12"`,
`recording_device="OT Bioelettronica EMGUSB2+"`, `electrode_type="ambu_pregelled_monopolar"`,
`subject_type="healthy"`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name` = gesture name (e.g. "Hand Open", "Rest"); `repetition` = 1..7.
`trial_key` = `S01/sess1/g10/t1` (subject / session / **native gesture 1..17** / trial).
**Normalization:** `Normalizer` `mode="global"` for cross-subject/LOSO; `per_session` for
cross-day work (design doc §5.5).

## 5. Validation + integrity
- **SHA256 verified against PhysioNet's published SHA256SUMS.txt: every present file is
  bit-perfect (30,705 OK, 0 mismatches).**
- subjects 1..43 ✓; sessions {1,2,3} ✓; labels 0..16 (17 incl rest=0) ✓; trial_keys unique ✓;
  multi-session → cross-day available ✓; global normalizer train-only ✓ (validator PASS).
- **RESOLVED — Session3/participant9 incompleteness (2026-06-16).** The local copy had
  originally been missing 10 files (8 broken trials: g4t3, g4t6, g4t7, g5t2, g6t2, g12t3,
  g12t5, g16t7). Participant9/session3 was re-downloaded; the 8 trials' 16 files were
  **SHA256-verified against the official SHA256SUMS.txt (all match)** and copied into
  `Session3/session3_participant9/`. The folder now has all 238 files (119 complete pairs),
  global count = 15,351 .dat / 15,351 .hea, and the ingest is the **full 15,351 trials,
  0 skipped**. (The temporary download folder `Session 3 newly downloaded set/` held browser
  `" (1)"` duplicates; only the SHA-verified canonical copies were used.)

## 6. What this enables / limits
- **Enables:** multi-day **cross-day** HGR (days 1/8/29); **forearm-vs-wrist** electrode-site
  studies (channels 0–15 vs 16–27); large 43-subject cross-user / calibration-free models;
  sEMG **biometrics** (the dataset's second purpose); electrode-shift-invariant studies (no
  marks left between sessions → natural shift).
- **Limits:** healthy only; static held gestures (no transitions — see `grabmyo_flow_dynamic`);
  S3/p9 has 8 fewer trials. Label taxonomy (17 GrabMyo gestures) ≠ NinaPro/EMAHA/FORS/SeNic →
  cross-dataset supervised use needs an explicit label map (design doc §5.7). 17 GrabMyo
  gestures are a near-superset basis comparable to the `grabmyo_flow_static` 12-ch wrist set.

## 7. How to re-run
```
python -m semg.adapters.grabmyo \
  --raw "E:/sEMG Research Enhanced/semg-datasets/grabmyo/1.1.0" \
  --out "E:/sEMG Research Enhanced/data/L1/grabmyo"
```
Reads ~30k WFDB files via `_wfdb_read.py`; ~17 GB h5; a few minutes.

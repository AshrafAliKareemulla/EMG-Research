# STATE — GrabMyo-Flow STATIC extension (L1 canonical dataset)

> Read before touching `grabmyo_flow_static`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> PhysioNet: https://physionet.org/content/grabmyo-flow/1.0.0/ (folder `static_extension_WFDB`).
> No dedicated paper; relies on the flow `readme.txt` + the original GrabMyo paper
> (`mineru_output/GrabMyo.md`). Sibling sets: `grabmyo` (original 28ch) and
> `grabmyo_flow_dynamic` (transitions).

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~3.49 GB) + `manifest.parquet` (7,140 trials) by
`semg/adapters/grabmyo_flow_static.py`. Validator → PASS. SHA256-verified intact.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **trial**, shape `(12, 10240)` `float32` (mV), `lzf`,
  chunked `(12, 2048)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — 7,140 rows.

Raw (L0): `grabmyo-flow/1.0.0/static_extension_WFDB/session{s}/session{s}_participant{p}/
...gesture{k}_trial{l}.{dat,hea}`, p ∈ 44..63. Read via `semg/adapters/_wfdb_read.py`.

## 2. What it is
- **20 NEW healthy subjects** (raw folders **#44–63**) — the extension that grows GrabMyo's
  subject pool from 43 to 63. SAME static protocol as the original.
- **Wrist-only: 12 monopolar EMG channels (W1–W12)** (no forearm). EMGUSB2+, 2048 Hz, **5 s =
  10240 samples**. WFDB `.dat` = 16 cols = 12 wrist + 4 unused U1–U4 (dropped).
- **17 gestures** (16 + REST), **7 trials**, **3 sessions** (multi-day, cross-day).

## 3. Decisions — and WHY (user-confirmed)
1. **Separate 12ch L1 dataset** (can't merge with the 28ch original).
2. **Subject renumbered 1..20** (`subject = native_participant − 43`, i.e. 44→1 … 63→20) to
   match the official `subject-info.csv` IDs AND so the **same person carries the same
   subject id across `grabmyo_flow_static` and `grabmyo_flow_dynamic`** (subject *j* in both
   = the same flow-study participant). `native_subject` keeps the raw folder ("participant44").
3. **Labels: REST (native 17) → 0**; gestures 1–16 keep 1–16. `native_label` = 1..17.
   Gesture names identical to the original (imported from `grabmyo.GESTURE_NAMES`).
4. **`session` = day 1/2/3** → cross-day. Store native (12 wrist ch, 2048 Hz, mV);
   `is_envelope=False`.
5. Wrist channel order W1..W12 matches the original's wrist channels → directly comparable.

## 4. Manifest schema (columns)
Same as `grabmyo` (REQUIRED 18 + `native_label`, `native_subject`).
Values differ: `n_channels=12`, `electrode_layout="grabmyo_wrist12"`. `native_subject` =
"participant44".."participant63". `trial_key` = `S01/sess1/g10/t1` (subject 1..20 / session /
native gesture 1..17 / trial). **Normalization:** `global` for cross-subject; `per_session`
for cross-day.

## 5. Validation + integrity
- **SHA256 verified (part of the flow SHA256SUMS.txt: 21,497/21,497 files OK, 0 mismatches).**
- subjects 1..20 ✓; sessions {1,2,3} ✓; labels 0..16 (17) ✓; trial_keys unique ✓;
  total **7,140 = 20×3×17×7** (0 skipped) ✓; multi-session cross-day ✓; normalizer train-only ✓.

## 6. What this enables / limits
- **Enables:** extends GrabMyo to 63 subjects for wrist-based HGR/biometrics; multi-day
  cross-day; pairs with `grabmyo_flow_dynamic` (same people) for **static→dynamic** within-
  person studies; wrist channels comparable to the original's W1–W12.
- **Limits:** wrist-only (no forearm); healthy; static held gestures. Subject ids 1..20 here
  are flow-study people, **distinct from** `grabmyo` subjects 1..20 (different persons; each
  L1 set namespaces subjects by `dataset_id`).

## 7. How to re-run
```
python -m semg.adapters.grabmyo_flow_static \
  --raw "E:/sEMG Research Enhanced/semg-datasets/grabmyo-flow/1.0.0/static_extension_WFDB" \
  --out "E:/sEMG Research Enhanced/data/L1/grabmyo_flow_static"
```

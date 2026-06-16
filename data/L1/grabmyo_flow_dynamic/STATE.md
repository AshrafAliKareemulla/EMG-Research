# STATE — GrabMyo-Flow DYNAMIC (L1 canonical dataset)

> Read before touching `grabmyo_flow_dynamic`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> PhysioNet: https://physionet.org/content/grabmyo-flow/1.0.0/ (folder `dynamic_WFDB`).
> No dedicated paper; relies on the flow `readme.txt` + `MotionSeq.txt`. Sibling sets:
> `grabmyo` (original 28ch) and `grabmyo_flow_static` (12ch static extension).

## 0. STATUS: INGESTED & validated on 2026-06-16 (Windows box)
`signals.h5` (~3.43 GB) + `manifest.parquet` (3,600 trials) by
`semg/adapters/grabmyo_flow_dynamic.py`. Validator → PASS. SHA256-verified intact.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **trial**, shape `(12, 20480)` `float32` (mV), `lzf`,
  chunked `(12, 2048)`. Attrs: `fs`, `label`, `subject`.
- `manifest.parquet` — 3,600 rows.

Raw (L0): `grabmyo-flow/1.0.0/dynamic_WFDB/session{s}/session{s}_participant{p}/
...gesture{k}_trial{l}.{dat,hea}`, p ∈ 1..20, k ∈ 1..30, l ∈ 1..2. Read via `_wfdb_read.py`.

## 2. What it is — DYNAMIC gesture transitions
- **20 healthy subjects** (raw folders **#1–20**) — the SAME 20 people as
  `grabmyo_flow_static` #44–63 (confirmed by `subject-info.csv`, which lists both Static and
  Dynamic sessions per participant 1..20).
- **30 dynamic gesture TRANSITIONS** (e.g. "Lateral Prehension → Wrist Flexion"), **2 trials**
  each, **3 sessions** (cross-day). **NO rest class.**
- **Wrist-only: 12 EMG channels (W1–W12)**. 2048 Hz, **20480 samples = 10 s** (double the
  static 5 s — each clip spans a transition between two static gestures). WFDB `.dat` = 16
  cols = 12 wrist + 4 unused U (dropped). Physical mV.

## 3. Decisions — and WHY (user-confirmed)
1. **Separate L1 dataset** (transitions, 30 classes, 2 trials, 10 s — distinct from the
   static sets).
2. **Subject = raw folder participant 1..20** (already 1-based). This id equals the
   `grabmyo_flow_static` subject for the **same person** (static subject = native_static − 43;
   both resolve to the flow-study participant 1..20). `native_subject` = "participant1"…
3. **Labels 1..30 (no rest)** — the 30 transitions; `native_label` = 1..30; `label_name` =
   transition string from `MotionSeq.txt`.
4. **`session` = day 1/2/3** → cross-day. Store native (12 wrist ch, 2048 Hz, 20480 samples,
   mV); `is_envelope=False`.

## 4. Manifest schema (columns)
Same shape as the other GrabMyo sets (REQUIRED 18 + `native_label`, `native_subject`).
Values: `n_channels=12`, `electrode_layout="grabmyo_wrist12"`, `label` 1..30,
`label_name` = e.g. "Lateral Prehension to Wrist Flexion", `repetition` 1..2.
`trial_key` = `S01/sess1/g10/t1` (subject / session / transition 1..30 / trial).
**Normalization:** `global` for cross-subject; `per_session` for cross-day.

## 5. Validation + integrity
- **SHA256 verified (flow SHA256SUMS.txt: 21,497/21,497 OK, 0 mismatches).**
- subjects 1..20 ✓; sessions {1,2,3} ✓; labels 1..30, no rest ✓; `n_samples`=20480 ✓;
  trial_keys unique ✓; total **3,600 = 20×3×30×2** (0 skipped) ✓; multi-session cross-day ✓;
  normalizer train-only ✓.

## 6. What this enables / limits
- **Enables:** realistic **gesture-transition / co-articulation** testing (the unique value of
  the flow set — most datasets only have held-static gestures); cross-day; pairs with
  `grabmyo_flow_static` (same people) to **train on static, test on dynamic transitions**.
- **Limits:** only 2 trials per transition; wrist-only; healthy; the 30 transition classes are
  a bespoke label space (no rest), not comparable to any static gesture taxonomy without an
  explicit transition→endpoint mapping. **No onset/transition-midpoint markers** → the 10 s
  clip contains the full transition; segmenting the "from"/"to" phases is a downstream choice.

## 7. How to re-run
```
python -m semg.adapters.grabmyo_flow_dynamic \
  --raw "E:/sEMG Research Enhanced/semg-datasets/grabmyo-flow/1.0.0/dynamic_WFDB" \
  --out "E:/sEMG Research Enhanced/data/L1/grabmyo_flow_dynamic"
```

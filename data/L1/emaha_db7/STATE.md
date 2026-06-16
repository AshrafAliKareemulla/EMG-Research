# STATE — EMAHA-DB7 (L1 canonical dataset)

> Read before touching `emaha_db7`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> Paper: Sreenivas, Turlapaty, Naidu, Sagar, "Impact of Activity Pace and Arm Position on
> Classification of ADLs" (EMBC) — `7_EMBC/mineru_output/ieee_10782913.md`. Kaggle DOI
> 10.34740/KAGGLE/DSV/8270742. Companions: EMAHA-DB1/DB4/DB5.

## 0. STATUS: INGESTED & validated on 2026-06-17 (Windows box)
`signals.h5` + `manifest.parquet` (5,981 valid trials) by `semg/adapters/emaha_db7.py`.
`scripts/validate_l1.py --datasets emaha_db7` → PASS. Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **valid trial**, shape `(4, T)` `float32` (µV), `lzf`,
  chunked `(4, 4000)`. Attrs: `fs`(4000), `label`, `subject`.
- `manifest.parquet` — one row per valid trial (5,981 rows).

Raw (L0): `…/archive (1)/Sub-{01..10}/Sub-{01..10}/{45,90,135}.mat` (MATLAB v5; **top var name
varies** = `record_<datetime>_Frequency_Honors_<angle>`; read the single non-`__` key). The
**arm position is the FILE** (by angle). Same Noraxon-struct format as DB4/DB5.

## 2. What EMAHA-DB7 is (the raw dataset)
- **10 healthy right-handed subjects**, **10 ADL activities**.
- Each activity at **2 paces (Slow/Fast)** × **3 arm positions (45°/90°/135°)** × 10 trials.
- 3 files per subject = the 3 arm-position angles; **200 movements per file** = 10 activities
  × 2 paces × 10 trials. 30 files total.
- **4-channel Noraxon Ultium, 4000 Hz (uniform), µV**, phases rest → action → release (markers).

## 3. ⚠ FLAGGED INCONSISTENCIES (all PROVEN by full 30-file scans)
**Integrity:** all 30 files load clean; **0 NaN, 0 all-zero** across all 5,981 valid trials.
100% internal consistency. (Single Kaggle source, no checksum to compare; no second copy.)

1. **Channel muscles ≠ paper.** Data — **identical across all 30 files** — `ANT.DELTOID,
   BICEPS BR., FLEX.CARP.R, BRACHIORAD.` (= EMAHA-DB4's first 4, a proximal-arm montage). Paper
   says Brachioradialis/EPL/ADM/FDI → overlap 1/4, **not a reordering**. The IIIT papers
   repeatedly mis-state the channel table; the **DATA labels are authoritative.**
2. **Arm-position angle = 45/90/135° (validated), paper says P1=0°.** Two proofs:
   (a) the angle is **intrinsic** — filename angle == internal record-key angle in **30/30
   files**; (b) **physiological**: rest-phase ANT.DELTOID (shoulder) amplitude swings ~5×
   across the 3 files (45°→33.6, 90°→7.1, 135°→6.9 µV) while forearm muscles stay flatter —
   a shoulder-specific gravity signature proving 3 genuinely distinct arm elevations. The
   paper's "P1 = 0°" is a documentation error. Stored as `arm_angle_deg` (45/90/135,
   authoritative) + `arm_position` (p1/p2/p3 mapped 45→p1, 90→p2, 135→p3).
3. **Pace named Slow/Fast** in data (paper: Normal/Fast). Stored as `pace` ∈ {slow, fast}.
4. **244 spurious movements** (no action+release markers) → EXCLUDED (criterion: must have
   both `action` and `release`). 
5. **Valid total 5,981 ≠ clean 6,000** (paper). A few (activity,pace) groups have 9 valid
   trials, not 10. Activity-name differences from paper (Typing↔Typing Phone, Rubbered_Exercise
   ↔Rubber Band Finger Extension, etc.).

**Cleaner than DB5:** uniform 4000 Hz (no mixed fs), uniform 4 channels, tight ~200 valid/file.

## 4. Decisions — and WHY (user-confirmed)
1. Keep ALL valid trials (action+release present); exclude the 244 spurious. `repetition`
   1..N per (arm-position-file, pace, activity).
2. **`arm_position` (p1/p2/p3) + `arm_angle_deg` (45/90/135)** — both; angle is authoritative.
   `pace` (slow/fast) column. `session`=1 (single session; arm/pace are conditions, not days).
3. **Labels 1–10** (activities, paper Table I order); 0 reserved (no rest-class trials). Store
   FULL clip + `action_time`/`release_time` (s) markers for action-segment extraction (paper:
   action + 2 s release).
4. DATA channel labels (proximal arm montage). Store native (4 ch, 4000 Hz, µV);
   `is_envelope=False`; HP20+notch50, Z-norm and 1 s windowing are load-time choices.

## 5. Manifest schema (columns)
REQUIRED 18 (design doc §8.1) + dataset-specific `native_activity, arm_position, arm_angle_deg,
pace, action_time, release_time`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_activity,
arm_position, arm_angle_deg, pace, action_time, release_time, fs, n_channels, n_samples,
electrode_layout, recording_device, electrode_type, subject_type, is_envelope,
adapter_version, ingested_utc, domain`.
Values: `fs=4000`, `n_channels=4`, `electrode_layout="noraxon_4_arm"`,
`recording_device="Noraxon Ultium wireless"`, `electrode_type="Ag/AgCl_dual"`,
`subject_type="healthy"`, `session=1`, `is_envelope=False`. `label_name` = readable activity;
`repetition` 1..N. `trial_key` = `S01/p1/slow/writing/r1`
(subject / arm-position / pace / activity / trial). **Normalization:** `mode="global"` for
cross-subject/LOSO.

## 6. Validation (ACTUAL ingest, 2026-06-17)
- 10 subjects ✓; arm_position {p1,p2,p3} / angles {45,90,135} ✓; pace {slow,fast} ✓;
  labels 1..10, rest_present=False ✓; **total 5,981 valid** (=proven scan; 244 spurious
  skipped) ✓; trial_keys unique ✓; channel labels identical across all 30 files ✓;
  uniform fs=4000 ✓; 0 NaN/all-zero ✓; LOSO disjoint ✓; global normalizer train-only ✓.
  `scripts/validate_l1.py --datasets emaha_db7` → PASS.

## 7. What this enables / limits
- **Enables:** **pace invariance** (train Slow / test Fast — the paper's headline; query
  `pace`), **arm-position invariance** (leave-one-arm-out over 45/90/135 via `arm_position`),
  LOSO, principled action-segment extraction via markers. Complements DB4 (arm position ×
  posture, 5ch) and DB5 (fine-ADL × posture/hand, 5ch). The arm-position factor is
  signal-validated (deltoid gravity gradient).
- **Limits:** single session → no cross-day; healthy only. **4-channel proximal montage**
  (deltoid/biceps/FCR/brachiorad) — differs from DB1 (5ch forearm), DB4 (5ch, = DB7's 4 + FCU),
  DB5 (5ch incl. pectoralis) → the EMAHA family is NOT channel-identical (DB7 = DB4's first 4).
  Variable trial counts; 10-activity ADL taxonomy ≠ other datasets → cross-dataset supervised
  use needs an explicit label map (design doc §5.7).

## 8. How to re-run
```
python -m semg.adapters.emaha_db7 \
  --raw "C:/Users/ashra/Downloads/archive (1)" \
  --out "E:/sEMG Research Enhanced/data/L1/emaha_db7"
```
Loads 30 v5 .mat (one at a time); a few minutes.

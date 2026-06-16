# STATE — EMAHA-DB4 (L1 canonical dataset)

> Read before touching `emaha_db4`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> Paper: Sagar, Turlapaty, Naidu, "Impact of Measurement Conditions on Classification of ADL
> using Surface EMG signals" — `mineru_output/EMAHA DB4.md`. Harvard Dataverse:
> 10.7910/DVN/IFPNRK. Companion: see `data/L1/emaha_db1/STATE.md` (the primary EMAHA set).

## 0. STATUS: INGESTED & validated on 2026-06-17 (Windows box)
`signals.h5` + `manifest.parquet` (4,800 trials) by `semg/adapters/emaha_db4.py`.
`scripts/validate_l1.py --datasets emaha_db4` → PASS. Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **movement/trial**, shape `(5, T)` `float32` (µV),
  `lzf`, chunked `(5, 4000)` = (channels, ~1 s). Attrs: `fs`, `label`, `subject`. T varies.
- `manifest.parquet` — one row per movement (4,800 rows).

Raw (L0): `…/EMAHA DB4/DB4 Matlab Files/Dataset/Sub{01..10}/{Sit,Stand,Walk}.mat`
(MATLAB v5; read with scipy). Each `.mat` = struct `Data` = `{info, movements[160]}`.

## 2. What EMAHA-DB4 is (the raw dataset)
- **10 healthy right-handed subjects** (Sub01–Sub10).
- **8 ADL activities**, each in **4 arm positions (p1–p4) × 3 body postures (sit/stand/walk)
  × 5 trials** → **160 movements per posture file**; 10 × 3 × 160 = **4,800 trials**.
- **5 channels, Noraxon Ultium wireless, 4000 Hz, units µV.** Each movement has 3 phase
  markers: **rest (t=0) → action (tₐ) → release (t_r)** in seconds.
- Each movement struct: `name`="<Activity>_p<k>", `time_begin/time_end`, `markers[3]`,
  `sources.signals.signal_1..signal_5` (fields: name, frequency, units, count, data).

### Channel muscles (from the DATA — authoritative; signal_1..signal_5 order)
1. **ANT.DELTOID** (Anterior Deltoid) · 2. **BICEPS BR.** (Biceps Brachii) ·
3. **FLEX.CARP.R** (Flexor Carpi Radialis) · 4. **BRACHIORAD.** (Brachioradialis) ·
5. **FLEX.CAPR.U** (Flexor Carpi Ulnaris — the raw label "CAPR" is a Noraxon typo for CARP).

## 3. ⚠ FLAGGED INCONSISTENCIES (verified; data is authoritative)
1. **Channel muscles in the DATA ≠ paper Table II.** Paper Table II lists BR, EPL, ADM, FDI,
   EPB (small thumb/finger muscles). **Rigorous cross-check (all 30 files):** the channel
   label tuple is *identical across every subject × posture* and is
   `(ANT.DELTOID, BICEPS BR., FLEX.CARP.R, BRACHIORAD., FLEX.CAPR.U)`. **Set-overlap with the
   paper list = at most 1/5 (Brachioradialis only)** → it is NOT a reordering (a permutation
   would need 5/5). The paper Table II is an error; the actual DB4 montage is proximal
   (shoulder + upper-arm + forearm), which is sensible for an arm-position study. **We store
   the data muscles.** (Proof: `_wfdb_read`-style scan in the session log / btbv7fj5w.)
2. **Activity names:** data uses `PushingTable`/`PullingTable`; paper Table I says "Pushing a
   Door"/"Pulling a Door". We keep the data stem in `native_activity`; `label_name` is the
   readable paper-style name.
3. **4000 Hz** sampling (DB1 was 2000 Hz). Stored native.

## 4. Decisions — and WHY (user-confirmed)
1. **`arm_position` (p1–p4) + `body_posture` (sit/stand/walk) as explicit columns;
   `session`=1.** DB4 is single-session — postures/positions are *measurement conditions*,
   not days (same approach as FORS orientation; rejected mapping posture→session, which would
   mislabel it as cross-day). The paper's leave-one-arm-out / leave-one-posture-out
   experiments are manifest queries on these columns.
2. **Labels 1–8 (the 8 activities); 0 reserved** (no separate rest-class trials). The full
   clip is stored natively + **`action_time`/`release_time`** (seconds, from the markers) so
   downstream can extract the action segment (paper uses action → release+1 s) or treat the
   rest phase. Rest is a *phase*, recoverable via the markers — never fabricated into storage.
3. Activity→label map (paper Table I order): 1 Lifting bottle, 2 PushingTable, 3 PullingTable,
   4 PhoneToEar, 5 Waving, 6 Zipper, 7 PuttingGlasses, 8 TVRemote.
4. Store native (5 ch, 4000 Hz, µV); `is_envelope=False`; the paper's 20 Hz HP + 50 Hz
   band-stop and 250 ms windowing are **load-time** choices, not baked in. HDF5 `(5,~1s)` lzf.

## 5. Manifest schema (columns)
REQUIRED 18 (design doc §8.1) + dataset-specific `native_activity, arm_position, body_posture,
action_time, release_time`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_activity,
arm_position, body_posture, action_time, release_time, fs, n_channels, n_samples,
electrode_layout, recording_device, electrode_type, subject_type, is_envelope,
adapter_version, ingested_utc, domain`.
Values: `fs=4000`, `n_channels=5`, `electrode_layout="noraxon_5_arm"`,
`recording_device="Noraxon Ultium wireless"`, `electrode_type="Ag/AgCl_dual"`,
`subject_type="healthy"`, `session=1`, `is_envelope=False`, `adapter_version="1.0.0"`.
`label_name` = readable activity; `repetition` 1..5.
`trial_key` = `S01/sit/liftingbottle/p1/r1` (subject / posture / activity / arm-pos / trial).
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO claim (design §5.5).

## 6. Validation (ACTUAL ingest, 2026-06-17)
- 10 subjects ✓; body_posture {sit,stand,walk} ✓; arm_position {p1..p4} ✓; labels 1..8,
  rest_present=False ✓; total **4,800 = 10×3×160** ✓; per (subject,posture)=160 ✓;
  per (subject,posture,activity,arm)=5 ✓; trial_keys unique ✓; all 30 files = 5ch @4000 Hz,
  3 markers ✓; channel labels identical across all 30 files ✓; global normalizer train-only ✓.
- `scripts/validate_l1.py --datasets emaha_db4` → PASS (full suite all datasets PASS).

## 7. What this enables / limits
- **Enables (the headline):** **arm-position invariance** (leave-one-arm-out: train p1–p3,
  test p4 — query `arm_position`) and **body-posture invariance** (leave-one-posture-out via
  `body_posture`) — the paper's core experiments. Also LOSO, within-subject, and (unlike
  DB1) **explicit phase markers** for principled action-segment extraction. 5-ch ADL set
  complements EMAHA-DB1 (also Noraxon 5-ch, but forearm-only and different activities).
- **Limits:** single session → no cross-day; healthy only. **Different 5-muscle montage than
  DB1** (proximal incl. deltoid/biceps) → DB1↔DB4 are NOT channel-identical despite both
  being "Noraxon 5-ch". 8-activity ADL taxonomy ≠ DB1's 22 / NinaPro / FORS / SeNic →
  cross-dataset supervised use needs an explicit label map (design doc §5.7).

## 8. How to re-run
```
python -m semg.adapters.emaha_db4 \
  --raw "C:/Users/ashra/OneDrive/Desktop/EMAHA Data/EMAHA DB4" \
  --out "E:/sEMG Research Enhanced/data/L1/emaha_db4"
```
Loads 30 × ~180 MB v5 .mat (one at a time); a few minutes.

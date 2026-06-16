# STATE — EMAHA-DB5 (L1 canonical dataset)

> Read before touching `emaha_db5`. Design doc: `semg-datasets/semg-dataset-setup.md`.
> Paper: Naidu, Turlapaty, Sagar, "Classification of Fine-ADL using sEMG signals under
> Different Measurement Conditions" — `mineru_output/EMAHA DB5.md`. Companions:
> `emaha_db1` (primary ADL) and `emaha_db4` (ADL × arm-position/posture).

## 0. STATUS: INGESTED & validated on 2026-06-17 (Windows box)
`signals.h5` + `manifest.parquet` (5,251 valid trials) by `semg/adapters/emaha_db5.py`.
`scripts/validate_l1.py --datasets emaha_db5` → PASS. Ready to copy to the GPU box.

## 1. What this folder is
- `signals.h5` — one HDF5 dataset per **valid movement/trial**, shape `(5, T)` `float32`
  (µV), `lzf`, chunked `(5, fs)`. Attrs: `fs` (2000 **or** 4000), `label`, `subject`.
- `manifest.parquet` — one row per valid trial (5,251 rows).

Raw (L0): `…/EMAHA-DB5-ADL-HVAR2/Sub{01..10}/<datetime>_GESTURES_HONOR_<POSTURE>[_Actual].mat`
(MATLAB v5; **top variable name varies per file** = `record_<datetime>_…`; read the single
non-`__` key). Same Noraxon-struct format as DB4.

## 2. What EMAHA-DB5 is (the raw dataset)
- **10 healthy subjects** (7M/3F; 8 right- / 2 left-handed). **Fine-ADL** focus.
- **10 fine-ADL activities × 5 body postures × 2 hand positions × ~5 trials.**
- 5 files per subject = the **5 body postures** (from filename): Sit, Sit-to-Stand, Standing,
  Folded Legs, Folded Knees → normalised `sit / sit_to_stand / stand / folded_legs /
  folded_knees`. 50 files total.
- **5-channel Noraxon Ultium**, µV, phases **rest → action → release** (3 markers).

## 3. ⚠ FLAGGED INCONSISTENCIES (all PROVEN by full-data scans; not corruption)
**Integrity proof:** the OneDrive copy and the Kaggle release
(kaggle.com/datasets/anishturlapaty/emaha-db5) are **byte-identical** (all 50 files same size;
SHA256 of the anomalous Sub01/SIT & Sub08/SIT identical). All 50 files load cleanly, 0 NaN,
0 all-zero, EMG-like µV ranges. So the flags below are intrinsic to the **official release**
(+ paper documentation errors), NOT a bad download. (EMAHA ships no checksum file.)

1. **Channel muscles ≠ paper Table I.** Data — **identical across all 50 files** —
   `ABDUCT.POL., PECT.MAJOR, BICEPS BR., FLEX.CARP.R, FLEX.CAPR.U` (signal_1..5). Paper Table I
   says Abductor digiti minimi / Extensor pollicis brevis / First dorsal interosseous /
   Abductor pollicis brevis / Brachioradialis → set-overlap ≤1, **not a reordering**. Data
   includes **Pectoralis Major** (chest/postural), sensible for a body-posture study. **We
   store the DATA labels (authoritative).** (`FLEX.CAPR.U` = Noraxon typo for FLEX.CARP.U =
   Flexor Carpi Ulnaris.) Caveat: these could be un-renamed Noraxon defaults — true physical
   placement vs paper is not independently verifiable, so treat muscle names as uncertain.
2. **Mixed sampling rate (per-file, no intra-file mix):** **4 files @2000 Hz** (Sub01
   SIT/SIT_TO_STAND/STANDING/FOLDED_LEGS, recorded Jan 26) + **46 @4000 Hz** (Sub01 FOLDED_Knees
   + all Sub02–10). The lab switched 2000→4000 after Sub01's first session. Stored **per-trial
   in `fs`** → resample at load time for any cross-rate work.
3. **Variable trial counts (NOT the paper's clean 5):** most (activity,position) groups have 5
   trials, but Sub01's SIT/STANDING/FOLDED_LEGS have ~10 and SIT_TO_STAND fewer. We **keep ALL
   valid trials** (number `repetition` 1..N per group). Paper claims 5 → 5000; data has **5,251
   valid**.
4. **223 spurious movements EXCLUDED:** they carry only a `rest` marker (no action/release),
   <1 s (e.g. 0.16 s / count 320) — aborted/empty manual segmentations. A movement is ingested
   only if it has **both `action` and `release`** markers.
5. **Hand positions labeled `p1`, `p3`** (paper says P1/P2). **Activity names differ** from
   the paper (data: SlidingGesture, UsingScalpel, HoldingASpoon, FlippingBottleCapOpen…).

## 4. Decisions — and WHY (user-confirmed)
1. **Keep ALL valid trials** (action+release present); exclude the 223 spurious. Faithful L1
   ("store everything real"); downstream can cap to 5/group to match the paper.
2. **`body_posture` + `hand_position` columns; `session`=1** (single session — postures/
   positions are measurement conditions, not days; like FORS/DB4). Leave-one-posture-out /
   leave-one-hand-position-out are manifest queries.
3. **Labels 1–10** (activities, paper Table II order); 0 reserved (no rest-class trials).
   Store FULL clip + `action_time`/`release_time` (s) markers for action-segment extraction
   (paper: action + 2 s of release). **Per-trial `fs`** (2000/4000). DATA channel labels.
4. Activity→label map (paper Table II order): 1 SlidingGesture(Swiping), 2 ZoomIn, 3 ZoomOut,
   4 PressingAButton, 5 FlippingASwitch, 6 UsingScalpel(Using a Knife), 7 HoldingASpoon(Eating
   with Spoon), 8 FlippingBottleCapOpen, 9 Writing, 10 Scissors.
5. Store native (5 ch, µV, native fs); `is_envelope=False`; HP20+notch50 & 250 ms windowing
   are load-time choices.

## 5. Manifest schema (columns)
REQUIRED 18 (design doc §8.1) + dataset-specific `native_activity, hand_position, body_posture,
action_time, release_time`:
`trial_key, dataset_id, subject, session, repetition, label, label_name, native_activity,
hand_position, body_posture, action_time, release_time, fs, n_channels, n_samples,
electrode_layout, recording_device, electrode_type, subject_type, is_envelope,
adapter_version, ingested_utc, domain`.
Values: `n_channels=5`, `electrode_layout="noraxon_5_db5"`, `recording_device="Noraxon Ultium
wireless"`, `electrode_type="Ag/AgCl_dual"`, `subject_type="healthy"`, `session=1`,
`is_envelope=False`; **`fs` ∈ {2000, 4000} (per trial)**. `label_name` = readable activity;
`repetition` 1..N (variable). `trial_key` = `S01/sit/slidinggesture/p1/r1`
(subject / posture / activity / hand-pos / trial). **Normalization:** `mode="global"` for
cross-subject; note per-trial fs differences → resample before pooling.

## 6. Validation (ACTUAL ingest, 2026-06-17)
- 10 subjects ✓; body_posture {sit,sit_to_stand,stand,folded_legs,folded_knees} ✓;
  hand_position {p1,p3} ✓; labels 1..10, rest_present=False ✓; **total 5,251 valid**
  (=proven scan count; 223 spurious skipped) ✓; trial_keys unique ✓; channel labels identical
  across all 50 files ✓; per-trial fs ∈ {2000,4000} ✓; LOSO disjoint ✓; global normalizer
  train-only ✓. `scripts/validate_l1.py --datasets emaha_db5` → PASS.

## 7. What this enables / limits
- **Enables:** **body-posture invariance** (leave-one-posture-out over 5 postures, incl. the
  dynamic Sit-to-Stand) and **hand-position invariance** (p1↔p3) for *fine* finger/wrist ADLs;
  LOSO; principled action-segment extraction via markers. Complements EMAHA-DB1 (coarse ADL,
  forearm) and DB4 (ADL × arm position, proximal montage) — a 3-dataset EMAHA family.
- **Limits:** single session → no cross-day; healthy only. **Mixed fs** (must resample to
  combine 2000+4000 trials, or even across this dataset). **Channel montage differs from DB1
  and DB4** (DB5 = abd.pol/pect.major/biceps/FCR/FCU) → the three EMAHA sets are NOT
  channel-identical. Variable trial counts; muscle labels uncertain vs paper. 10-activity
  fine-ADL taxonomy ≠ other datasets → cross-dataset supervised use needs an explicit label
  map (design doc §5.7).

## 8. How to re-run
```
python -m semg.adapters.emaha_db5 \
  --raw "C:/Users/ashra/OneDrive/Desktop/EMAHA Ninapro EMG Data/EMAHA Dataset/EMAHA-DB5-ADL-HVAR2" \
  --out "E:/sEMG Research Enhanced/data/L1/emaha_db5"
```
Loads 50 × ~80–243 MB v5 .mat (one at a time); a few minutes. (The Kaggle copy under
`C:/Users/ashra/Downloads/archive/Sub{NN}-*/Sub{NN}/` is byte-identical; either works.)

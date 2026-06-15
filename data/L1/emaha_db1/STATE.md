# STATE — EMAHA-DB1 (L1 canonical dataset)

> Read before touching `emaha_db1`. Records the raw data, decisions, conversion, and
> **why**. Design doc: `semg-datasets/semg-dataset-setup.md`. Paper:
> `literature-review-papers/dataset-papers/EMAHA DB1.pdf` (Karnam et al., IEEE TIM 2023).
> EMAHA-DB1 is the user's PRIMARY ADL dataset.

## 1. What this folder is
Canonical L1 form produced by `semg/adapters/emaha_db1.py`.
- `signals.h5` (~2.35 GB) — one HDF5 dataset per trial, shape `(5, T)` float32, lzf, chunks `(5, 2000)`. Attrs: fs, label, subject.
- `manifest.parquet` (15,736 trials) — one row per trial.

Raw source (L0): `C:\Users\ashra\OneDrive\Desktop\EMAHA Ninapro EMG Data\EMAHA Dataset\EMAHA-DB1-ADL-DATA\{daily_sub_wise_Train_split, daily_sub_wise_Test_split}\S###_{tr,tt}.mat` (MATLAB v7.3 / HDF5; read with h5py).
NOTE: an alternate folder `...\EMAHA Data\EMAHA DB1\{Train_CSV,Test_CSV}` holds the **identical** data as CSV. Either works; we used the .mat folder. The `DB1_*ms_Segmentation` folders (precomputed windowed features) are IGNORED — we keep the continuous signal and window at load time.

## 2. What EMAHA-DB1 is (raw dataset)
- **25 healthy subjects** (22M/3F, age 28±6; 24 right-handed, 1 left). Indian population.
- **5 channels**, **Noraxon Ultium**, Ag/AgCl, **2000 Hz**, 24-bit ADC. Channel→muscle: Ch1 Brachioradialis, Ch2 Flexor Carpi Radialis, Ch3 Flexor Carpi Ulnaris, Ch4 Biceps Brachii, Ch5 Abductor Pollicis Brevis.
- **22 activities A0–A21** (A0 = rest; A1–A21 real ADLs: coin toss, grasping objects, writing, typing, drinking, etc.), **10 reps** each. Grouped into **6 FAABOS categories G0–G5** (rest, no-object, pull/push, grasping, finger flex/ext, writing).
- Each rep = continuous clip (~8–20 s) with phases **rest → action → release**. Variable durations (Table III).
- Single session (no cross-day).

### Raw matrix = (10, N). Rows decoded (verified against data):
| Row | Name | Meaning / use |
|---|---|---|
| 0–4 | Chan_1..5 | sEMG, raw bipolar (signed, zero-mean) → stored |
| 5 | Marker | audio-cue phase: 0 rest / 1 action / 2 release |
| 6 | Label | activity id **1..21 == A1..A21** (A0/rest is NOT a Label value) |
| 7 | MSegment | refined 3 dB muscle-activity flag: 1 = inside real activation |
| 8 | Repition | 1..10; train={1,3,4,6,8,9,10}, test={2,5,7} |
| 9 | Trial | global counter 1..210 = (Label-1)*10 + Repition |

Verified `Label k = Ak`: Trial 1 (Label 1) = 16000 samples = 8 s = A1's total duration in Table III ✓.

## 3. Decisions — and WHY
1. **Labeling basis = `MSegment`** (refined muscle-activity), NOT `Marker` (audio cue). Default; switchable via `--label-basis marker`. **Why:** the paper's published baselines — 22-class 75.39%, FAABOS 83.21%, **LOSO 58.59% (the gap our research targets)** — were computed on MSegment labels. Using MSegment keeps our numbers directly comparable. Per-sample effective label = `Label` where `MSegment==1`, else `0` (rest).
2. **Segment into contiguous runs of constant (effective_label, Trial)** — each action burst and each rest gap = one stored trial. Same pattern as the NinaPro adapters. Result: subject 1 → exactly 210 action runs (one clean burst per rep).
3. **Rest = class 0** (= activity A0). Consistent with NinaPro.
4. **Ingest BOTH train and test** into one signals.h5; `orig_split` column preserves the paper's within-subject split (reps 1,3,4,6,8,9,10 vs 2,5,7) while still allowing LOSO across subjects.
5. **Store native** (5 ch, 2000 Hz, raw); `is_envelope=False`. Resample/filter at load time.
6. **Also store** `label_name` (Table I human names) + `faabos_group`/`faabos_name` (Table IV) — enables the FAABOS-category experiments central to the user's ADL focus, with no extra cost.

## 4. Manifest schema (columns)
`trial_key, dataset_id, subject, session, repetition, label, label_name, faabos_group, faabos_name, native_trial, orig_split, fs, n_channels, n_samples, electrode_layout, recording_device, electrode_type, subject_type, is_envelope, label_basis, adapter_version, ingested_utc, domain`.
Values: `fs=2000`, `n_channels=5`, `electrode_layout="noraxon_5_forearm"`, `recording_device="Noraxon Ultium wireless"`, `electrode_type="Ag/AgCl_wet_gel"`, `subject_type="healthy"`, `session=1`, `is_envelope=False`, `label_basis="msegment"`, `adapter_version="1.0.0"`. (device/electrode/version/timestamp cols added per external review — design doc §15.)
**Normalization:** `Normalizer` `mode="global"` for any cross-subject/LOSO claim (default); `per_subject` only for within-subject comparison to the paper. See design doc §5.5.
`trial_key`: action = `S001/train/A05/rep03/s<start>`; rest = `S001/train/rest/t<trial>/s<start>` (start sample makes keys unique).

## 5. Validation that passed
- 25 subjects, labels `0..21` (22 classes incl rest), FAABOS groups `0..5` ✓
- action trials = **5,250 = 25 × 210** ✓; rest trials = 10,486; total 15,736
- subject 1: 210 action runs, 21 distinct activities ✓; all trial_keys unique ✓
- orig_split: train 11,015 / test 4,721
- action-run length: min 78 (≈39 ms, e.g. coin toss) / median 8221 (≈4 s) / max 31798 samples
- **Smoke test (loader+LOSO+normalizer) PASSED locally**: win=200 ms → (5,400) windows, 546,804 total, batch (256,5,400), labels 0..21; LOSO subject-disjoint; normalizer fit on 24 train subjects only (test subject excluded). Report: `data/L1/smoke_test_report.txt`.

## 6. What this enables / limits
- **Enables:** 22-class ADL classification, 6-class FAABOS classification, LOSO (the headline gap), reproduction of the paper's within-subject split (via `orig_split`). Track-1 and Track-2 both.
- **Limits:** single session → no cross-day. Healthy only → no clinical transfer from DB1 alone.

## 7. Caveats / open
- **Preprocessing state of the stored channels is uncertain.** The paper applies notch 50 Hz → 1st-order Butterworth LP @500 Hz → wavelet (symlet-8) denoising. Whether the stored channel values are already filtered or fully raw is not documented; values show large spikes (Chan_3 to −11292), suggesting little/no aggressive filtering. We store as-is and treat preprocessing as a load-time choice. Flag if a downstream result looks off.
- A few very short action runs (<200 ms) are skipped by the 200 ms window index (648 short trials skipped in the smoke test); reduce window_ms to capture quick movements like coin toss if needed.

## 8. How to re-run
```
python -m semg.adapters.emaha_db1 \
  --raw "C:/Users/ashra/OneDrive/Desktop/EMAHA Ninapro EMG Data/EMAHA Dataset/EMAHA-DB1-ADL-DATA" \
  --out "E:/sEMG Research Enhanced/data/L1/emaha_db1"
# add --label-basis marker to relabel using the audio-cue phase instead.
```

## 9. Related EMAHA datasets present (not yet ingested)
Same parent folder also has `EMAHA-DB4-ADL-VAR1` and `EMAHA-DB5-ADL-HVAR2` (DB4/DB5 — force/orientation variation variants). Adapters TBD.

# CLAUDE.md — Contains detailed information reagrding user's background, Goals and Priorities of Machine Learning research in the field of EMG signals


## First: Read USER-REQUIREMENTS.md



Read `.claude/USER-REQUIREMENTS.md`. It tells you what are the user requirements and why this repository was started in first place. Only then read the area-specific files relevant to your current task.

## First focus on literature review part and follow the instructions in @.claude/literature-review/deep-literature-review-instructions.md properly. These instructions are the most important ones. Do exactly as stated by the user.



## Dataset infrastructure (read these before any dataset/ML/DL work)

Canonical data lives in `data/L1/<dataset>/` as `signals.h5` + `manifest.parquet`. To understand the setup, read in this order:

1. `semg-datasets/semg-dataset-setup.md` — the master design (storage, lazy loading, splits, normalization modes; **§15 = external-review fixes**).
2. `data/L1/<dataset>/STATE.md` — per-dataset rationale (what/how/why). Done: `ninapro_db1`, `ninapro_db2`, `emaha_db1` (primary). NinaPro DB3 = empty (amputee set, re-download).
3. Code: adapters `semg/adapters/`, loader `semg/data/`, splitter+`Normalizer` `semg/splits/`, smoke test `scripts/smoke_test.py`.

Key rules: store native (resample/filter at load time); LOSO/within-subject = manifest queries; **`Normalizer` `mode="global"` for any cross-subject claim**; one adapter per dataset (bump `adapter_version` on logic change).

## Five Rules

1. **Read before you act.** Check `STATE.md` before proposing work.
2. **Update state when done.** Update `STATE.md` + the relevant area status file at session end.
3. **Append to session log.** Add an entry to `.claude/sessions/SESSIONS.md` (never edit old entries).
4. Check if user has specified any specific questions or tasks in the current session and prioritize them over the general instructions and always take clairty from the user if anything is unclear or not stated.

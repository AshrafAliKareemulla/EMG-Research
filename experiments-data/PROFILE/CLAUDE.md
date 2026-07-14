# CLAUDE.md — the operating manual for PROFILE (the data-science track)

**This file governs every action taken in this folder.** If an instruction here conflicts with a
habit, a memory, or an older document, this file wins. Read it before touching anything.

**Scope, and it is absolute.** PROFILE is the **data-science track only**. It never reads, writes,
imports from, or depends on any other folder in the repository — not `experiments-dl/`, not the
Track-1 code, nothing. Deep learning belongs to a different track. If a question can only be
answered with a deep model, it is **out of scope here** and must be retired with a written reason
(as X3 was), not smuggled in.

---

## 0. The two machines, and the exact paths

There are **two** machines and they have different jobs. Never confuse them.

| | **Analysis machine** (Windows) | **Compute box** (Ubuntu, 20 cores) |
|---|---|---|
| PROFILE | `E:\sEMG Research Enhanced\experiments-data\PROFILE\` | `/home/honors/Ashraf_Ali_2025_Batch/sEMG-Research-Latest/experiments-data/PROFILE/` |
| L1 data | `E:\sEMG Research Enhanced\data\L1\` | `/home/honors/Ashraf_Ali_2025_Batch/data/L1` |
| Job | write code, read results, update docs, build the ledger | **run the experiments** |

**Paths are auto-discovered — never hardcode them and never set an env var.** `config.py` tries
`REPO_ROOT/data/L1` then `REPO_ROOT.parent/data/L1`; on the box the second one resolves to
`/home/honors/Ashraf_Ali_2025_Batch/data/L1`. `python -m cli.preflight` prints what it resolved and
exits non-zero if anything is missing. Run it first, always.

**Transfer is by AnyDesk file copy, not rsync/scp.** So instructions must be given as *folders to
drag*, never as shell commands the user cannot run.

### Setting the box up after a code change (do this every time code is synced)

1. **Copy → box `PROFILE/`, overwriting:** `dsprofile/ paper_experiments/ addons/ cli/ tsuite/
   tests/ notebooks/ docs/ CLAUDE.md STATE.md RESEARCH_PREVIEW.md requirements.txt`
2. **Delete from the box's `PROFILE/` root** any loose `*.py` (`exp_*.py`, `run_*.py`,
   `validate_results.py`, `preflight.py`, `invalidate_stale.py`, `floor_effect*.py`) and any old
   `*.md` other than the three live ones. They are stale duplicates of code that now lives in `cli/`
   and `addons/`, and **if left behind they will silently run the OLD logic.**
3. **Results layout on the box must mirror this repo:** the frozen evidence in
   `results/legacy_v1/`, new runs to `results/v2/`, and **`results/_feature_cache/` left exactly where
   it is** (≈20 GB; it is keyed by dataset+window+cap+normalize+decimate+seed, so moving it forces a
   full rebuild for nothing).

---

## 1. The four rules that everything else follows from

### Rule 1 — A name, once used, means one thing forever
A file, a folder, a results directory or an experiment tag, once it has produced a number, **may
never be repointed at different logic.** If the logic changes, the name changes.

- Changing an experiment's method → bump the tag: `t3_moment_ladder` → `t3_moment_ladder_v2`.
  The old results stay exactly where they are, under the old name, still valid, still quotable as
  "what the old method said."
- Never edit a file in `results/legacy_v1/`. It is frozen. New runs write to `results/v2/`.
- Never delete a result. Move it to a `_superseded_DO_NOT_QUOTE/` folder with a note saying what
  replaced it and why.

*Why:* on 2026-07-13 an audit found the project's own documents misquoting its own results in five
places, because thirty folders had been overwritten across four days and five code generations, and
nobody could tell which number came from which code. That must never happen again.

### Rule 2 — Ground truth before real data
An experiment may not run on a real dataset until its synthetic ground-truth checks pass. The gate
is enforced in code (`tsuite/run.py` refuses to proceed), not by good intentions.

A ground-truth check validates the **instrument**, and it needs **both** halves:
- **positive control** — on data where the effect is present *by construction*, the code must find it;
- **negative control** — on data where the effect is absent *by construction*, the code must find
  nothing. An experiment that always finds an effect is not an experiment.

*This is not ceremony.* When the T-suite was built, the gate failed 8 of 36 checks on the first run.
Every one was a flaw in the experiment's own reasoning (an offset placed along the discriminative
direction instead of a nuisance one; a "gain" corruption that also moved the mean; a "structureless"
null whose random class centres actually had structure). All eight were caught **before a single real
dataset was touched.**

### Rule 3 — Every novel claim is tested on all 14 datasets
One dataset is an anecdote. A claim enters the paper only with evidence across the panel, reported
per-dataset **and** pooled, with the datasets that disagree named out loud.

The only exception is a question that is *inherently* about one dataset (T2 asks why **senic**
reverses). Even then the experiment must run **control datasets** alongside, so that "it is about
senic" can be distinguished from "it is about our method, everywhere."

### Rule 4 — A null result is a result. Log it and move on
Every experiment **pre-registers its branches before it runs**: what a success means, what a failure
means, and what each implies for the paper. When a branch fires, we write it down — including the
disappointing one — and we move to the next experiment. We do not re-run an experiment with different
settings until it says something nicer. That is how the floor-effect claim got mis-stated as
"RESOLVED" for two days.

A well-written null ("cross-dataset transfer fails, and here is the alignment control proving it is
not simply that we forgot to normalise") is more publishable than a weak positive.

---

## 2. How to read an experiment

Every experiment file opens with the same four sections, in this order. If a file does not have
them, it is not finished.

1. **WHY THIS EXPERIMENT EXISTS** — the specific gap in the committed evidence. Not "it would be
   interesting", but "X9 tried this and failed, and here is the control it never ran."
2. **WHAT IT DOES** — the protocol, and the leakage discipline. Which data the model sees, which it
   never sees, and where the standardisation statistics come from.
3. **PRE-REGISTERED BRANCHES** — A/B/C/D. Decided before looking.
4. **GROUND TRUTH** — the positive and negative controls, and what each proves.

Then read `results/LEDGER.xlsx` for the numbers. **Never read numbers out of a markdown file** —
markdown drifts, the ledger is generated from the JSONs.

---

## 3. Where everything lives

```
PROFILE/
├── CLAUDE.md              this file — the protocol
├── STATE.md               the living state: what is done, what is running, what is next
├── RESEARCH_PREVIEW.md    the paper-facing view: every claim, its number, its evidence file
├── dsprofile/             core library (Phase-1/2 modules, blocks, figures) — STABLE, don't churn
├── paper_experiments/     the X-suite (X1-X15) + common.py = THE SHARED MATHS
├── addons/                add-on experiments A-E
├── tsuite/                the T-suite (T1-T8) — the 2026-07-13 experiments
├── cli/                   entrypoints: preflight, validate_results, build_ledger, run_*, floor_effect_x1
├── tests/                 247 checks pinning the core maths
├── notebooks/             parallel-runnable notebooks for the X-suite
├── docs/archive/          every superseded document, verbatim. Never deleted.
├── logs/                  run logs (tsuite_runs.jsonl = the audit trail)
└── results/
    ├── _feature_cache/    ~20 GB of parquet frames. Derived from L1 only. Never versioned.
    │                      Keyed by dataset+window+cap+normalize+decimate+seed. NEVER move it.
    ├── legacy_v1/         FROZEN evidence, 2026-07-13. Read-only. MANIFEST.md explains it.
    │   └── _superseded_DO_NOT_QUOTE/   floor_effect/ (self-contradicting) + stray test fixtures
    ├── v2/                every new run lands here. THIS is what comes back from the box.
    ├── _smoke_2026-07-13/ 1-dataset smoke tests proving each T experiment RUNS. Not evidence —
    │                      one dataset is never a result. Kept only to show the setup was verified.
    ├── _test_sandbox/     where the test suite writes its synthetic fixtures. NOT results.
    │                      (Before 2026-07-13 these landed in the live tree and were being counted
    │                      as a 15th dataset. `PROFILE_RESULTS_DIR` now redirects them here.)
    └── LEDGER.xlsx        THE results log — generated, never hand-typed
```

**Folders beginning with `_` are never evidence.** The ledger skips them; so should you.

**The maths lives in exactly one place.** `paper_experiments/common.py` holds MMD, the median-bandwidth
heuristic, CORAL, per-subject centring, LOSO accuracy, the statistics (Pearson/Spearman/FDR/
random-effects pooling/cluster bootstrap/permutation) and the synthetic frame generators. The T-suite
imports them; it does not re-implement them. A number computed by a T experiment and a number
computed by an X experiment must come from the same code, or they cannot be compared.

---

## 4. Coding protocol

- **Reuse the shared maths.** Before writing a statistical function, check
  `paper_experiments/common.py`. If you find yourself writing a second MMD, stop.
- **Leakage discipline, stated in the docstring.** Every evaluator says where its standardisation
  statistics come from and what the held-out subject is allowed to contribute. The rule: a held-out
  subject may contribute its own **unlabelled** data (a deployed system genuinely has that) and
  **nothing else** — no labels, no influence on the fit, no train-set statistics.
- **Check the equations by hand.** Write the identity the code is supposed to satisfy, then assert it
  in the selftest. (`kappa` is 0 at chance and 1 at perfect. `whiten` produces identity covariance.
  `coral(X, X)` is the identity map. `per_subject_center` zeroes every subject's mean.)
- **Cost belongs in the protocol, not in the results.** If an RBF-SVM cannot fit 360k rows, every
  model in that comparison gets the *same* stratified subsample. A difference between models must
  never be a difference between budgets.
- **Fail loudly, per dataset.** One dataset erroring must never kill a run or silently vanish. It is
  logged as `[FAIL]`, recorded in the run log, and reported in the pooled verdict.

## 4b. HOW TO ADD A NEW EXPERIMENT (the recipe — follow it exactly)

Say the next experiment is "T9 — does electrode count predict difficulty?".

1. **Name it.** File `tsuite/t9_electrode_count.py`. Inside it, `TAG = "t9_electrode_count"`.
   The TAG is the results folder name and the ledger key. **A TAG, once it has produced a number,
   is frozen forever** (Rule 1) — if you later change the method, the new TAG is
   `t9_electrode_count_v2` and the old results stay exactly where they are.
2. **Write the four-section docstring**, in this order and with these headings — WHY THIS EXPERIMENT
   EXISTS (name the specific gap in the committed evidence), WHAT IT DOES (protocol + leakage
   discipline: state where the standardisation statistics come from and what the held-out subject is
   allowed to contribute), PRE-REGISTERED BRANCHES (A/B/C/D, decided *before* running), GROUND TRUTH.
3. **Reuse the maths.** Import from `tsuite/common.py` and `paper_experiments/common.py`. If you
   are writing a second MMD, stop. If you need a genuinely new primitive, add it to
   `tsuite/common.py` with a comment saying why it is not already in the X-suite library.
4. **Implement three functions:**
   - `run_one(dataset, seed=42, n_jobs=1) -> dict` — one dataset, returns the per-dataset record.
     Store the *raw ingredients* of every later decision (p-values, per-subject values, counts), so
     that FDR, cohort tallies and branch thresholds can be recomputed later **without re-running**.
   - `build_pooled(tag=TAG) -> dict` — reads the per-dataset JSONs, applies FDR **within the family of
     tests**, counts **both ways** (`C.count_both_ways`: k=14 datasets AND k=9 cohorts), and returns
     `dict(branch=…, verdict=…)`.
   - `selftest(check)` — a **positive** control (finds the effect when it is there by construction)
     and a **negative** control (finds nothing when it is not). Both. Always.
5. **Register it** in `tsuite/run.py`'s `REGISTRY`: `"t9": ("t9_electrode_count", "per_dataset_pooled", None)`.
6. **Add it to `tsuite/selftest.py`'s `MODULES`** so the gate covers it.
7. **Add one line to `CATALOGUE`** in `cli/build_ledger.py` — the question it answers, in plain
   English. The ledger prints a warning for any uncatalogued experiment.
8. **Run the gate, then smoke it on ONE small dataset** (`--datasets emaha_db4`) before committing to
   the panel. This is not optional: two experiments (T2, T5) originally could not be smoke-tested at
   all, and that is exactly how a crash-on-the-last-line bug survived into T1.

## 5. Review protocol

Before an experiment's numbers may enter `RESEARCH_PREVIEW.md`:
1. Ground truth green (`python -m tsuite.selftest`).
2. Cross-check against the frozen tree where the experiments overlap. (T3's `baseline`/`center`/
   `coral` rungs must reproduce X4's committed numbers for the same dataset. When T3 first ran on
   emaha_db4 it returned 0.3247 / 0.3417 / 0.3284 — identical to frozen X4 to four decimals. That is
   what a passing cross-check looks like.)
3. An independent reviewer (a fresh agent, or a human) reads the code for: leakage, a wrong
   equation, a flag that disagrees with its own numbers, and a verdict that overstates its evidence.
4. `python -m cli.validate_results --root v2` exits 0.

---

## 6. Running experiments

**Always, in this order:**
```bash
python -m cli.preflight                 # paths, data, disk. Exit 0 = safe to run.
python -m tsuite.selftest               # ground truth. Exit 0 = safe to touch real data.
```

**Then shard by EXPERIMENT across terminals — never split one experiment's datasets across two
terminals** (a pooled verdict built from half the datasets is a wrong verdict). Keep the sum of
`--jobs` across all terminals at or below your physical core count.

```bash
# T1 is the long one (5 model families x 14 datasets) — give it its own terminal
python -m tsuite.run --exp t1 --datasets all --jobs 6 2>&1 | tee logs/t1.log

python -m tsuite.run --exp t3 t4 --datasets all --jobs 5 2>&1 | tee logs/t3_t4.log
python -m tsuite.run --exp t6 t8 --datasets all --jobs 5 2>&1 | tee logs/t6_t8.log
python -m tsuite.run --exp t2 t5 --datasets all --jobs 3 2>&1 | tee logs/t2_t5.log

# T7 rebuilds the feature cache for 3 extra seeds -> run it when the others are done
python -m tsuite.run --exp t7 --datasets all --jobs 8 2>&1 | tee logs/t7.log
```
**Always `tee` to `logs/`.** Nothing else writes stdout to disk, and the log is what the state
update is built from.

**Afterwards, always:**
```bash
python -m cli.build_ledger              # regenerate results/LEDGER.xlsx from the JSONs
```

### The runner's full flag set (`python -m tsuite.run`)

| Flag | What it does |
|---|---|
| `--exp t1 t3 …` / `--exp all` | which experiments. **Space-separated.** |
| `--datasets all` | the 14. An **explicit** comma list (`--datasets emaha_db4`) *overrides* an experiment's built-in default — this is how T2 (which defaults to senic + 3 controls) and T5 (which defaults to 4 heavy pairs) can be smoke-tested on one small dataset. For T5 an explicit list also **restricts the pairs** to those with both ends named. |
| `--jobs N` | joblib workers for THIS terminal. Keep the sum across terminals ≤ physical cores. |
| `--force` | recompute even if a result JSON already exists (default is resume/skip). |
| `--pooled-only` | **no compute.** Rebuild every `pooled.json` verdict from the per-dataset JSONs already on disk. Use this after a change to how results are *counted* (FDR, cohort tallies, branch thresholds) — none of that needs the expensive per-dataset loop re-run. |
| `--seed N` | default 42. Do not change it casually; the frame cache is keyed by seed. |

**Everything is resume-safe.** A killed terminal is restarted with the same command; finished
datasets are skipped. That is also why a crashed run never needs `--force`.

### What each experiment costs (measured on the 20-core box, 2026-07-13)

| Exp | Cost | Note |
|---|---|---|
| T1 | **~3–6 h** for all 14 | the long one — 5 model families × 311 subjects. Give it its own terminal. |
| T2 | ~1–2 h | hits grabmyo + ninapro_db2 (large) |
| T3, T4, T6, T8 | minutes per dataset | all read the cached fast frames |
| T5 | ~1 h | 4 pairs + a 200-rep subject bootstrap per arm |
| T7 | **slow** | sweeps seeds {42, 7, 1, 2026}; seeds ≠42 have different cache keys, so it **builds 3 new fast frames × 14 datasets = 42 frames**. That is its whole cost and it is unavoidable — a new seed *must* mean new data (that was the F5 fix). Run it last. |
| **T9** | **slow** | builds the `complex` frames (fast + entropy columns). All feature families must be scored on the SAME rows, so the comparison is only fair on one frame. |
| **T10** | **slow** | builds 14 `rest0` frames (everything else is `rest1`). Applicable on only 7/14 datasets — the rest report `applicable: false` with a reason. |
| **T11** | minutes | reads the cached `rest1` frames. |

**BATCH SAFETY.** T9 and T10 build NEW frames. Their cache filenames differ from everything else
(`_e3`, `_rest0`), so they can never corrupt a running job — but they will fight it for CPU. **Never
run T9/T10 alongside T1–T8.** T11 is read-only and safe any time.

The T-suite uses **fast frames only**. Preflight's warning about rebuilding *complex* (entropy)
frames applies to `module1`/`block_a` and will not fire during a T-suite run.

---

## 7. What to bring back from the compute box

Transfer is **AnyDesk file copy**, so this is a list of folders to drag — not a command.

**Bring back exactly two folders**, from
`/home/honors/Ashraf_Ali_2025_Batch/sEMG-Research-Latest/experiments-data/PROFILE/`:

| Folder | Why |
|---|---|
| `results/v2/` | the new results (one JSON per dataset + a `pooled.json` verdict per experiment) |
| `logs/` | the run logs + `tsuite_runs.jsonl` — the audit trail the STATE update is built from |

Drop them into the same places on the analysis machine, overwriting.

**Do NOT bring back:**
- `results/_feature_cache/` — ≈20 GB, derived from the L1 signals, regenerates itself.
- `results/legacy_v1/` — already here, frozen, identical.
- `results/LEDGER.xlsx` — regenerate it locally (`python -m cli.build_ledger`); it is built from the
  JSONs, so a copied one would just be stale.

---

## 8. HOW TO UPDATE STATE — the part people get wrong

There are exactly **three** documents that may be edited, and each has one job. Everything else in
`docs/archive/` is history and is never edited again.

| File | What goes in it | When |
|---|---|---|
| `STATE.md` | what is done / running / next, and an append-only history log | at the START of a run and again when its results land |
| `RESEARCH_PREVIEW.md` | the claims, with numbers and the evidence file for each | only when results land |
| `results/LEDGER.xlsx` | every number | **never by hand** — `python -m cli.build_ledger` |

### 8a. WHEN YOU SET AN EXPERIMENT RUNNING
Append to the STATE.md history log, before you walk away:

```
- **2026-07-14 09:15 — T1/T3/T4 DISPATCHED (box, 3 terminals)**
  Command: python -m tsuite.run --exp t1 --datasets all --jobs 6
  Ground truth: 36/36 green. Expected: ~6-10 h. Pre-registered branches: see tsuite/t1_model_family.py.
  Awaiting: results/v2/t1_model_family/{14 datasets}__t1_model_family.json + pooled.json
```
Then set STATE.md's **Status** line to `RUNNING: T1, T3, T4 (dispatched 2026-07-14 09:15)`.

### 8b. WHEN THE RESULTS LAND — the mandatory sequence, no exceptions
1. `rsync` back `results/v2/` and `logs/` (§7).
2. `python -m cli.build_ledger` — the ledger is now the truth.
3. `python -m cli.validate_results --root v2` — must exit 0.
4. **Read the `pooled.json` verdict and write down the branch that actually fired** — not the one
   you hoped for. If branch B fired, STATE.md says branch B fired.
5. Update, in this order:
   - `STATE.md` — move the experiment from RUNNING to DONE, with the date, the branch, the headline
     number, and the path to the evidence.
   - `RESEARCH_PREVIEW.md` — add or amend the claim, **with its evidence file path**. If a result
     contradicts a claim already there, **the claim changes**; the result does not get re-run.
   - Append to the STATE.md history log with an absolute timestamp.
6. If any number contradicts something in `docs/archive/`, that is expected — the archive is history.
   Do not "fix" the archive.

### 8c. THE FORMAT OF A HISTORY ENTRY (copy this)
```
- **2026-07-15 14:40 — T3 moment ladder: DONE, branch B**
  14/14 datasets. z-score (mean + per-channel scale) helps 12/14 (+4.1 pp mean); centring alone
  helps 13/14 (+3.6 pp); full covariance alignment (CORAL) helps 0/14 and hurts grabmyo by 11 pp.
  RULE: align the mean and the per-channel gain; never the full covariance.
  Evidence: results/v2/t3_moment_ladder/pooled.json · ledger row T3 · log logs/t3_t4.log
  Cross-check: baseline/center/coral reproduce frozen X4 on all 14. ✓
  Consequence: RESEARCH_PREVIEW claim N5 upgraded from "centring helps" to the stated rule.
```
Absolute dates and times, always. Never "yesterday", never "recently". A future reader has no idea
when you wrote it.

### 8d. NEVER
- Never write a number into a markdown file that you did not read out of a JSON **today**.
- Never mark something "RESOLVED" or "SOLID" when the artifact backing it does not close it. Use
  "open", "partial", "one dataset only". The register of the writing must match the strength of the
  evidence.
- Never update STATE.md from memory of what an experiment was *supposed* to show.

---

## 9. The 14 datasets, and what "all 14" means

`emaha_db1, emaha_db4, emaha_db5, emaha_db7, fors_emg, grabmyo, grabmyo_flow_dynamic,
grabmyo_flow_static, myobit, ninapro_db1, ninapro_db2, ninapro_db4, ninapro_db5, senic`

**They are only 9 independent cohorts.** The four EMAHA sets are one cohort; grabmyo_flow_static and
_dynamic are one 20-subject cohort; ninapro_db4 and db5 are one. Every pooled statistic must be
reported **both** ways (k=14 and k=9, one per cohort). Quoting only k=14 overstates the evidence.

Healthy subjects only — a deliberate scope choice, not a gap. Do not apologise for it, and do not
flag the absence of clinical data as a limitation.

"""Central config: paths, datasets, windowing, and feature/entropy parameters.

All parameter choices are traceable to the reviewed papers (see paper-summaries/). This
module also puts the repo root on sys.path so `import semg` works from inside PROFILE.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---- paths --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]          # E:/sEMG Research Enhanced
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# L1 data root. `SEMG_L1_ROOT` env wins; otherwise auto-discover — try <repo>/data/L1 (the original
# layout), then <repo>/../data/L1 (e.g. .../Ashraf_Ali_2025_Batch/data/L1 when the repo is a sibling
# of data/). This makes the code portable across machines without hardcoding any absolute path.
_L1_ENV = os.environ.get("SEMG_L1_ROOT")
if _L1_ENV:
    L1_ROOT = Path(_L1_ENV)
else:
    _L1_CANDIDATES = [REPO_ROOT / "data" / "L1", REPO_ROOT.parent / "data" / "L1"]
    L1_ROOT = next((c for c in _L1_CANDIDATES if c.exists()), _L1_CANDIDATES[0])
PROFILE_DIR = Path(__file__).resolve().parents[1]         # experiments-data/PROFILE

# ---- results: one frozen tree + one live tree (see CLAUDE.md §"Results are immutable") ----
# RESULTS_ROOT holds BOTH the frozen evidence and the live outputs. Nothing ever overwrites
# LEGACY_DIR: a re-run of any module writes into RESULTS_DIR (v2) instead, so an old number and a
# new number can never occupy the same path.
RESULTS_ROOT = PROFILE_DIR / "results"
LEGACY_DIR = RESULTS_ROOT / "legacy_v1"                   # FROZEN 2026-07-13. Read-only.
# `PROFILE_RESULTS_DIR` redirects every WRITE. The test suite sets it to a sandbox so that synthetic
# fixtures (`synth__*.json`) can never land in a real results tree — they did, before 2026-07-13, and
# they were being counted as a 15th dataset by anything that globbed the folder.
RESULTS_DIR = Path(os.environ.get("PROFILE_RESULTS_DIR") or (RESULTS_ROOT / "v2"))
# The feature cache is deliberately OUTSIDE the versioned trees: it is derived from L1 signals only
# (never from results), it costs ~20 GB and many CPU-hours, and its filenames already encode every
# parameter that can change it (window, cap, normalize, decimate, seed). Moving it would force a
# full rebuild on the box for no benefit.
L3_CACHE = RESULTS_ROOT / "_feature_cache"


def find(*parts) -> Path:
    """Resolve a results path for READING: prefer the live tree, fall back to the frozen one.

    `config.find("module5", "emaha_db1__difficulty.parquet")` returns the v2 file if a newer run
    has produced one, else the legacy_v1 file. Returns the v2 path when neither exists, so callers
    get a sensible `FileNotFoundError` message.
    """
    live = RESULTS_DIR.joinpath(*parts)
    if live.exists():
        return live
    frozen = LEGACY_DIR.joinpath(*parts)
    if frozen.exists():
        return frozen
    return live


def find_all(sub: str, pattern: str) -> list[Path]:
    """Glob `sub/pattern` across both trees; a live file SHADOWS the frozen file of the same name."""
    live = {p.name: p for p in sorted(RESULTS_DIR.joinpath(sub).glob(pattern))}
    frozen = {p.name: p for p in sorted(LEGACY_DIR.joinpath(sub).glob(pattern))}
    merged = {**frozen, **live}                            # live wins on a name collision
    return [merged[k] for k in sorted(merged)]

# ---- datasets ----------------------------------------------------------------------
# the original starter six; ALL14 is what every experiment actually runs on
SIX = ["emaha_db1", "fors_emg", "grabmyo", "ninapro_db1", "ninapro_db2", "ninapro_db5"]
ALL14 = SIX + ["emaha_db4", "emaha_db5", "emaha_db7", "grabmyo_flow_dynamic",
               "grabmyo_flow_static", "myobit", "ninapro_db4", "senic"]
# datasets that have >1 session/subject -> support the inter-day (A4) axis
MULTISESSION = ["grabmyo", "senic"]

# ---- windowing (decision #3: ~250 ms primary for entropy stability) -----------------
WINDOW_MS = 250.0
OVERLAP = 0.5
WINDOW_MS_SECONDARY = 100.0        # sensitivity axis

# drop the rest class (label 0) for separability/shift (matches Track-1 --drop-rest)
DROP_REST = True

# ---- entropy / complexity parameters (paper-traced) ---------------------------------
ENT = dict(
    m=2,                # embedding dim (Xie 2010, Pincus): m=2 validated for EMG (summary 03)
    r_sampen=0.20,      # SampEn tolerance = r*std (standard)
    r_fuzzy=0.30,       # FEn tolerance = r*std, best operating point (summary 01)
    n_fuzzy=5,          # FEn membership gradient exp(-d^n/r), n=5 best (summary 01)
    r_fapen=0.25,       # fApEn tolerance = r*std (summaries 02, 03)
    perm_d=4,           # PEn embedding; d!<<N -> d=4 safe at 250-500 sample windows (summary 01)
    perm_tau=1,
    ms_scales=10,       # MSfApEn coarse-grain scales 1..10 (summary 02)
    hfd_kmax=10,        # Higuchi fractal dimension kmax (summary 12)
    min_samples=200,    # entropy needs >= ~200 samples/window (summary 01) — ENFORCED in
                        # features_extra.slow_features: shorter windows -> NaN, not garbage.
)

# ---- validity guards (added after the 2026-07-10 results audit) ---------------------
# Entropy/complexity is only meaningful when a window holds >= ENT["min_samples"] samples.
# 250 ms at 100-200 Hz gives 25-50 samples -> SampEn NaN'd but FuzzyEn/fApEn/PermEn/HFD
# silently returned values computed on 25 samples. Datasets below this fs are flagged.
ENTROPY_MIN_FS = ENT["min_samples"] / (WINDOW_MS / 1000.0)     # = 800 Hz at 250 ms

# E6 sampling-rate sufficiency. With a proper anti-alias filter, decimating to 500 Hz is a
# LEGITIMATE test point (low-pass to 250 Hz + resample) — that IS the K2 question. The guards
# are instead: (a) the dataset must start high enough that there is bandwidth to remove, and
# (b) the decimated window must still hold enough samples for the features to be stable.
E6_MIN_NATIVE_FS_HZ = 1000.0     # below this there is nothing to decimate away
E6_MIN_EFFECTIVE_FS_HZ = 500.0   # 250 ms at 500 Hz = 125 samples; Nyquist 250 Hz

# Cohort structure for the meta-analysis. Datasets sharing a cohort are NOT independent
# samples; k=14 overstates the evidence. Used for the leave-one-cohort-out sensitivity.
COHORTS = {
    "emaha_db1": "emaha", "emaha_db4": "emaha", "emaha_db5": "emaha", "emaha_db7": "emaha",
    "grabmyo": "grabmyo", "grabmyo_flow_dynamic": "grabmyo_flow",
    "grabmyo_flow_static": "grabmyo_flow",
    "ninapro_db1": "ninapro_db1", "ninapro_db2": "ninapro_db2",
    "ninapro_db4": "ninapro_db45", "ninapro_db5": "ninapro_db45",
    "fors_emg": "fors", "myobit": "myobit", "senic": "senic",
}

# Module-5 primary predictor, fixed A PRIORI to avoid the winner's-curse of picking the
# best-of-4 per dataset. All four are still reported; only this one carries the headline.
PRIMARY_PREDICTOR = "mmd_to_pool"

# ---- MMD kernel bandwidth (F4) -------------------------------------------------------
# "median" = median heuristic, gamma = 1/(2*median||xi-xj||^2). This is the field standard and
# is now PRIMARY: a fixed gamma=1/d does not adapt to data scale/dimension, so cross-dataset MMD
# magnitudes were not comparable. "inv_d" reproduces the pre-2026-07-12 committed numbers.
# X7 reports the full sensitivity envelope (median vs inv_d vs multi-kernel) — it found the
# difficulty sign stable on 12/14, so this switch changes magnitudes, not conclusions.
MMD_GAMMA = os.environ.get("PROFILE_MMD_GAMMA", "median")     # "median" | "inv_d"

# senic is the lone sign-flipped dataset; report headline stats with and without it.
OUTLIER_DATASETS = ["senic"]

# ---- parallelism (multiprocessing via joblib/loky) ----------------------------------
# -1 = use all cores. LOWER THIS (e.g. 8) when running alongside GPU training to stay
# memory-safe: `--jobs 8` on the CLI or `export PROFILE_JOBS=8`.
N_JOBS = int(os.environ.get("PROFILE_JOBS", "-1"))


def resolve_jobs(upper=None):
    cpu = os.cpu_count() or 1
    j = cpu if N_JOBS in (-1, 0) else max(1, N_JOBS)
    return max(1, min(j, upper) if upper else j)


# per-(subject,class) window subsample cap for the O(N^2) entropy features (Module 1)
ENTROPY_MAX_WINDOWS_PER_CLASS = 40
# cap the samples-per-window fed to the O(N^2) entropy features (decimate longer windows).
# fApEn is robust to window length (summary 03), so this trades negligible accuracy for speed.
ENTROPY_MAX_SAMPLES = 400
# MSfApEn (10x fApEn per series) is opt-in — enable for the dedicated multiscale figure only.
COMPUTE_MSFAPEN = False
# general subsample cap per (subject,class) for the fast feature frame. 200 is plenty: the
# downstream stats subsample further (silhouette/kNN/TwoNN to <=4000, module3 to 600/subject,
# module5 to 800/subject), so a bigger cap only inflates build time without changing results.
MAX_WINDOWS_PER_CLASS = 200

# ---- feature groups (names resolved in features_extra.py) ---------------------------
FAST_TIME = ["MAV", "RMS", "WL", "VAR", "IEMG", "SSI", "DASDV", "AAC", "LOG",
             "LOGRMS", "NLE", "ZC", "SSC", "WAMP", "MYOP", "SKEW", "KURT", "P75",
             "HJ_ACT", "HJ_MOB", "HJ_COM", "MFL"]
FAST_FREQ = ["MNF", "MDF", "SENT", "MNP", "TTP"]           # skipped on envelope datasets
SLOW_COMPLEX = ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD"]   # per-series O(N^2)/loop
MS_COMPLEX = ["MSFAPEN"]                                   # optional multiscale (3 outputs)

# thresholds for ZC/SSC/WAMP/MYOP on z-scored signal (dimensionless)
THRESH = 0.05

# representative de-duplicated basis for separability/shift (Phinyomark summary 04 + 12)
REPR_BASIS = ["MAV", "WL", "WAMP", "RMS", "HJ_MOB", "HJ_COM", "MFL"]


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    L3_CACHE.mkdir(parents=True, exist_ok=True)

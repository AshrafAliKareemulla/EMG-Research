"""Preflight — run this FIRST on the Ubuntu box, before any experiment.

Verifies every path and input the suite depends on, and prints exactly what it resolved, so a long
run cannot die hours in on a missing file or a wrong path.

    python preflight.py

Exit code 0 = safe to run. Non-zero = fix what it prints.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, BAD, WARN = [], [], []


def ok(m):
    OK.append(m); print(f"  [OK]   {m}")


def bad(m):
    BAD.append(m); print(f"  [BAD]  {m}")


def warn(m):
    WARN.append(m); print(f"  [WARN] {m}")


print("=" * 78)
print("PROFILE preflight")
print("=" * 78)

# ------------------------------------------------------------------ 1. paths
print("\n1. Paths")
from dsprofile import config

print(f"     REPO_ROOT   = {config.REPO_ROOT}")
print(f"     L1_ROOT     = {config.L1_ROOT}   (env SEMG_L1_ROOT={os.environ.get('SEMG_L1_ROOT')})")
print(f"     RESULTS_DIR = {config.RESULTS_DIR}")
print(f"     CACHE       = {config.L3_CACHE}")

if config.L1_ROOT.exists():
    ok(f"L1 data root exists: {config.L1_ROOT}")
else:
    bad(f"L1 data root NOT FOUND: {config.L1_ROOT}  ->  export SEMG_L1_ROOT=/path/to/data/L1")

# ------------------------------------------------------------------ 2. datasets
print("\n2. Datasets (need signals.h5 + manifest.parquet)")
missing = []
for ds in config.ALL14:
    d = config.L1_ROOT / ds
    if (d / "signals.h5").exists() and (d / "manifest.parquet").exists():
        continue
    missing.append(ds)
if missing:
    bad(f"{len(missing)}/14 datasets missing or incomplete: {missing}")
else:
    ok(f"all 14 datasets present under {config.L1_ROOT}")

# ------------------------------------------------------------------ 3. env
print("\n4. Environment")
try:
    import numpy, scipy, sklearn, pandas          # noqa: F401
    ok(f"numpy {numpy.__version__} · scipy {scipy.__version__} · "
       f"sklearn {sklearn.__version__} · pandas {pandas.__version__}")
except Exception as e:                            # pragma: no cover
    bad(f"missing dependency: {e}  ->  pip install -r requirements.txt")
try:
    import h5py, statsmodels                      # noqa: F401
    ok("h5py + statsmodels present (statsmodels powers X1's MixedLM cross-check)")
except Exception as e:
    warn(f"optional dep missing: {e}")

print(f"\n     PROFILE_JOBS     = {os.environ.get('PROFILE_JOBS', '(unset -> all cores)')}")
print(f"     PROFILE_MMD_GAMMA= {config.MMD_GAMMA}   (F4: 'median' is now PRIMARY; 'inv_d' = legacy)")
print(f"     cpu_count        = {os.cpu_count()}")

# ------------------------------------------------------------------ 4. cache
print("\n5. Feature cache")
if config.L3_CACHE.exists():
    fast = list(config.L3_CACHE.glob("*__fast__*.parquet"))
    cplx = list(config.L3_CACHE.glob("*__complex__*.parquet"))
    old_cplx = [p for p in cplx if "_e3" not in p.name]
    ok(f"cache dir present: {len(fast)} fast frame(s), {len(cplx)} complex frame(s)")
    if fast:
        ok("fast frames are REUSED (seed=42 keeps its historical filename; F5 does not invalidate)")
    if old_cplx:
        warn(f"{len(old_cplx)} PRE-F-dec complex frame(s) will be rebuilt (aliased entropy) — expected, "
             f"this is the slow part of the run")
else:
    warn("no cache yet — the first run will build it (this is the expensive step)")

# ------------------------------------------------------------------ verdict
print("\n" + "=" * 78)
print(f"{len(OK)} ok · {len(WARN)} warn · {len(BAD)} blocking")
if BAD:
    print("PREFLIGHT FAILED — fix the [BAD] items above before running.")
    sys.exit(1)
print("PREFLIGHT PASSED — safe to run the suite.")
print("=" * 78)

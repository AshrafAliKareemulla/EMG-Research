"""Shared infra for the add-on experiments A / B / C.

Gives every experiment the same three properties the user asked for:

  * RESUME / SKIP   — a finished per-dataset output is not recomputed (so a crashed or sharded
                      run just continues). `--force` overrides.
  * ATOMIC WRITES   — outputs are written to a temp file and renamed, so a killed process can
                      never leave a half-written JSON that a later `--collect` would choke on.
  * SHARD-SAFE      — the per-dataset compute writes ONLY per-dataset files; the cross-dataset
                      summary is a separate `--collect` pass that reads whatever per-dataset
                      files exist. Two terminals running disjoint `--datasets` therefore never
                      write the same file.

Parallelism / cache safety (READ THIS):
  build_fast_frame reads-or-BUILDS a parquet cache keyed by (dataset, window, normalize,
  decimate). The ONLY experiment that builds new caches is A (100 ms + 500 ms frames); B/C/D/E
  only read the existing 250 ms cache. So:
    - two A shards are safe iff their --datasets are DISJOINT (disjoint cache files);
    - B/C/D/E are safe next to A and each other (read-only);
    - keep total joblib workers across all terminals <= physical cores.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import traceback

from dsprofile import config


def resolve_datasets(spec):
    """'all' -> the 14; else a comma list. Accepts a str or an already-split list."""
    if spec in ("all", ["all"], None):
        return list(config.ALL14)
    if isinstance(spec, (list, tuple)):
        return [str(d).strip() for d in spec if str(d).strip()]
    return [d.strip() for d in str(spec).split(",") if d.strip()]


def experiments_dir():
    d = config.RESULTS_DIR / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write_json(path, obj):
    """Write JSON to a temp file in the same dir, then os.replace() it into place (atomic on the
    same filesystem). Guarantees a reader/collector never sees a partial file."""
    path = str(path)
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None and os.path.exists(tmp):
            os.remove(tmp)


def per_dataset_path(tag, dataset):
    return experiments_dir() / f"exp_{tag}_{dataset}__{tag_suffix(tag)}.json"


def tag_suffix(tag):
    return {"A": "window", "B": "recal", "C": "crosssession"}.get(tag, tag)


def run_sharded(tag, datasets, run_one, force=False):
    """Process `datasets` with `run_one(ds) -> dict`, one atomic per-dataset file each, with
    resume/skip and per-dataset error isolation. Returns {dataset: result_or_error}. Does NOT
    write the cross-dataset summary (that is `--collect`), so it is safe to shard."""
    out = {}
    for ds in datasets:
        p = per_dataset_path(tag, ds)
        if p.exists() and not force:
            print(f"[SKIP] {tag} :: {ds} :: already done ({p.name})", flush=True)
            try:
                out[ds] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                out[ds] = dict(note="existing file unreadable; use --force")
            continue
        t0 = time.time()
        try:
            r = run_one(ds)
        except Exception as e:                       # isolate: one bad dataset never kills the shard
            traceback.print_exc()
            print(f"[FAIL] {tag} :: {ds} :: {type(e).__name__}: {e}", flush=True)
            out[ds] = dict(error=f"{type(e).__name__}: {e}")
            continue
        atomic_write_json(p, r)
        out[ds] = r
        print(f"[OK] {tag} :: {ds} :: {time.time() - t0:.1f}s -> {p.name}", flush=True)
    return out


def collect(tag, datasets):
    """Read all existing per-dataset files for `tag` and return {dataset: result}. Missing ones
    are reported so `--collect` can tell you the run is incomplete."""
    out, missing = {}, []
    for ds in datasets:
        p = per_dataset_path(tag, ds)
        if p.exists():
            out[ds] = json.loads(p.read_text(encoding="utf-8"))
        else:
            missing.append(ds)
    return out, missing


def _write_summary(tag, results, build_summary, missing):
    summary = build_summary(results)
    summary["_missing_datasets"] = missing
    summary["_complete"] = not missing
    p = experiments_dir() / f"exp_{tag}_summary.json"
    atomic_write_json(p, summary)
    print(f"\n=== {tag}: cross-dataset summary -> {p.name} "
          f"({'COMPLETE' if not missing else 'INCOMPLETE, missing ' + ','.join(missing)}) ===",
          flush=True)
    for line in summary.get("_console", []):
        print("  " + line, flush=True)
    return summary


def main(tag, run_one, build_summary, all_datasets, default_datasets=None):
    """Standard CLI for an add-on experiment.

    run_one(ds, seed, jobs) -> dict         : compute one dataset (must be self-contained)
    build_summary(results) -> dict          : {..., "_console": [lines]} cross-dataset summary
    all_datasets                            : full scope the summary spans
    default_datasets                        : what to run when --datasets is omitted (defaults to all)

    --datasets a,b,c | all   which datasets this terminal handles (shard here)
    --jobs N                 joblib workers for THIS terminal (keep sum over terminals <= cores)
    --force                  recompute even if a per-dataset file exists
    --collect                only rebuild the cross-dataset summary from existing per-dataset files
    """
    import argparse
    default_datasets = list(default_datasets or all_datasets)
    ap = argparse.ArgumentParser(description=f"Experiment {tag}")
    ap.add_argument("--datasets", default=",".join(default_datasets))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()
    if args.jobs is not None:
        config.N_JOBS = args.jobs
    datasets = resolve_datasets(args.datasets)

    if args.collect:
        results, missing = collect(tag, all_datasets)
        _write_summary(tag, results, build_summary, missing)
        return

    print(f"[{tag}] datasets={datasets} jobs={config.N_JOBS} force={args.force}", flush=True)
    run_sharded(tag, datasets, lambda ds: run_one(ds, args.seed, args.jobs), args.force)

    if set(datasets) >= set(all_datasets):           # a full (non-sharded) run -> auto-collect
        results, missing = collect(tag, all_datasets)
        _write_summary(tag, results, build_summary, missing)
    else:
        print(f"\n[shard done] processed {len(datasets)} dataset(s). After ALL shards finish, run:"
              f"\n    python exp_{tag}_*.py --collect\n"
              f"to build the cross-dataset summary.", flush=True)

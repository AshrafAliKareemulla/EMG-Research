"""Build a (meta + feature) frame for one dataset, reusing semg's leakage-safe windowing.

Two builders:
  * build_fast_frame  -> FAST features on windows capped per (subject, class).
  * build_complex_frame -> FAST + SLOW (entropy/HFD) on a small subsample per (subject,class),
    because the entropy features are O(N^2).

Normalisation: train-only GLOBAL z-score via semg's Normalizer(mode="global"). For this
descriptive paper the whole dataset acts as the "train" pool (there is no held-out test);
this equalises per-channel scale across subjects so distances are comparable (summaries 05/07/14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import h5py

from . import config
from . import features_extra as fx
from . import progress

# semg is importable because config.py put REPO_ROOT on sys.path
from semg.data.window_index import build_window_index
from semg.splits.splitter import Normalizer

META = ["subject", "session", "repetition", "label"]


def _cache_file(dataset, kind, cap, normalize="global", decimate=1, seed=42):
    config.ensure_dirs()
    nz = "" if normalize == "global" else f"_nz-{normalize}"
    # v2 = anti-aliased decimation (was naive `[::q]` subsampling). The tag forces a rebuild
    # rather than silently reusing the aliased Phase-2 caches.
    dc = "" if decimate == 1 else f"_dec{decimate}v2"
    # F5 — the SEED is part of the frame's identity. The per-(subject,class) window subsample is
    # drawn with `rng = default_rng(seed)`, so two seeds are two different frames; without the seed
    # in the key, a re-run with seed=7 silently reused seed=42's parquet and "seed robustness" was
    # unfalsifiable without deleting the cache by hand. seed=42 keeps its historical (unsuffixed)
    # filename so the existing caches are still valid and are NOT needlessly rebuilt.
    sd = "" if seed == 42 else f"_s{seed}"
    # e3 = anti-aliased ENTROPY decimation inside slow_features (F-dec). Only the `complex` frame
    # carries entropy columns, so only that cache is invalidated by the fix.
    ev = "_e3" if kind == "complex" else ""
    return config.L3_CACHE / (
        f"{dataset}__{kind}__win{int(config.WINDOW_MS)}_ov{int(config.OVERLAP*100)}"
        f"_cap{cap}_rest{int(config.DROP_REST)}{nz}{dc}{sd}{ev}.parquet")


def _load(dataset):
    root = config.L1_ROOT / dataset
    manifest = pd.read_parquet(root / "manifest.parquet")
    return str(root), manifest


def _subsample_positions(idx, cap, rng):
    """Return the WINDOW positions to keep, capped per (subject, label) group."""
    df = pd.DataFrame({"subject": idx.subjects, "label": idx.labels})
    keep = []
    for _, g in df.groupby(["subject", "label"], sort=False):
        pos = g.index.to_numpy()
        if cap is not None and len(pos) > cap:
            pos = rng.choice(pos, size=cap, replace=False)
        keep.append(pos)
    return np.sort(np.concatenate(keep)) if keep else np.array([], dtype=int)


def _assemble(feat_dict, n_ch):
    """{name: (B, C)} -> DataFrame with columns NAME_c{ch} (deterministic order)."""
    cols = {}
    for name, arr in feat_dict.items():
        arr = np.asarray(arr)
        for c in range(n_ch):
            cols[f"{name}_c{c}"] = arr[:, c]
    return pd.DataFrame(cols)


def _build(dataset, feature_names, cap, with_slow, seed=42, normalize="global", decimate=1):
    root, manifest = _load(dataset)
    is_env = bool(manifest.is_envelope.iloc[0])
    n_ch = int(manifest.n_channels.iloc[0])
    if config.DROP_REST:
        manifest = manifest[manifest.label != 0].reset_index(drop=True)

    idx = build_window_index(manifest, config.WINDOW_MS, config.OVERLAP)
    if len(idx) == 0:
        raise ValueError(f"{dataset}: no full windows at {config.WINDOW_MS} ms")

    rng = np.random.default_rng(seed)
    keep = _subsample_positions(idx, cap, rng)

    # normalisation fit on ALL subjects present ("global" z-score, or "none" -> raw features)
    subs = sorted(int(s) for s in np.unique(idx.subjects))
    norm = Normalizer.fit(root, subs, mode=normalize)

    # drop frequency features on envelope datasets (no real spectrum)
    names = list(feature_names)
    if is_env:
        names = [n for n in names if n not in config.FAST_FREQ]

    # group kept positions by trial so each trial is read once
    keep_keys = np.array([idx.trial_keys[p] for p in keep])
    order = np.argsort(keep_keys, kind="stable")
    keep = keep[order]; keep_keys = keep_keys[order]

    meta_rows = {k: [] for k in META}
    fast_parts, slow_wins = [], []          # slow (entropy) windows collected, computed in parallel later
    ent = config.ENT

    n_trials = len(np.unique(keep_keys))
    done = 0
    step = max(1, n_trials // 10)
    # precompute per-trial fs + repetition as dicts (O(1) lookup; avoids an O(n^2) manifest scan)
    fs_by_key = dict(zip(manifest.trial_key, manifest.fs.astype(int)))
    rep_by_key = dict(zip(manifest.trial_key, manifest.repetition.astype(int)))
    with h5py.File(f"{root}/signals.h5", "r") as f:
        i = 0
        while i < len(keep):
            key = keep_keys[i]
            j = i
            while j < len(keep) and keep_keys[j] == key:
                j += 1
            done += 1
            if done % step == 0:
                progress.log(f"  {dataset} {'complex' if with_slow else 'fast'} frame: "
                             f"trial {done}/{n_trials} ({100*done/n_trials:.0f}%)")
            sig = f[key][:].astype(np.float64)                      # (C, T)
            fsv = fs_by_key[key]
            positions = keep[i:j]
            wl = int(idx.ends[positions[0]] - idx.starts[positions[0]])
            wins = np.empty((j - i, sig.shape[0], wl), dtype=np.float64)
            for w, p in enumerate(positions):
                seg = sig[:, idx.starts[p]:idx.ends[p]]
                if not is_env:
                    seg = seg - seg.mean(axis=1, keepdims=True)     # per-window detrend (raw)
                seg = norm(seg, int(idx.subjects[p]), int(idx.sessions[p]))
                wins[w] = seg
                meta_rows["subject"].append(int(idx.subjects[p]))
                meta_rows["session"].append(int(idx.sessions[p]))
                meta_rows["repetition"].append(rep_by_key.get(key, 0))
                meta_rows["label"].append(int(idx.labels[p]))
            if decimate > 1:                                       # sampling-rate sufficiency (E6)
                wins = _antialias_decimate(wins, decimate)
                fsv = max(1, fsv // decimate)
            fast_parts.append(_assemble(fx.fast_features(wins, fsv, names, config.THRESH), sig.shape[0]))
            if with_slow:
                slow_wins.append(wins)
            i = j

    meta_df = pd.DataFrame(meta_rows)
    frame = pd.concat([meta_df.reset_index(drop=True),
                       pd.concat(fast_parts, ignore_index=True)], axis=1)

    if with_slow and slow_wins:
        slow_names = list(config.SLOW_COMPLEX) + (config.MS_COMPLEX if config.COMPUTE_MSFAPEN else [])
        slow_df = _slow_parallel(slow_wins, slow_names, ent, n_ch, dataset)
        frame = pd.concat([frame, slow_df], axis=1)

    frame.attrs["dataset"] = dataset
    frame.attrs["is_envelope"] = is_env
    frame.attrs["n_channels"] = n_ch
    return frame


def _slow_chunk(wins_list, slow_names, ent, max_samples):
    """Compute slow (entropy) features for a contiguous list of per-trial window batches."""
    return [fx.slow_features(w, slow_names, ent, max_samples=max_samples) for w in wins_list]


def _slow_parallel(slow_wins, slow_names, ent, n_ch, dataset):
    """Parallelise the O(N^2) entropy features across CPU cores (loky), preserving row order.
    Trials are split into contiguous chunks so concatenation matches the fast-frame order."""
    from joblib import Parallel, delayed
    workers = config.resolve_jobs(upper=len(slow_wins))
    bounds = np.array_split(np.arange(len(slow_wins)), workers)
    chunks = [[slow_wins[k] for k in b] for b in bounds if len(b) > 0]
    with progress.timer(f"{dataset} entropy features ({len(slow_wins)} trials, {workers} workers)"):
        results = Parallel(n_jobs=workers, backend="loky")(
            delayed(_slow_chunk)(chunk, slow_names, ent, config.ENTROPY_MAX_SAMPLES)
            for chunk in chunks)
    dicts = [d for chunk in results for d in chunk]         # flatten in chunk order == trial order
    parts = [_assemble(d, n_ch) for d in dicts]
    return pd.concat(parts, ignore_index=True)


def _antialias_decimate(wins, q):
    """Decimate (B, C, T) windows by `q` WITH an anti-aliasing filter.

    The original code did `wins[:, :, ::q]` — naive subsampling. sEMG carries content to
    ~450 Hz, so decimating 2 kHz by 4 folds the 250-500 Hz band back into the passband and
    corrupts every frequency feature (MNF/MDF/SENT/MNP/TTP). Amplitude features survive
    aliasing largely intact, which is precisely why the accuracy curve looked deceptively
    flat. `scipy.signal.decimate` applies a zero-phase FIR low-pass first.
    """
    from scipy.signal import decimate as _dec
    n = wins.shape[-1]
    # FIR of order 20*q needs ~3*order samples of padding for filtfilt; fall back if short.
    if n <= 27 * q:
        return wins[:, :, ::q]
    return np.ascontiguousarray(_dec(wins, q, ftype="fir", zero_phase=True, axis=-1))


def _load_or_build(dataset, feature_names, cap, with_slow, kind, seed, normalize="global", decimate=1):
    """Compute the frame ONCE per (dataset, kind, cap, window, normalize, decimate) and cache it to
    parquet, so modules reuse each other's work and a re-run after a crash skips finished datasets."""
    cf = _cache_file(dataset, kind, cap, normalize, decimate, seed)
    root, manifest = _load(dataset)
    if cf.exists():
        progress.log(f"cache HIT  {kind} :: {dataset}  ({cf.name})")
        frame = pd.read_parquet(cf)
    else:
        with progress.timer(f"build {kind} frame :: {dataset} (norm={normalize}, dec={decimate})"):
            frame = _build(dataset, feature_names, cap, with_slow, seed, normalize, decimate)
        frame.to_parquet(cf, index=False)                 # incremental save of the heavy step
        progress.log(f"cached     {kind} :: {dataset}  ({len(frame)} windows -> {cf.name})")
    # parquet drops DataFrame.attrs -> re-attach (cheap, from the manifest)
    frame.attrs["dataset"] = dataset
    frame.attrs["is_envelope"] = bool(manifest.is_envelope.iloc[0])
    frame.attrs["n_channels"] = int(manifest.n_channels.iloc[0])
    return frame


def dataset_fs(dataset):
    """Native sampling rate (min across trials), or None when there is no manifest.

    Returns None rather than raising so synthetic/in-memory frames (tests) work unchanged.
    """
    try:
        _, manifest = _load(dataset)
        return int(manifest.fs.min())
    except Exception:
        return None


def window_samples(dataset):
    """Samples per window at the configured WINDOW_MS, or None if fs is unknown."""
    fs = dataset_fs(dataset)
    return None if fs is None else int(np.floor(fs * config.WINDOW_MS / 1000.0))


def entropy_valid(dataset):
    """Is the window long enough for the entropy/fractal block to mean anything?

    250 ms at 100-200 Hz is 25-50 samples. SampEn returns NaN there (no template matches),
    but FuzzyEn / fApEn / PermEn / HFD return finite numbers that are meaningless. Those
    numbers reached the Module-1 cards, Block-A reliability and the meta predictors.

    Unknown fs (no manifest) -> True: the per-window guard in `features_extra.slow_features`
    is the authoritative check, and it cannot be bypassed.
    """
    n = window_samples(dataset)
    return True if n is None else n >= int(config.ENT["min_samples"])


def mask_invalid_complexity(frame, dataset):
    """NaN the complexity columns when the window is too short to support them.

    Applied on READ so existing `complex` parquet caches (built before the guard landed)
    do not need the ~46 min rebuild; validity is a pure function of fs and WINDOW_MS.
    """
    if entropy_valid(dataset):
        return frame
    bad = tuple(config.SLOW_COMPLEX) + ("MSFAPEN",)
    cols = [c for c in frame.columns if c.split("_c")[0] in bad or c.startswith("MSFAPEN")]
    if cols:
        attrs = dict(frame.attrs)
        frame = frame.copy()
        frame[cols] = np.nan
        frame.attrs.update(attrs)                 # .copy() drops attrs on some pandas versions
        progress.log(f"{dataset}: window={window_samples(dataset)} samples "
                     f"< {config.ENT['min_samples']} -> complexity features masked to NaN")
    frame.attrs["entropy_valid"] = False
    return frame


def build_fast_frame(dataset, seed=42, normalize="global", decimate=1):
    """FAST features, capped per (subject, class). For Modules 2/3/4/5 (shared, cached).
    normalize='none' -> raw-scale features (E3); decimate>1 -> lower effective fs (E6)."""
    return _load_or_build(dataset, config.FAST_TIME + config.FAST_FREQ,
                          config.MAX_WINDOWS_PER_CLASS, False, "fast", seed, normalize, decimate)


def build_complex_frame(dataset, seed=42):
    """FAST + SLOW (entropy/HFD/MSfApEn) on a small subsample. For Module 1's complexity block.

    Complexity columns are NaN'd for datasets whose 250 ms window holds < 200 samples
    (fs < 800 Hz): ninapro_db1 (100 Hz, envelope), ninapro_db5 / senic (200 Hz), myobit (176 Hz).
    """
    frame = _load_or_build(dataset, config.FAST_TIME + config.FAST_FREQ,
                           config.ENTROPY_MAX_WINDOWS_PER_CLASS, True, "complex", seed)
    frame.attrs["entropy_valid"] = True
    return mask_invalid_complexity(frame, dataset)


def feature_cols(frame):
    return [c for c in frame.columns if c not in META]

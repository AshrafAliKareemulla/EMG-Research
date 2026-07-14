"""Lightweight progress + timing helpers (tqdm if available, else timestamped prints)."""
from __future__ import annotations

import time

_T0 = time.time()


def log(msg):
    el = time.time() - _T0
    print(f"[{time.strftime('%H:%M:%S')} +{el:7.1f}s] {msg}", flush=True)


class timer:
    """Context manager: logs 'start ...' and 'done ... (X.Xs)'."""
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.t = time.time()
        log(f"START {self.msg}")
        return self

    def __exit__(self, *a):
        log(f"DONE  {self.msg} ({time.time() - self.t:.1f}s)")


def pbar(iterable, desc, total=None):
    """tqdm progress bar if installed; otherwise a plain iterator that logs every ~10%."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, mininterval=1.0)
    except Exception:
        return _fallback(iterable, desc, total)


def _fallback(iterable, desc, total):
    if total is None:
        try:
            total = len(iterable)
        except Exception:
            total = None
    step = max(1, (total or 100) // 10)
    for i, x in enumerate(iterable):
        if total and (i % step == 0):
            log(f"  {desc}: {i}/{total} ({100*i/total:.0f}%)")
        yield x
    if total:
        log(f"  {desc}: {total}/{total} (100%)")

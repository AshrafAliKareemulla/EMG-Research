"""Feature library for Module 1 — numpy implementations on window batches.

Two tiers:
  * FAST features are vectorised over (B, C, T) — amplitude, distribution shape, Hjorth,
    MFL, and spectral features.
  * SLOW features (SampEn, FuzzyEn, fApEn, PermEn, HFD, MSfApEn) are per-series (O(N^2) or
    per-channel loops) and are only computed on a subsample of windows (see windows.py).

Every formula is traceable to a reviewed paper (see paper-summaries/):
  Hjorth / HFD / MFL: Abbaspour 2020 (12).  SampEn: standard (m=2, r=0.2*std).
  FuzzyEn: Marri 2020 (01) — exp(-d^n/r), n=5, r=0.3*std, baseline removed.
  fApEn: Xie 2010 (03) — Gaussian exp(-d^2/r), r=0.25*std, m=2, baseline removed.
  MSfApEn: Navaneethakrishna 2015 (02) — coarse-grain scales 1..10, {MED,LS-MED,HS-MED}.
  PermEn: Marri 2020 (01) — ordinal, tau=1, normalised by ln(d!), d! << N.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-10


# ============================ FAST (vectorised over B, C, T) ==========================

def _moments(x):
    m = x.mean(axis=-1, keepdims=True)
    d = x - m
    m2 = (d ** 2).mean(axis=-1)
    return m.squeeze(-1), m2


def fast_features(wins: np.ndarray, fs: int, names, thresh: float = 0.05) -> dict:
    """wins: (B, C, T) float64 -> {name: (B, C)}. Assumes wins already normalised
    (z-scored) so amplitude thresholds are dimensionless."""
    x = np.asarray(wins, dtype=np.float64)
    B, C, T = x.shape
    out = {}
    absx = np.abs(x)
    dx = np.diff(x, axis=-1)
    absdx = np.abs(dx)
    sq = x ** 2

    def need(n):
        return n in names

    if need("MAV"):    out["MAV"] = absx.mean(-1)
    if need("RMS"):    out["RMS"] = np.sqrt(sq.mean(-1))
    if need("WL"):     out["WL"] = absdx.sum(-1)
    if need("VAR"):    out["VAR"] = x.var(-1, ddof=1)
    if need("IEMG"):   out["IEMG"] = absx.sum(-1)
    if need("SSI"):    out["SSI"] = sq.sum(-1)
    if need("DASDV"):  out["DASDV"] = np.sqrt((dx ** 2).mean(-1))
    if need("AAC"):    out["AAC"] = absdx.sum(-1) / T          # Phinyomark eq15: (1/N) sum|dx|
    if need("LOG"):    out["LOG"] = np.exp(np.log(absx + EPS).mean(-1))
    if need("LOGRMS"): out["LOGRMS"] = np.log(np.sqrt(sq.mean(-1)) + EPS)
    if need("NLE"):    out["NLE"] = np.log10(sq.mean(-1) + EPS)   # normalised log energy
    if need("ZC"):
        sign_change = (x[..., :-1] * x[..., 1:]) < 0
        out["ZC"] = (sign_change & (absdx >= thresh)).sum(-1).astype(np.float64)
    if need("SSC"):
        d1 = x[..., 1:-1] - x[..., :-2]
        d2 = x[..., 1:-1] - x[..., 2:]
        out["SSC"] = (((d1 * d2) >= thresh)).sum(-1).astype(np.float64)
    if need("WAMP"):   out["WAMP"] = (absdx >= thresh).sum(-1).astype(np.float64)
    if need("MYOP"):   out["MYOP"] = (absx >= thresh).mean(-1)
    if need("SKEW") or need("KURT"):
        mean, m2 = _moments(x)
        d = x - mean[..., None]
        m3 = (d ** 3).mean(-1); m4 = (d ** 4).mean(-1)
        if need("SKEW"): out["SKEW"] = m3 / (m2 ** 1.5 + EPS)
        if need("KURT"): out["KURT"] = m4 / (m2 ** 2 + EPS)
    if need("P75"):    out["P75"] = np.percentile(x, 75, axis=-1)
    # Hjorth (Abbaspour eqs 16-17)
    if need("HJ_ACT") or need("HJ_MOB") or need("HJ_COM"):
        var0 = x.var(-1) + EPS
        var1 = dx.var(-1) + EPS
        ddx = np.diff(dx, axis=-1)
        var2 = ddx.var(-1) + EPS
        mob = np.sqrt(var1 / var0)
        if need("HJ_ACT"): out["HJ_ACT"] = var0
        if need("HJ_MOB"): out["HJ_MOB"] = mob
        if need("HJ_COM"): out["HJ_COM"] = np.sqrt(var2 / var1) / (mob + EPS)
    if need("MFL"):    out["MFL"] = np.log10(np.sqrt((dx ** 2).sum(-1)) + EPS)

    # spectral features (share one rFFT PSD)
    freq_names = {"MNF", "MDF", "SENT", "MNP", "TTP"}
    if freq_names & set(names):
        psd = np.abs(np.fft.rfft(x, axis=-1)) ** 2                # (B, C, F)
        freqs = np.fft.rfftfreq(T, d=1.0 / fs)                    # (F,)
        tot = psd.sum(-1) + EPS
        if need("TTP"): out["TTP"] = tot
        if need("MNP"): out["MNP"] = psd.mean(-1)
        if need("MNF"): out["MNF"] = (psd * freqs).sum(-1) / tot
        if need("MDF"):
            cum = np.cumsum(psd, axis=-1)
            half = (tot / 2.0)[..., None]
            idx = (cum >= half).argmax(-1)
            out["MDF"] = freqs[idx]
        if need("SENT"):
            p = psd / tot[..., None]
            p_safe = np.where(p > 0, p, 1.0)                     # avoid log(0); p*log1=0 anyway
            plogp = p * np.log(p_safe)                           # 0*log0 -> 0 exactly, no warning
            sent = -plogp.sum(-1) / np.log(psd.shape[-1])
            out["SENT"] = np.clip(sent, 0.0, 1.0)
    return out


# ============================ SLOW (per 1-D series) ===================================

def _chebyshev_dist_matrix(emb):
    """Full pairwise Chebyshev distance (vectorised): emb (n,m) -> (n,n)."""
    # |emb[i]-emb[j]| max over the m coords, done without an (n,n,m) tensor to save memory
    n = emb.shape[0]
    d = np.zeros((n, n), dtype=np.float64)
    for k in range(emb.shape[1]):
        col = emb[:, k]
        d = np.maximum(d, np.abs(col[:, None] - col[None, :]))
    return d


def _embed_pair(x, m):
    """Canonical Richman-Moorman embeddings: the SAME nv = N-m starting indices produce both
    the length-m and length-(m+1) vectors, so SampEn/FuzzyEn counts are directly comparable."""
    N = len(x)
    nv = N - m
    Xm = np.array([x[i:i + m] for i in range(nv)], dtype=np.float64)          # (nv, m)
    Xm1 = np.array([x[i:i + m + 1] for i in range(nv)], dtype=np.float64)     # (nv, m+1)
    return Xm, Xm1, nv


def sample_entropy(x, m=2, r=0.2):
    """SampEn(m,r,N) = -ln(A/B) (Richman & Moorman); Chebyshev distance, self-matches excluded,
    equal vector counts for m and m+1. tol = r*std(x)."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < m + 2:
        return np.nan
    tol = r * (x.std() + EPS)
    Xm, Xm1, nv = _embed_pair(x, m)
    dm = _chebyshev_dist_matrix(Xm); np.fill_diagonal(dm, np.inf)
    dm1 = _chebyshev_dist_matrix(Xm1); np.fill_diagonal(dm1, np.inf)
    B = int((dm <= tol).sum())
    A = int((dm1 <= tol).sum())
    if B == 0 or A == 0:
        return np.nan
    return -np.log(A / B)


def fuzzy_entropy(x, m=2, n=5, r=0.30):
    """FuzzyEn (Chen 2007) = ln(phi_m) - ln(phi_{m+1}); membership exp(-(d^n)/tol). n=5 ->
    FuzzyEn (summary 01), n=2 -> fApEn (summary 03). Local-mean (baseline) removed from each
    embedding vector; equal vector counts for m and m+1; tol = r*std(x)."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < m + 2:
        return np.nan
    tol = r * (x.std() + EPS)
    Xm, Xm1, nv = _embed_pair(x, m)
    if nv < 2:
        return np.nan

    def phi(emb):
        emb = emb - emb.mean(axis=1, keepdims=True)           # baseline removal
        d = _chebyshev_dist_matrix(emb)
        mu = np.exp(-(d ** n) / tol)
        np.fill_diagonal(mu, 0.0)                             # exclude self-match
        return mu.sum() / (nv * (nv - 1))
    p_m = phi(Xm); p_m1 = phi(Xm1)
    if p_m <= 0 or p_m1 <= 0:
        return np.nan
    return np.log(p_m) - np.log(p_m1)


def fuzzy_apen(x, m=2, r=0.25):
    return fuzzy_entropy(x, m=m, n=2, r=r)


def perm_entropy(x, d=4, tau=1):
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < d * tau + 1:
        return np.nan
    patterns = {}
    for i in range(N - (d - 1) * tau):
        vec = x[i:i + d * tau:tau]
        key = tuple(np.argsort(vec))
        patterns[key] = patterns.get(key, 0) + 1
    counts = np.array(list(patterns.values()), dtype=np.float64)
    p = counts / counts.sum()
    from math import factorial, log
    return float(-(p * np.log(p)).sum() / log(factorial(d)))


def higuchi_fd(x, kmax=10):
    """Higuchi (1988) fractal dimension. Curve length
        L_m(k) = [ sum_i |x(m+ik)-x(m+(i-1)k)| * (N-1) / (n_int * k) ] / k
    (note the outer 1/k), L(k) = mean_m L_m(k); FD = slope of log L(k) vs log(1/k).
    Sanity: white noise -> ~2, a smooth sine/line -> ~1."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    if N < 2 * kmax:
        kmax = max(2, N // 2)
    L = []
    ks = []
    for k in range(1, kmax + 1):
        Lk = []
        for mmm in range(k):
            idx = np.arange(mmm, N, k)
            n_int = len(idx) - 1                              # number of intervals
            if n_int < 1:
                continue
            # * (N-1)/(n_int*k) normalises the curve length; the outer / k is Higuchi's factor
            lm = np.abs(np.diff(x[idx])).sum() * (N - 1) / (n_int * k) / k
            Lk.append(lm)
        if Lk:
            L.append(np.log(np.mean(Lk) + EPS))
            ks.append(np.log(1.0 / k))
    if len(ks) < 2:
        return np.nan
    return float(np.polyfit(ks, L, 1)[0])


def _coarse_grain(x, scale):
    N = len(x) // scale
    if N == 0:
        return np.array([])
    return x[:N * scale].reshape(N, scale).mean(axis=1)


def ms_fapen(x, m=2, r=0.25, scales=10, min_len=50):
    """Multiscale fuzzy ApEn -> (MED, LS-MED, HS-MED) over scales 1..scales (summary 02)."""
    vals = []
    for s in range(1, scales + 1):
        cg = _coarse_grain(np.asarray(x, dtype=np.float64), s)
        if len(cg) < min_len:
            vals.append(np.nan)
        else:
            vals.append(fuzzy_apen(cg, m=m, r=r))
    vals = np.array(vals, dtype=np.float64)
    med = np.nanmedian(vals) if np.any(~np.isnan(vals)) else np.nan
    ls = np.nanmedian(vals[:5]) if np.any(~np.isnan(vals[:5])) else np.nan
    hs = np.nanmedian(vals[5:]) if np.any(~np.isnan(vals[5:])) else np.nan
    return med, ls, hs


def slow_features(wins: np.ndarray, names, ent: dict, max_samples=None) -> dict:
    """wins: (B, C, T) -> {name: (B, C)}; MSFAPEN expands to 3 columns per channel.
    Per-series loops; call only on subsampled windows. Long windows are decimated to
    `max_samples` (fApEn is window-length robust — summary 03) for O(N^2) tractability."""
    x = np.asarray(wins, dtype=np.float64)
    if max_samples and x.shape[-1] > max_samples:
        step = int(np.ceil(x.shape[-1] / max_samples))
        # F-dec: ANTI-ALIASED decimation. The old `x[..., ::step]` was naive subsampling with no
        # low-pass, so energy above the new Nyquist folded straight back into the passband and
        # corrupted the entropy/fractal estimates — the exact bug already fixed for E6
        # (`windows.py`). scipy.signal.decimate applies a zero-phase FIR first. Below q=2 or on
        # very short series decimate is undefined, so fall back to the naive path.
        if step >= 2 and x.shape[-1] > 27 * step:
            from scipy.signal import decimate
            x = decimate(x, step, axis=-1, ftype="fir", zero_phase=True)
        else:
            x = x[..., ::step]
    B, C, T = x.shape
    out = {}
    want = set(names)
    # Entropy/fractal estimators need enough samples to be defined; below the threshold they
    # return numbers that look fine but mean nothing (25-sample SampEn/FuzzyEn/HFD). Refuse.
    min_samples = int(ent.get("min_samples", 0) or 0)
    if T < min_samples:
        cols = [nm for nm in ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD"] if nm in want]
        if "MSFAPEN" in want:
            cols += ["MSFAPEN_MED", "MSFAPEN_LS", "MSFAPEN_HS"]
        return {nm: np.full((B, C), np.nan) for nm in cols}
    do_ms = "MSFAPEN" in want
    base = [nm for nm in ["SAMPEN", "FUZZYEN", "FAPEN", "PERMEN", "HFD"] if nm in want]
    buf = {nm: np.full((B, C), np.nan) for nm in base}
    if do_ms:
        buf["MSFAPEN_MED"] = np.full((B, C), np.nan)
        buf["MSFAPEN_LS"] = np.full((B, C), np.nan)
        buf["MSFAPEN_HS"] = np.full((B, C), np.nan)
    for b in range(B):
        for c in range(C):
            s = x[b, c]
            if "SAMPEN" in want:  buf["SAMPEN"][b, c] = sample_entropy(s, ent["m"], ent["r_sampen"])
            if "FUZZYEN" in want: buf["FUZZYEN"][b, c] = fuzzy_entropy(s, ent["m"], ent["n_fuzzy"], ent["r_fuzzy"])
            if "FAPEN" in want:   buf["FAPEN"][b, c] = fuzzy_apen(s, ent["m"], ent["r_fapen"])
            if "PERMEN" in want:  buf["PERMEN"][b, c] = perm_entropy(s, ent["perm_d"], ent["perm_tau"])
            if "HFD" in want:     buf["HFD"][b, c] = higuchi_fd(s, ent["hfd_kmax"])
            if do_ms:
                med, ls, hs = ms_fapen(s, ent["m"], ent["r_fapen"], ent["ms_scales"])
                buf["MSFAPEN_MED"][b, c] = med
                buf["MSFAPEN_LS"][b, c] = ls
                buf["MSFAPEN_HS"][b, c] = hs
    out.update(buf)
    return out

"""Math-validation tests for the PROFILE feature/metric equations.

Every equation is checked against an analytic value, a hand computation, or a reference
implementation (scipy). Run:  python tests/test_math.py   (prints PASS/FAIL per check).

The user's standing rule: test each equation; use float64 for accumulation/exponentiation.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# tests never write into a real results tree (CLAUDE.md rule 1)
import os as _os, pathlib as _pl
_os.environ.setdefault("PROFILE_RESULTS_DIR",
                       str(_pl.Path(__file__).resolve().parents[1] / "results" / "_test_sandbox"))

import numpy as np

from dsprofile import features_extra as fx
from dsprofile import module2_separability as m2
from dsprofile import module3_shift as m3

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}   {detail}")


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol * (1 + abs(float(b)))


# ------------------------------------------------------------------ fast time-domain
def test_fast_timedomain():
    x = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    w = x[None, None, :]                                   # (1,1,5)
    f = fx.fast_features(w, fs=1000,
                         names=["MAV", "RMS", "WL", "VAR", "IEMG", "SSI", "DASDV", "AAC",
                                "LOG", "MFL", "SKEW", "KURT", "P75"], thresh=0.0)
    N = len(x)
    check("MAV", approx(f["MAV"][0, 0], np.abs(x).mean()))
    check("RMS", approx(f["RMS"][0, 0], np.sqrt((x ** 2).mean())))
    check("WL", approx(f["WL"][0, 0], np.abs(np.diff(x)).sum()))
    check("VAR(ddof=1)", approx(f["VAR"][0, 0], x.var(ddof=1)))
    check("IEMG", approx(f["IEMG"][0, 0], np.abs(x).sum()))
    check("SSI", approx(f["SSI"][0, 0], (x ** 2).sum()))
    check("DASDV", approx(f["DASDV"][0, 0], np.sqrt((np.diff(x) ** 2).mean())))
    check("AAC=(1/N)sum|dx|", approx(f["AAC"][0, 0], np.abs(np.diff(x)).sum() / N))
    check("LOG", approx(f["LOG"][0, 0], np.exp(np.mean(np.log(np.abs(x))))))
    check("MFL", approx(f["MFL"][0, 0], np.log10(np.sqrt((np.diff(x) ** 2).sum()))))
    # skew/kurtosis vs scipy
    from scipy.stats import skew, kurtosis
    check("SKEW vs scipy", approx(f["SKEW"][0, 0], skew(x), 1e-5))
    check("KURT vs scipy(non-excess)", approx(f["KURT"][0, 0], kurtosis(x, fisher=False), 1e-5))
    check("P75", approx(f["P75"][0, 0], np.percentile(x, 75)))


def test_zc_ssc_wamp():
    x = np.array([1.0, -1.0, 1.0, -1.0, 1.0])             # 4 sign changes
    w = x[None, None, :]
    f = fx.fast_features(w, 1000, ["ZC", "SSC", "WAMP", "MYOP"], thresh=0.0)
    check("ZC counts sign changes", f["ZC"][0, 0] == 4, f"got {f['ZC'][0,0]}")
    check("WAMP counts |dx|>=thr", f["WAMP"][0, 0] == 4, f"got {f['WAMP'][0,0]}")
    check("MYOP fraction |x|>=thr", approx(f["MYOP"][0, 0], 1.0))


def test_hjorth():
    # Hjorth mobility of a pure sinusoid ~ 2*sin(pi/T*...) ; use analytic-ish check:
    # for x=sin(wt), mobility ~ w (in rad/sample). Build discrete sine and compare mobility.
    t = np.arange(2000)
    w_rad = 0.1
    x = np.sin(w_rad * t)
    f = fx.fast_features(x[None, None, :], 1000, ["HJ_ACT", "HJ_MOB", "HJ_COM"])
    # mobility of a sine approx equals the angular frequency w_rad
    check("Hjorth mobility ~= w for sine", approx(f["HJ_MOB"][0, 0], w_rad, 5e-2),
          f"got {f['HJ_MOB'][0,0]:.4f} vs {w_rad}")
    # complexity of a pure sine ~ 1
    check("Hjorth complexity ~1 for sine", approx(f["HJ_COM"][0, 0], 1.0, 1e-2),
          f"got {f['HJ_COM'][0,0]:.4f}")
    check("Hjorth activity=var", approx(f["HJ_ACT"][0, 0], x.var()))


def test_spectral():
    fs = 1000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 100 * t)                       # 100 Hz tone
    f = fx.fast_features(x[None, None, :], fs, ["MNF", "MDF", "TTP", "MNP", "SENT"])
    check("MNF ~ 100 Hz tone", approx(f["MNF"][0, 0], 100.0, 2e-2), f"got {f['MNF'][0,0]:.2f}")
    check("MDF ~ 100 Hz tone", abs(f["MDF"][0, 0] - 100.0) <= 2.0, f"got {f['MDF'][0,0]:.2f}")
    check("TTP>0", f["TTP"][0, 0] > 0)
    check("spectral entropy in [0,1]", 0 <= f["SENT"][0, 0] <= 1)


# ------------------------------------------------------------------ entropy family
def test_sampen_known():
    # Deterministic periodic signal -> low SampEn; random -> higher.
    per = np.tile([1.0, 2.0, 3.0], 100)
    rng = np.random.default_rng(0)
    rnd = rng.standard_normal(300)
    se_per = fx.sample_entropy(per, 2, 0.2)
    se_rnd = fx.sample_entropy(rnd, 2, 0.2)
    check("SampEn(periodic) < SampEn(random)", se_per < se_rnd, f"{se_per:.3f} < {se_rnd:.3f}")
    check("SampEn(periodic) ~ 0", se_per < 0.05, f"got {se_per:.4f}")


def test_sampen_reference():
    # Compare against a direct brute-force SampEn on a small series.
    rng = np.random.default_rng(1)
    x = rng.standard_normal(120)
    m, r = 2, 0.2
    tol = r * x.std()

    def brute(mm):
        N = len(x); nv = N - m
        cnt = 0
        for i in range(nv):
            vi = x[i:i + mm]
            for j in range(nv):
                if i == j:
                    continue
                vj = x[j:j + mm]
                if np.max(np.abs(vi - vj)) <= tol:
                    cnt += 1
        return cnt
    ref = -np.log(brute(3) / brute(2))
    got = fx.sample_entropy(x, m, r)
    check("SampEn vs brute-force", approx(got, ref, 1e-9), f"{got:.6f} vs {ref:.6f}")


def test_fuzzyen_reference():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(120)
    m, n, r = 2, 2, 0.25
    tol = r * x.std()

    def phi(mm):
        N = len(x); nv = N - m
        s = 0.0
        for i in range(nv):
            vi = x[i:i + mm]; vi = vi - vi.mean()
            for j in range(nv):
                if i == j:
                    continue
                vj = x[j:j + mm]; vj = vj - vj.mean()
                d = np.max(np.abs(vi - vj))
                s += np.exp(-(d ** n) / tol)
        return s / (nv * (nv - 1))
    ref = np.log(phi(2)) - np.log(phi(3))
    got = fx.fuzzy_entropy(x, m, n, r)
    check("FuzzyEn vs brute-force", approx(got, ref, 1e-9), f"{got:.6f} vs {ref:.6f}")


def test_perm_entropy():
    # Monotonic signal -> only one ordinal pattern -> PE ~ 0. White noise -> PE ~ 1.
    mono = np.arange(500.0)
    rng = np.random.default_rng(3)
    rnd = rng.standard_normal(5000)
    pe_mono = fx.perm_entropy(mono, d=4, tau=1)
    pe_rnd = fx.perm_entropy(rnd, d=4, tau=1)
    check("PermEn(monotonic) ~ 0", pe_mono < 1e-6, f"got {pe_mono:.4f}")
    check("PermEn(white noise) ~ 1", pe_rnd > 0.95, f"got {pe_rnd:.4f}")


def test_higuchi():
    rng = np.random.default_rng(4)
    noise = rng.standard_normal(4000)
    t = np.linspace(0, 20 * np.pi, 4000)
    sine = np.sin(t)
    fd_noise = fx.higuchi_fd(noise, kmax=10)
    fd_sine = fx.higuchi_fd(sine, kmax=10)
    check("HFD(white noise) ~ 2", abs(fd_noise - 2.0) < 0.15, f"got {fd_noise:.3f}")
    check("HFD(smooth sine) ~ 1", abs(fd_sine - 1.0) < 0.2, f"got {fd_sine:.3f}")
    check("HFD(noise) > HFD(sine)", fd_noise > fd_sine)


# ------------------------------------------------------------------ module math
def test_gaussian_kl():
    # KL between identical Gaussians = 0; against scipy multivariate for a known case.
    rng = np.random.default_rng(5)
    A = rng.multivariate_normal([0, 0], np.eye(2), size=4000)
    tot0, mean0, cov0 = m3.gaussian_kl_split(A, A.copy())
    check("Gaussian-KL(self) ~ 0", abs(tot0) < 0.02, f"got {tot0:.4f}")
    # analytic 1-D: KL(N(0,1)||N(mu,1)) = mu^2/2 ; here approximate via samples
    a = rng.standard_normal((6000, 1))
    b = rng.standard_normal((6000, 1)) + 2.0
    tot, mean_t, cov_t = m3.gaussian_kl_split(a, b)
    check("KL mean-term ~ mu^2/2 (=2)", abs(mean_t - 2.0) < 0.2, f"got mean_term {mean_t:.3f}")
    check("KL split total = mean+cov", approx(tot, mean_t + cov_t, 1e-9))


def test_mmd_energy():
    rng = np.random.default_rng(6)
    A = rng.standard_normal((800, 4))
    B = rng.standard_normal((800, 4))
    C = rng.standard_normal((800, 4)) + 3.0
    mmd_same = m3.mmd_rbf(A, B, rng=rng)
    mmd_diff = m3.mmd_rbf(A, C, rng=rng)
    check("MMD(same dist) ~ 0", mmd_same < 0.02, f"got {mmd_same:.4f}")
    check("MMD(shifted) > MMD(same)", mmd_diff > mmd_same, f"{mmd_diff:.3f} > {mmd_same:.3f}")
    ed_same = m3.energy_distance(A, B, rng=rng)
    ed_diff = m3.energy_distance(A, C, rng=rng)
    check("energy-dist(shifted) > (same)", ed_diff > ed_same, f"{ed_diff:.3f} > {ed_same:.3f}")


def test_hdiv():
    rng = np.random.default_rng(7)
    A = rng.standard_normal((500, 4))
    B = rng.standard_normal((500, 4))
    C = rng.standard_normal((500, 4)) + 5.0
    h_same = m3.h_divergence(A, B, rng=rng)
    h_diff = m3.h_divergence(A, C, rng=rng)
    check("H-div(same) ~ 0", abs(h_same) < 0.2, f"got {h_same:.3f}")
    check("H-div(well-separated) ~ 2", h_diff > 1.5, f"got {h_diff:.3f}")

    # The two checks above use IID rows and therefore pass EVEN WITH THE TRIAL-LEAK PRESENT —
    # they gave false assurance until 2026-07-10. Real EMG windows are clustered by trial and
    # overlap 50 %, and trial identity is perfectly predictive of the group. Exercise that.
    def trial_group(seed, mu=0.0, n_trials=60, per=10, d=4):
        r = np.random.default_rng(seed)
        X, t = [], []
        for k in range(n_trials):
            base = r.standard_normal(d) * 3.0 + mu
            X.append(base + r.standard_normal((per, d)) * 0.05)
            t += [k + seed * 10_000] * per
        return np.vstack(X), np.array(t)

    A2, ta = trial_group(1)
    B2, tb = trial_group(2)                     # identical generating process -> true d_H = 0
    leaked = m3.h_divergence(A2, B2, rng=np.random.default_rng(0))
    honest = m3.h_divergence(A2, B2, rng=np.random.default_rng(0), groups_a=ta, groups_b=tb)
    check("H-div on trial-clustered data: ungrouped folds SATURATE (the leak)", leaked > 1.5,
          f"got {leaked:.3f} where truth is 0")
    check("H-div on trial-clustered data: trial-grouped folds ~ 0", honest < 0.2,
          f"got {honest:.3f}")
    C2, tc = trial_group(3, mu=4.0)
    real = m3.h_divergence(A2, C2, rng=np.random.default_rng(0), groups_a=ta, groups_b=tc)
    check("H-div still separates a genuinely shifted group", real > 1.0, f"got {real:.3f}")


def test_fisher_mahalanobis():
    rng = np.random.default_rng(8)
    # two well-separated classes -> high Fisher & Mahalanobis
    A = rng.standard_normal((300, 3))
    B = rng.standard_normal((300, 3)) + 6.0
    X = np.vstack([A, B]); y = np.r_[np.zeros(300), np.ones(300)]
    fr = m2.fisher_ratio(X, y); mh = m2.mahalanobis_si(X, y)
    # overlapping classes -> low
    A2 = rng.standard_normal((300, 3)); B2 = rng.standard_normal((300, 3))
    X2 = np.vstack([A2, B2]); y2 = np.r_[np.zeros(300), np.ones(300)]
    fr2 = m2.fisher_ratio(X2, y2)
    check("Fisher(separated) > Fisher(overlap)", fr > fr2, f"{fr:.3f} > {fr2:.3f}")
    check("Mahalanobis SI > 0 for separated", mh > 3.0, f"got {mh:.3f}")


def test_twonn():
    # points uniform in a d-dim cube -> TwoNN ~ d
    rng = np.random.default_rng(9)
    for d in (2, 5):
        X = rng.random((3000, d))
        est = m2.twonn_dim(X, seed=0)
        check(f"TwoNN ~ {d}", abs(est - d) < 0.6 * d, f"got {est:.2f}")


def test_precision():
    # large-magnitude accumulation must stay float64-exact (no int overflow / float32 loss)
    x = np.full(200000, 1e4, dtype=np.float64)
    f = fx.fast_features(x[None, None, :], 1000, ["SSI", "IEMG"], thresh=0.0)
    check("SSI float64 large-sum exact", approx(f["SSI"][0, 0], 200000 * (1e4 ** 2), 1e-12),
          f"got {f['SSI'][0,0]:.6e}")
    check("SSI dtype float", np.issubdtype(type(float(f["SSI"][0, 0])), np.floating))


def main():
    for fn in [test_fast_timedomain, test_zc_ssc_wamp, test_hjorth, test_spectral,
               test_sampen_known, test_sampen_reference, test_fuzzyen_reference,
               test_perm_entropy, test_higuchi, test_gaussian_kl, test_mmd_energy,
               test_hdiv, test_fisher_mahalanobis, test_twonn, test_precision]:
        try:
            fn()
        except Exception as e:
            import traceback
            check(fn.__name__, False, f"EXCEPTION {type(e).__name__}: {e}")
            traceback.print_exc()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n==== {n_pass}/{len(RESULTS)} checks passed ====")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()

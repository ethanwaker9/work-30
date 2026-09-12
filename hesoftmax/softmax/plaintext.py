import numpy as np
from .common import fit_coeffs, softmax_ref
from ..polyapprox.chebyshev import cheb_eval
from . import baselines as B
from . import ours as O

DMAX = 512


def _P(c, a, b, x):
    return cheb_eval(c, a, b, x)


def acs_pt(X, cfg):
    n, spread, alpha, k = cfg["n"], cfg["spread"], cfg["alpha"], cfg["k"]
    tol = 2.0 ** (-(alpha + np.log2(n)))
    z = X - X.mean(axis=1, keepdims=True)
    c = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -spread, spread, tol, d_max=O.DMAX,
                   key=("exp", n, spread, k, alpha))
    y = _P(c, -spread, spread, z)
    for idx, j in enumerate(range(k, 0, -1)):
        hi, dj, t = cfg["stages"][idx]
        cc = fit_coeffs(B._isqrt, 0.85, hi, t, relative=True, d_max=O.DMAX, log_grid=True,
                        key=("stage", round(hi, 6), t))
        u = (y ** 2).sum(axis=1, keepdims=True) / n
        y = (y * _P(cc, 0.85, hi, u)) ** 2
    cf = fit_coeffs(O._inv, O.FIN_LO, O.FIN_HI, 2.0 ** (-alpha), relative=True, d_max=O.DMAX,
                    key=("fin", alpha))
    u = y.sum(axis=1, keepdims=True) / n
    return y * _P(cf, O.FIN_LO, O.FIN_HI, np.clip(u, O.FIN_LO, O.FIN_HI)) / n


def cho_pt(X, cfg, anchor):
    n, k, alpha, spread = cfg["n"], cfg["k"], cfg["alpha"], cfg["spread"]
    tol = 2.0 ** (-(alpha + np.log2(n)))
    y = X - anchor
    c = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -spread, 0.0, tol, d_max=DMAX,
                   key=("choexp", n, spread, k, alpha))
    y = _P(c, -spread, 0.0, y)
    for j in range(k):
        if j == 0:
            lo, hi = cfg["b1"]
            t = 2.0 ** (-alpha) if k == 1 else 0.05
        else:
            lo, hi = cfg["b2"]
            t = 2.0 ** (-alpha) if j == k - 1 else 0.05
        cc = fit_coeffs(B._isqrt, lo, hi, t, relative=True, d_max=DMAX, log_grid=True,
                        key=("choinv", round(lo, 9), round(hi, 6), t))
        u = (y ** 2).sum(axis=1, keepdims=True)
        y = (y * _P(cc, lo, hi, u)) ** 2
    return y


def _gold(s, lo, hi, alpha):
    f0, it = B.gold_params(lo, hi, alpha)
    nn = f0 * np.ones_like(s)
    dd = f0 * s
    for _ in range(it):
        f = 2 - dd
        nn = nn * f
        dd = dd * f
    return nn


def thor_pt(X, cfg, mid):
    n, alpha, spread = cfg["n"], cfg["alpha"], cfg["spread"]
    d1, d2 = cfg["d1"], cfg["d2"]
    half = spread / 2.0
    tol = 2.0 ** (-(alpha + np.log2(n)))
    y = X - mid
    c = fit_coeffs(lambda u: np.exp(u / (d1 * d2)), -half, half, tol, d_max=DMAX,
                   key=("thorexp", n, round(half, 6), d1, d2, alpha))
    y = _P(c, -half, half, y)
    for _ in range(int(np.log2(d1))):
        y = y ** 2
    lo = float(np.exp(-half / d2)) * 0.85
    hi = float(np.exp(half / d2)) * 1.15
    for j in range(int(np.log2(d2)) + 1):
        s = y.sum(axis=1, keepdims=True) / n
        y = y * _gold(s, lo, hi, alpha) / n
        if j < int(np.log2(d2)):
            y = y ** 2
            lo, hi = 0.85 / n ** 2, 1.15 / n
    return y


def nexus_pt(X, cfg, anchor):
    n, r = cfg["n"], cfg["r"]
    y = 1.0 + (X - anchor) / (2.0 ** r)
    for _ in range(r):
        y = y ** 2
    s = y.sum(axis=1, keepdims=True) / n
    return y * _gold(s, 0.85 / n, 1.15, cfg["alpha"]) / n


def quad_pt(X, cfg, anchor):
    n = cfg["n"]
    y = (1.0 + X / max(cfg["spread"], 1.0)) ** 2
    s = y.sum(axis=1, keepdims=True) / n
    return y * _gold(s, 0.85 / n, 4.6, cfg["alpha"]) / n


def _sign_pt(x, cfg):
    cf = B._poly_cheb(B.SIGN_F)
    cg = B._poly_cheb(B.SIGN_G)
    s = x
    for _ in range(cfg["dg"]):
        s = _P(cg, -1.0, 1.0, s)
    for _ in range(cfg["df"]):
        s = _P(cf, -1.0, 1.0, s)
    return s


def hetal_max_pt(X, cfg, bound):
    cur = X / bound
    m = X.shape[1]
    step = 1
    for _ in range(cfg["rounds"]):
        other = np.roll(cur, -step, axis=1)
        a = 0.5 * (cur + other)
        d = 0.5 * (cur - other)
        cur = a + d * _sign_pt(d, cfg)
        step *= 2
    return cur * bound


def hetal_pt(X, cfg, bound):
    n, alpha, spread = cfg["n"], cfg["alpha"], cfg["spread"]
    mx = hetal_max_pt(X, cfg, bound)
    c = fit_coeffs(np.exp, -spread, 0.0, 2.0 ** (-(alpha + np.log2(n))), d_max=DMAX,
                   key=("hexp", n, spread, alpha))
    y = _P(c, -spread, 0.0, np.clip(X - mx, -spread, 0.0))
    s = y.sum(axis=1, keepdims=True) / n
    return y * _gold(s, 0.85 / n, 1.15, alpha) / n

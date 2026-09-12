import numpy as np
from .common import fit_coeffs
from ..polyapprox.chebyshev import cheb_eval
from . import baselines as B
from . import ours as O
from . import plaintext as PT


def _P(c, a, b, x):
    return cheb_eval(c, a, b, x)


def _counts(mask):
    return mask.sum(axis=-1, keepdims=True).astype(np.float64)


def acs_masked(att, mask, cfg):
    n, spread, alpha, k = cfg["n"], cfg["spread"], cfg["alpha"], cfg["k"]
    nv = _counts(mask)
    tol = 2.0 ** (-(alpha + np.log2(n)))
    mu = np.where(mask, att, 0.0).sum(axis=-1, keepdims=True) / nv
    z = np.where(mask, att - mu, 0.0)
    c = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -spread, spread, tol, d_max=O.DMAX,
                   key=("exp", n, spread, k, alpha))
    y = np.where(mask, _P(c, -spread, spread, np.clip(z, -spread, spread)), 0.0)
    for idx, j in enumerate(range(k, 0, -1)):
        hi, dj, t = cfg["stages"][idx]
        cc = fit_coeffs(B._isqrt, 0.85, hi, t, relative=True, d_max=O.DMAX, log_grid=True,
                        key=("stage", round(hi, 6), t))
        u = (y ** 2).sum(axis=-1, keepdims=True) / nv
        y = (y * _P(cc, 0.85, hi, np.clip(u, 0.85, hi))) ** 2
    cf = fit_coeffs(O._inv, O.FIN_LO, O.FIN_HI, 2.0 ** (-alpha), relative=True, d_max=O.DMAX,
                    key=("fin", alpha))
    u = y.sum(axis=-1, keepdims=True) / nv
    return y * _P(cf, O.FIN_LO, O.FIN_HI, np.clip(u, O.FIN_LO, O.FIN_HI)) / nv


def cho_masked(att, mask, cfg, anchor):
    n, k, alpha, spread = cfg["n"], cfg["k"], cfg["alpha"], cfg["spread"]
    nv = _counts(mask)
    tol = 2.0 ** (-(alpha + np.log2(n)))
    c = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -spread, 0.0, tol, d_max=B.DMAX,
                   key=("choexp", n, spread, k, alpha))
    z = np.clip(att - anchor, -spread, 0.0)
    y = np.where(mask, _P(c, -spread, 0.0, z), 0.0)
    for j in range(k):
        if j == 0:
            lo, hi = cfg["b1"]
            t = 2.0 ** (-alpha) if k == 1 else 0.05
        else:
            lo, hi = cfg["b2"]
            t = 2.0 ** (-alpha) if j == k - 1 else 0.05
        cc = fit_coeffs(B._isqrt, lo, hi, t, relative=True, d_max=B.DMAX, log_grid=True,
                        key=("choinv", round(lo, 9), round(hi, 6), t))
        u = (y ** 2).sum(axis=-1, keepdims=True)
        y = (y * _P(cc, lo, hi, np.clip(u, lo, hi))) ** 2
    return y


def nexus_masked(att, mask, cfg, anchor):
    n, r = cfg["n"], cfg["r"]
    nv = _counts(mask)
    y = np.where(mask, (1.0 + (att - anchor) / (2.0 ** r)), 0.0)
    for _ in range(r):
        y = y ** 2
    s = y.sum(axis=-1, keepdims=True) / nv
    return y * PT._gold(s, 0.85 / n, 1.15, cfg["alpha"]) / nv


def thor_masked(att, mask, cfg, mid):
    n, alpha, spread = cfg["n"], cfg["alpha"], cfg["spread"]
    d1, d2 = cfg["d1"], cfg["d2"]
    nv = _counts(mask)
    half = spread / 2.0
    tol = 2.0 ** (-(alpha + np.log2(n)))
    c = fit_coeffs(lambda u: np.exp(u / (d1 * d2)), -half, half, tol, d_max=B.DMAX,
                   key=("thorexp", n, round(half, 6), d1, d2, alpha))
    y = np.where(mask, _P(c, -half, half, np.clip(att - mid, -half, half)), 0.0)
    for _ in range(int(np.log2(d1))):
        y = y ** 2
    lo = float(np.exp(-half / d2)) * 0.85
    hi = float(np.exp(half / d2)) * 1.15
    for j in range(int(np.log2(d2)) + 1):
        s = y.sum(axis=-1, keepdims=True) / nv
        y = y * PT._gold(s, lo, hi, alpha) / nv
        if j < int(np.log2(d2)):
            y = y ** 2
            lo, hi = 0.85 / n ** 2, 1.15
    return y


def quad_masked(att, mask, cfg, anchor):
    n = cfg["n"]
    nv = _counts(mask)
    y = np.where(mask, (1.0 + att / max(cfg["spread"], 1.0)) ** 2, 0.0)
    s = y.sum(axis=-1, keepdims=True) / nv
    return y * PT._gold(s, 0.85 / n, 4.6, cfg["alpha"]) / nv


def hetal_masked(att, mask, cfg, bound):
    n, alpha, spread = cfg["n"], cfg["alpha"], cfg["spread"]
    nv = _counts(mask)
    big = np.where(mask, att, -bound)
    cur = big / bound
    step = 1
    for _ in range(cfg["rounds"]):
        other = np.roll(cur, -step, axis=-1)
        a = 0.5 * (cur + other)
        d = 0.5 * (cur - other)
        cur = a + d * PT._sign_pt(np.clip(d, -1, 1), cfg)
        step *= 2
    mx = cur[..., :1] * bound
    c = fit_coeffs(np.exp, -spread, 0.0, 2.0 ** (-(alpha + np.log2(n))), d_max=B.DMAX,
                   key=("hexp", n, spread, alpha))
    y = np.where(mask, _P(c, -spread, 0.0, np.clip(att - mx, -spread, 0.0)), 0.0)
    s = y.sum(axis=-1, keepdims=True) / nv
    return y * PT._gold(s, 0.85 / n, 1.15, alpha) / nv

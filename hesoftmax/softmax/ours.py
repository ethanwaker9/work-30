import numpy as np
from .common import fit_coeffs
from ..polyapprox.chebyshev import poly_depth, min_degree

CRUDE = 0.05
DMAX = 384


def _isqrt(u):
    return u ** -0.5


def _inv(u):
    return 1.0 / u


FIN_LO = (1.0 - CRUDE) ** 2 * 0.98
FIN_HI = (1.0 + CRUDE) ** 2 * 1.02


def _deg(lo, hi, tol):
    return min_degree(_isqrt, lo, hi, tol, relative=True, d_max=DMAX, log_grid=True)


HI_CAP = 1e5


def band_of(n, spread, j, k, slack=1.15):
    e = 2.0 * spread / (2.0 ** j) if j == k else spread / (2.0 ** j)
    hi = float(np.exp(e)) if e < 60 else float("inf")
    if j != k:
        hi = min(1.05 * n, hi)
    return max(hi * slack, 1.1)


def ks_est(d):
    return 2.0 * np.sqrt(d + 1) + np.log2(d + 1) if d >= 1 else 0.0


def plan(n, spread, alpha=16, k_max=12, w_ks=0.06):
    tol_exp = 2.0 ** (-(alpha + np.log2(n)))
    d_fin = min_degree(_inv, FIN_LO, FIN_HI, 2.0 ** (-alpha), relative=True, d_max=DMAX)
    best = None
    for k in range(1, k_max + 1):
        de = min_degree(lambda u, kk=k: np.exp(u / (2.0 ** kk)), -spread, spread, tol_exp,
                        d_max=DMAX)
        if de is None:
            continue
        depth = poly_depth(de)
        cost = float(depth) + w_ks * ks_est(de)
        stages, ok = [], True
        for j in range(k, 0, -1):
            hi = band_of(n, spread, j, k)
            tol = CRUDE
            dj = None if hi > HI_CAP else _deg(0.85, hi, tol)
            if dj is None:
                ok = False
                break
            stages.append((hi, dj, tol))
            depth += 1 + poly_depth(dj) + 2
            cost += 1 + poly_depth(dj) + 2 + w_ks * (ks_est(dj) + 2)
        if not ok:
            continue
        depth += poly_depth(d_fin) + 1
        cost += poly_depth(d_fin) + 1 + w_ks * (ks_est(d_fin) + 1)
        cand = (cost, depth, k, spread / 2.0 ** k, de, stages)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        raise ValueError("no feasible plan")
    cost, depth, k, w, de, stages = best
    return {"depth": depth, "k": k, "window": w, "deg_exp": de, "stages": stages,
            "deg_fin": d_fin, "n": n, "spread": spread, "alpha": alpha,
            "rot": int(np.log2(n)) * (k + 2),
            "name": "ACS"}


def acs_softmax(rt, ct, cfg):
    ctx = rt.ctx
    n, spread, alpha, k = cfg["n"], cfg["spread"], cfg["alpha"], cfg["k"]
    tol_exp = 2.0 ** (-(alpha + np.log2(n)))
    mu = ctx.set_mag(ctx.mul_const(rt.block_sum(ct), 1.0 / n), spread)
    z = ctx.set_mag(ctx.sub(ct, mu), spread)
    c_exp = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -spread, spread, tol_exp, d_max=DMAX,
                       key=("exp", n, spread, k, alpha))
    z = rt.need(z, poly_depth(len(c_exp) - 1) + 1)
    y = ctx.set_mag(rt.ev.evaluate(z, c_exp, -spread, spread), float(np.exp(spread / 2.0 ** k)))
    for idx, j in enumerate(range(k, 0, -1)):
        hi, dj, tol = cfg["stages"][idx]
        cj = fit_coeffs(_isqrt, 0.85, hi, tol, relative=True, d_max=DMAX, log_grid=True,
                        key=("stage", round(hi, 6), tol))
        y = rt.need(y, 1 + poly_depth(dj) + 2)
        u = ctx.set_mag(ctx.mul_const(rt.block_sum(ctx.square(y)), 1.0 / n), hi)
        lam = ctx.set_mag(rt.ev.evaluate(u, cj, 0.85, hi, out_mag=1.09), 1.09)
        y = ctx.set_mag(ctx.square(ctx.set_mag(ctx.mul(y, lam), np.sqrt(n) * 1.1)), n * 1.2)
    c_fin = fit_coeffs(_inv, FIN_LO, FIN_HI, 2.0 ** (-alpha), relative=True, d_max=DMAX,
                       key=("fin", alpha))
    y = rt.need(y, poly_depth(len(c_fin) - 1) + 1)
    u = ctx.set_mag(ctx.mul_const(rt.block_sum(y), 1.0 / n), FIN_HI)
    lam = ctx.set_mag(rt.ev.evaluate(u, c_fin, FIN_LO, FIN_HI, out_mag=1.0 / FIN_LO),
                      1.0 / FIN_LO)
    return ctx.mul_const(ctx.mul(y, lam), 1.0 / n), cfg

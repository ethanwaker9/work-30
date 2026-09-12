import numpy as np
from .common import fit_coeffs
from ..polyapprox.chebyshev import poly_depth, min_degree, cheb_coeffs

DMAX = 512


def _isqrt(u):
    return u ** -0.5


def gold_params(lo, hi, alpha):
    f0 = 2.0 / (lo + hi)
    e0 = (hi - lo) / (hi + lo)
    if e0 <= 0:
        return f0, 1
    it = int(np.ceil(np.log2(max(alpha * np.log(2) / (-np.log(max(e0, 1e-12))), 1.0)))) + 1
    return f0, max(it, 1)


def goldschmidt_inv(rt, ct, lo, hi, alpha):
    ctx = rt.ctx
    f0, iters = gold_params(lo, hi, alpha)
    ct = rt.need(ct, 2 * iters + 2)
    d = ctx.mul_const(ct, float(f0))
    f = ctx.add_const(ctx.mul_const(d, -1.0), 2.0)
    out = ctx.mul_const(f, float(f0))
    d = ctx.mul(d, f)
    for _ in range(iters - 1):
        f = ctx.add_const(ctx.mul_const(d, -1.0), 2.0)
        out = ctx.mul(out, f)
        d = ctx.mul(d, f)
    return out, iters


def cho_plan(n, spread, alpha=16):
    k = max(int(np.ceil(np.log2(spread) - np.log2(np.log(n)))), 1)
    w = spread / (2.0 ** k)
    de = min_degree(np.exp, -w, 0.0, 2.0 ** (-(alpha + np.log2(n))), d_max=DMAX)
    b1 = (float(n) * np.exp(-2 * w) * 0.85, float(n) * 1.15)
    b2 = (0.85 / n, 1.15)
    d_first = min_degree(_isqrt, b1[0], b1[1], 0.05, relative=True, d_max=DMAX, log_grid=True)
    d_mid = min_degree(_isqrt, b2[0], b2[1], 0.05, relative=True, d_max=DMAX, log_grid=True)
    d_last = min_degree(_isqrt, b2[0], b2[1], 2.0 ** (-alpha), relative=True, d_max=DMAX,
                        log_grid=True)
    d_first_acc = min_degree(_isqrt, b1[0], b1[1], 2.0 ** (-alpha), relative=True, d_max=DMAX,
                             log_grid=True)
    depth = poly_depth(de)
    for j in range(k):
        if j == 0:
            d = d_first_acc if k == 1 else d_first
        else:
            d = d_last if j == k - 1 else d_mid
        depth += 1 + poly_depth(d) + 3
    return {"k": k, "window": w, "deg_exp": de, "deg_first": d_first, "deg_mid": d_mid,
            "deg_last": d_last, "depth": depth, "n": n, "spread": spread, "alpha": alpha,
            "b1": b1, "b2": b2, "name": "Cho24",
            "rot": (2 * int(np.log2(n))) * k}


def cho_softmax(rt, ct, cfg, anchor):
    ctx = rt.ctx
    n, k, alpha = cfg["n"], cfg["k"], cfg["alpha"]
    y = ctx.add_const(ct, -float(anchor))
    c_exp = fit_coeffs(lambda u: np.exp(u / (2.0 ** k)), -cfg["spread"], 0.0,
                       2.0 ** (-(alpha + np.log2(n))), d_max=DMAX,
                       key=("choexp", n, cfg["spread"], k, alpha))
    y = rt.need(y, poly_depth(len(c_exp) - 1) + 1)
    y = rt.ev.evaluate(y, c_exp, -cfg["spread"], 0.0)
    for j in range(k):
        if j == 0:
            lo, hi = cfg["b1"]
            tol = 2.0 ** (-alpha) if k == 1 else 0.05
        else:
            lo, hi = cfg["b2"]
            tol = 2.0 ** (-alpha) if j == k - 1 else 0.05
        c = fit_coeffs(_isqrt, lo, hi, tol, relative=True, d_max=DMAX, log_grid=True,
                       key=("choinv", round(lo, 9), round(hi, 6), tol))
        y = rt.need(y, 1 + poly_depth(len(c) - 1) + 3)
        u = rt.block_sum(ctx.square(y))
        lam = rt.ev.evaluate(u, c, lo, hi)
        lam = rt.mask_broadcast(lam)
        y = ctx.square(ctx.mul(y, lam))
    return y, cfg


def thor_plan(n, spread, alpha=16):
    d2 = int(2 ** max(np.ceil(np.log2(max(spread / 4.0, 1.0))), 0))
    d1 = 4
    w = spread / (d1 * d2)
    de = min_degree(lambda u: np.exp(u / (d1 * d2)), -spread / 2.0, spread / 2.0,
                    2.0 ** (-(alpha + np.log2(n))), d_max=DMAX)
    _, it0 = gold_params(np.exp(-spread / (2.0 * d2)), np.exp(spread / (2.0 * d2)), alpha)
    _, it1 = gold_params(1.0 / n ** 2, 1.15 / n, alpha)
    depth = poly_depth(de) + int(np.log2(d1)) + (2 * it0 + 3)
    depth += int(np.log2(d2)) * (1 + 2 * it1 + 3)
    return {"d1": d1, "d2": d2, "deg_exp": de, "it0": it0, "it1": it1, "depth": depth, "n": n,
            "spread": spread, "alpha": alpha, "name": "THOR",
            "rot": int(np.log2(n)) * (int(np.log2(d2)) + 1)}


def thor_softmax(rt, ct, cfg, mid):
    ctx = rt.ctx
    n, alpha = cfg["n"], cfg["alpha"]
    d1, d2 = cfg["d1"], cfg["d2"]
    w = cfg["spread"] / (d1 * d2)
    y = ctx.add_const(ct, -float(mid))
    half = cfg["spread"] / 2.0
    c_exp = fit_coeffs(lambda u: np.exp(u / (d1 * d2)), -half, half,
                       2.0 ** (-(alpha + np.log2(n))), d_max=DMAX,
                       key=("thorexp", n, round(half, 6), d1, d2, alpha))
    y = rt.need(y, poly_depth(len(c_exp) - 1) + 1 + int(np.log2(d1)))
    y = rt.ev.evaluate(y, c_exp, -half, half)
    for _ in range(int(np.log2(d1))):
        y = ctx.square(y)
    lo = float(np.exp(-cfg["spread"] / (2.0 * d2))) * 0.85
    hi = float(np.exp(cfg["spread"] / (2.0 * d2))) * 1.15
    for j in range(int(np.log2(d2)) + 1):
        s = ctx.mul_const(rt.block_sum(y), 1.0 / n)
        lam, _ = goldschmidt_inv(rt, s, lo, hi, alpha)
        y = rt.need(y, 1)
        y = ctx.mul(y, ctx.mul_const(lam, 1.0 / n))
        if j < int(np.log2(d2)):
            y = rt.need(y, 1)
            y = ctx.square(y)
            lo, hi = 0.85 / n ** 2, 1.15 / n
    return y, cfg


def nexus_plan(n, spread, alpha=16, r=7, gold=None):
    if gold is None:
        _, gold = gold_params(1.0 / n, 1.15, alpha)
    depth = r + 1 + 2 * gold + 3
    return {"r": r, "gold": gold, "depth": depth, "n": n, "spread": spread, "alpha": alpha,
            "name": "NEXUS", "rot": int(np.log2(n))}


def nexus_softmax(rt, ct, cfg, anchor):
    ctx = rt.ctx
    n, r = cfg["n"], cfg["r"]
    y = ctx.add_const(ct, -float(anchor))
    y = rt.need(y, r + 1)
    y = ctx.add_const(ctx.mul_const(y, 1.0 / (2.0 ** r)), 1.0)
    for _ in range(r):
        y = ctx.square(y)
    s = ctx.mul_const(rt.block_sum(y), 1.0 / n)
    lam, _ = goldschmidt_inv(rt, s, 0.85 / n, 1.15, cfg["alpha"])
    y = rt.need(y, 1)
    return ctx.mul(y, ctx.mul_const(lam, 1.0 / n)), cfg


def quad_plan(n, spread, alpha=16):
    _, gold = gold_params(1.0 / n, 1.15, alpha)
    return {"gold": gold, "depth": 2 + 2 * gold + 3, "n": n, "spread": spread, "alpha": alpha,
            "name": "2Quad", "rot": int(np.log2(n))}


def quad_softmax(rt, ct, cfg, anchor):
    ctx = rt.ctx
    n = cfg["n"]
    y = ctx.add_const(ctx.mul_const(ct, 1.0 / max(cfg["spread"], 1.0)), 1.0)
    y = rt.need(y, 1)
    y = ctx.square(y)
    s = ctx.mul_const(rt.block_sum(y), 1.0 / n)
    lam, _ = goldschmidt_inv(rt, s, 0.85 / n, 1.15 * 4, cfg["alpha"])
    y = rt.need(y, 1)
    return ctx.mul(y, ctx.mul_const(lam, 1.0 / n)), cfg


SIGN_F = np.array([0.0, 35.0 / 16, 0.0, -35.0 / 16, 0.0, 21.0 / 16, 0.0, -5.0 / 16])
SIGN_G = np.array([0.0, 4589.0 / 1024, 0.0, -16577.0 / 1024, 0.0, 25614.0 / 1024, 0.0,
                   -12860.0 / 1024])


def _poly_cheb(coeffs):
    return cheb_coeffs(lambda t: np.polyval(np.asarray(coeffs)[::-1], t), -1.0, 1.0,
                       len(coeffs) - 1)


def hetal_plan(n, spread, alpha=16, df=4, dg=4):
    d_sign = (df + dg) * poly_depth(7)
    rounds = int(np.log2(n))
    de = min_degree(np.exp, -spread, 0.0, 2.0 ** (-(alpha + np.log2(n))), d_max=DMAX)
    _, gold = gold_params(1.0 / n, 1.15, alpha)
    depth = rounds * (d_sign + 2) + poly_depth(de) + 2 * gold + 4
    return {"df": df, "dg": dg, "rounds": rounds, "d_sign": d_sign, "deg_exp": de,
            "gold": gold, "depth": depth, "n": n, "spread": spread, "alpha": alpha,
            "name": "HETAL", "rot": rounds + int(np.log2(n))}


def hetal_max(rt, ct, cfg, bound):
    ctx = rt.ctx
    cf, cg = _poly_cheb(SIGN_F), _poly_cheb(SIGN_G)
    cur = ctx.mul_const(ct, 1.0 / float(bound))
    step = rt.pk.batch
    for _ in range(cfg["rounds"]):
        cur = rt.need(cur, cfg["d_sign"] + 3)
        other = ctx.rotate(cur, step)
        a = ctx.mul_const(ctx.add(cur, other), 0.5)
        d = ctx.mul_const(ctx.sub(cur, other), 0.5)
        s = d
        for _ in range(cfg["dg"]):
            s = rt.need(s, poly_depth(7) + 1)
            s = rt.ev.evaluate(s, cg, -1.0, 1.0)
        for _ in range(cfg["df"]):
            s = rt.need(s, poly_depth(7) + 1)
            s = rt.ev.evaluate(s, cf, -1.0, 1.0)
        s = rt.need(s, 2)
        cur = ctx.add(a, ctx.mul(d, s))
        step *= 2
    return ctx.mul_const(cur, float(bound))


def hetal_softmax(rt, ct, cfg, bound):
    ctx = rt.ctx
    n, alpha = cfg["n"], cfg["alpha"]
    mx = hetal_max(rt, ct, cfg, bound)
    y = ctx.sub(rt.need(ct, 0), mx)
    c_exp = fit_coeffs(np.exp, -cfg["spread"], 0.0, 2.0 ** (-(alpha + np.log2(n))), d_max=DMAX,
                       key=("hexp", n, cfg["spread"], alpha))
    y = rt.need(y, poly_depth(len(c_exp) - 1) + 1)
    y = rt.ev.evaluate(y, c_exp, -cfg["spread"], 0.0)
    s = ctx.mul_const(rt.block_sum(y), 1.0 / n)
    lam, _ = goldschmidt_inv(rt, s, 0.85 / n, 1.15, alpha)
    y = rt.need(y, 1)
    return ctx.mul(y, ctx.mul_const(lam, 1.0 / n)), cfg

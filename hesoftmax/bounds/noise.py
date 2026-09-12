import numpy as np


def tail_factor(n, lam):
    return float(np.sqrt(2.0 * np.log(4.0 * n * 2.0 ** lam)))


class NoiseModel:
    def __init__(self, params, lam=128, h=None, heuristic=False):
        self.p = params
        self.n = params.n
        self.lam = lam
        self.h = (2.0 * self.n / 3.0) if h is None else h
        self.delta = 2.0 ** params.log_scale
        self.t = 6.0 if heuristic else tail_factor(self.n, lam)
        self.sigma = params.sigma
        self.heuristic = heuristic

    def can(self, var_per_coeff):
        return self.t * np.sqrt(self.n * var_per_coeff)

    @property
    def b_fresh(self):
        return self.can(self.sigma ** 2)

    @property
    def b_rescale(self):
        v_tau = 1.0 / 12.0
        return self.can(v_tau) + self.can(v_tau) * self.can(self.h / self.n)

    def b_keyswitch(self, level_bits, log_p, dnum):
        digit = 2.0 ** (level_bits / dnum)
        var = dnum * (digit ** 2) * (self.sigma ** 2) / 12.0
        return self.can(var) / (2.0 ** log_p) + self.b_rescale


class CircuitBound:
    def __init__(self, model, log_q_top, log_p, dnum):
        self.m = model
        self.log_q = log_q_top
        self.log_p = log_p
        self.dnum = dnum
        self.bks = model.b_keyswitch(log_q_top, log_p, dnum)
        self.brs = model.b_rescale
        self.eps_unit = (self.bks + self.brs) / model.delta

    def fresh(self, mag):
        return (float(mag), float(self.m.b_fresh / self.m.delta))

    def add(self, a, b):
        return (a[0] + b[0], a[1] + b[1])

    def mul_const(self, a, c):
        return (abs(c) * a[0], abs(c) * a[1] + self.brs / self.m.delta)

    def mul(self, a, b):
        m = a[0] * b[0]
        e = a[0] * b[1] + b[0] * a[1] + a[1] * b[1] + self.eps_unit
        return (m, e)

    def rotate(self, a):
        return (a[0], a[1] + self.eps_unit)

    def poly(self, a, degree, coeff_l1, out_mag):
        d = int(np.ceil(np.log2(degree + 1)))
        e = a[1]
        m = a[0]
        for _ in range(d):
            m, e = self.mul((m, e), (m, e))
            m = min(m, 1.0) if m > 1.0 else m
        return (float(out_mag), float(coeff_l1 * e + d * self.eps_unit))


def flooding_bits(err_bound, kappa=128, queries=1, log_n=16):
    return (kappa / 2.0 + 3.0 + 0.5 * np.log2(queries) + log_n
            + np.log2(max(err_bound, 1e-300)))


def required_scale_bits(err_bound, out_precision, kappa=128, queries=1, log_n=16):
    return flooding_bits(err_bound, kappa, queries, log_n) + out_precision


def summarize(params, plan_depth, n_softmax, mag_bound, kappa=128, queries=1, lam=128):
    rig = NoiseModel(params, lam=lam)
    heu = NoiseModel(params, lam=lam, heuristic=True)
    out = {}
    for tag, mdl in (("rigorous", rig), ("heuristic", heu)):
        cc = CircuitBound(mdl, params.log_q0 + params.log_scale * params.n_levels,
                         params.log_p * params.n_special, params.dnum)
        st = cc.fresh(mag_bound)
        for _ in range(plan_depth):
            st = cc.mul(st, (1.0, st[1]))
        for _ in range(n_softmax):
            st = cc.rotate(st)
        out[tag] = {"err": st[1], "log2_err": float(np.log2(max(st[1], 1e-300))),
                    "flood_bits": float(flooding_bits(st[1], kappa, queries))}
    return out


def unit_error(params, lam=128, heuristic=False):
    m = NoiseModel(params, lam=lam, heuristic=heuristic)
    cc = CircuitBound(m, params.log_q0 + params.log_scale * params.n_levels,
                     params.log_p * params.n_special, params.dnum)
    return float(cc.eps_unit)


def acs_error_bound(params, n, spread, alpha=16, lam=128, heuristic=False, cfg=None):
    from ..softmax import ours as O
    from ..polyapprox.chebyshev import poly_depth
    from .symbolic import tail_factor
    cfg = cfg or O.plan(n, spread / 2.0, alpha)
    u = unit_error(params, lam, heuristic)
    k = cfg["k"]

    def ops(d):
        return 3.0 * np.sqrt(d + 1.0) + np.log2(d + 1.0) + 2.0

    r = ops(cfg["deg_exp"]) * u
    for hi, dj, tol in cfg["stages"]:
        r_lam = np.sqrt(n) * ops(dj) * u
        r = 2.0 * r + 2.0 * r_lam + 3.0 * u
    eps_last = 2.0 ** (-alpha)
    return {"u": u, "rel": float(r), "err": float(r + eps_last),
            "log2_err": float(np.log2(r + eps_last)), "k": k, "depth": cfg["depth"]}

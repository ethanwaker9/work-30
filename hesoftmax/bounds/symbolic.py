import numpy as np

from ..ckks.scheme import OpCounter


def tail_factor(n, lam):
    return float(np.sqrt(2.0 * np.log(4.0 * n * 2.0 ** lam)))


class BoundCt:
    __slots__ = ("mag", "var", "level", "scale")

    def __init__(self, mag, var, level, scale):
        self.mag = float(mag)
        self.var = float(var)
        self.level = int(level)
        self.scale = float(scale)

    def copy(self):
        return BoundCt(self.mag, self.var, self.level, self.scale)


class BoundContext:
    def __init__(self, params, lam=128, heuristic=False, mag0=1.0):
        self.params = params
        self.n = params.n
        self.lam = lam
        self.t = 6.0 if heuristic else tail_factor(self.n, lam)
        self.sigma = params.sigma
        self.delta = float(2 ** params.log_scale)
        self.dscale = [self.delta] * (params.n_levels + 1)
        self.max_level = params.n_levels
        self.counter = OpCounter()
        self.h = 2.0 * self.n / 3.0
        self.mag0 = mag0
        self.cap = float("inf")
        self.v_fresh = self._can_var(self.sigma ** 2)
        self.v_rescale = self._can_var(1.0 / 12.0) * (1.0 + self.h)
        log_digit = (params.log_q0 + params.log_scale * params.n_levels) / params.dnum
        log_p = params.log_p * params.n_special
        self.v_ks = params.dnum * self._can_var((2.0 ** (2 * (log_digit - log_p))) *
                                                self.sigma ** 2 * self.n / 12.0) + self.v_rescale
        self.unit = (self.v_ks + self.v_rescale) / self.delta ** 2

    def _can_var(self, var_per_coeff):
        return self.n * var_per_coeff

    def _cap(self, m):
        return min(float(m), self.cap)

    def set_mag(self, ct, mag):
        ct.mag = min(ct.mag, float(mag))
        return ct

    def bound(self, ct):
        return self.t * np.sqrt(max(ct.var, 0.0))

    def encrypt_sk(self, values, level=None, scale=None):
        lv = self.max_level if level is None else level
        mag = float(np.max(np.abs(np.asarray(values)))) if np.ndim(values) else float(values)
        return BoundCt(max(mag, 1e-12), self.v_fresh / self.delta ** 2, lv, self.delta)

    def add(self, x, y):
        self.counter.add += 1
        lv = min(x.level, y.level)
        return BoundCt(self._cap(x.mag + y.mag), x.var + y.var, lv, self.delta)

    def sub(self, x, y):
        self.counter.add += 1
        lv = min(x.level, y.level)
        return BoundCt(self._cap(x.mag + y.mag), x.var + y.var, lv, self.delta)

    def add_const(self, x, c):
        self.counter.add += 1
        return BoundCt(self._cap(x.mag + abs(float(np.real(c)))), x.var, x.level, self.delta)

    def mul_const(self, x, c, rescale=True, target_scale=None):
        self.counter.mult_pt += 1
        a = abs(float(np.real(c)))
        v = a * a * x.var + (self.v_rescale / self.delta ** 2 if rescale else 0.0)
        lv = x.level - 1 if rescale else x.level
        return BoundCt(self._cap(a * x.mag), v, lv, self.delta)

    def mul(self, x, y, rescale=True):
        self.counter.mult_cc += 1
        self.counter.relin += 1
        self.counter.keyswitch += 1
        v = x.mag ** 2 * y.var + y.mag ** 2 * x.var + x.var * y.var + self.unit
        lv = min(x.level, y.level) - (1 if rescale else 0)
        return BoundCt(self._cap(x.mag * y.mag), v, lv, self.delta)

    def square(self, x, rescale=True):
        return self.mul(x, x, rescale=rescale)

    def rescale(self, x):
        self.counter.rescale += 1
        return BoundCt(x.mag, x.var + self.v_rescale / self.delta ** 2, x.level - 1, self.delta)

    def rotate(self, x, r):
        self.counter.rot += 1
        self.counter.keyswitch += 1
        return BoundCt(x.mag, x.var + self.unit, x.level, self.delta)

    def level_down(self, ct, target=None):
        target = ct.level - 1 if target is None else target
        return BoundCt(ct.mag, ct.var, min(ct.level, target), self.delta)

    def match(self, x, y):
        lv = min(x.level, y.level)
        return self.level_down(x, lv), self.level_down(y, lv)

    def decrypt(self, ct):
        return np.zeros(1)


class BoundRuntime:
    def __init__(self, ctx, n, refresh_at=1):
        from ..softmax.common import Packing
        from ..polyapprox.evaluate import ChebyshevEvaluator
        self.ctx = ctx
        self.pk = Packing(max(n, 1) * 2, n)
        self.ev = ChebyshevEvaluator(ctx)
        self.refresh_at = refresh_at
        self.boot = 0
        self.consumed = 0
        self.max_level = ctx.max_level

    def need(self, ct, levels):
        if ct.level - levels < self.refresh_at:
            self.boot += 1
            self.consumed += self.max_level - ct.level
            return BoundCt(ct.mag, ct.var, self.max_level, ct.scale)
        return ct

    def refresh(self, ct):
        return self.need(ct, 10 ** 6)

    def block_sum(self, ct):
        out = ct
        for _ in range(int(np.log2(self.pk.n))):
            out = self.ctx.add(out, self.ctx.rotate(out, 1))
        return out


def bound_acs(params, n, spread, alpha=16, lam=128, heuristic=False):
    from ..softmax import ours as O
    ctx = BoundContext(params, lam=lam, heuristic=heuristic)
    ctx.cap = float(max(spread, n))
    rt = BoundRuntime(ctx, n)
    cfg = O.plan(n, spread / 2.0, alpha)
    ct = ctx.encrypt_sk([spread / 2.0])
    out, _ = O.acs_softmax(rt, ct, cfg)
    return {"depth": cfg["depth"], "mag": out.mag, "err": ctx.bound(out),
            "log2_err": float(np.log2(max(ctx.bound(out), 1e-300))),
            "boot": rt.boot, "keyswitch": ctx.counter.keyswitch,
            "mult_cc": ctx.counter.mult_cc, "rot": ctx.counter.rot}

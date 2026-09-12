import numpy as np
from ..polyapprox.chebyshev import cheb_coeffs, min_degree, poly_depth
from ..polyapprox.evaluate import ChebyshevEvaluator


class Packing:
    def __init__(self, slots, n):
        assert slots % n == 0
        self.slots = slots
        self.n = n
        self.batch = slots // n

    def pack(self, mat):
        mat = np.asarray(mat, dtype=np.float64)
        out = np.zeros(self.slots)
        for j in range(mat.shape[0]):
            out[j::self.batch] = mat[j]
        return out

    def unpack(self, vec):
        vec = np.asarray(vec)
        return np.stack([vec[j::self.batch].real for j in range(self.batch)])


class Runtime:
    def __init__(self, ctx, packing, refresh_at=1):
        self.ctx = ctx
        self.pk = packing
        self.ev = ChebyshevEvaluator(ctx)
        self.refresh_at = refresh_at
        self.boot = 0
        self.consumed = 0
        self.max_level = ctx.max_level

    def need(self, ct, levels):
        if ct.level - levels < self.refresh_at:
            return self.refresh(ct)
        return ct

    def refresh(self, ct):
        self.boot += 1
        self.consumed += self.max_level - ct.level
        self.ctx.counter.boot += 1
        vals = self.ctx.decrypt(ct)
        return self.ctx.encrypt_sk(vals, level=self.max_level)

    def block_sum(self, ct):
        ctx, pk = self.ctx, self.pk
        out = ct
        step = pk.batch
        for _ in range(int(np.log2(pk.n))):
            out = ctx.add(out, ctx.rotate(out, step))
            step *= 2
        return out

    def mask_broadcast(self, ct):
        ctx, pk = self.ctx, self.pk
        m = np.zeros(pk.slots)
        m[:pk.batch] = 1.0
        pt = ctx.encode_plain(m, ct.level, ctx.dscale[ct.level])
        out = ctx.mul_plain(ct, pt)
        step = pk.batch
        for _ in range(int(np.log2(pk.n))):
            out = ctx.add(out, ctx.rotate(out, -step))
            step *= 2
        return out

    def poly(self, ct, f, a, b, tol, relative=False, d_max=512, log_grid=False, levels_hint=None):
        c = fit_coeffs(f, a, b, tol, relative, d_max, log_grid)
        ct = self.need(ct, poly_depth(len(c) - 1) + 1)
        return self.ev.evaluate(ct, c, a, b), len(c) - 1


_FIT_CACHE = {}


def fit_coeffs(f, a, b, tol, relative=False, d_max=512, log_grid=False, key=None):
    kk = (key, float(a), float(b), float(tol), relative, d_max, log_grid)
    if kk in _FIT_CACHE:
        return _FIT_CACHE[kk]
    d = min_degree(f, a, b, tol, relative, d_max, log_grid)
    if d is None:
        d = d_max
    c = cheb_coeffs(f, a, b, d)
    if key is not None:
        _FIT_CACHE[kk] = c
    return c


def softmax_ref(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)

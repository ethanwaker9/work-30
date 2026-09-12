import numpy as np


class ChebyshevEvaluator:
    def __init__(self, ctx):
        self.ctx = ctx

    def _affine(self, ct, a, b):
        s = 2.0 / (b - a)
        t = -(a + b) / (b - a)
        if abs(s - 1.0) < 1e-15 and abs(t) < 1e-15:
            return ct
        out = self.ctx.mul_const(ct, s)
        if abs(t) > 1e-15:
            out = self.ctx.add_const(out, t)
        return self.ctx.set_mag(out, 1.0)

    def _baby(self, t, k):
        ctx = self.ctx
        T = {1: t}
        for i in range(2, k + 1):
            if i % 2 == 0:
                h = i // 2
                sq = ctx.square(T[h])
                T[i] = ctx.add_const(ctx.add(sq, sq), -1.0)
            else:
                h = i // 2
                p = ctx.mul(T[h], T[h + 1])
                T[i] = ctx.sub(ctx.add(p, p), T[1])
            ctx.set_mag(T[i], 1.0)
        lv = min(v.level for v in T.values())
        return {i: ctx.level_down(v, lv) for i, v in T.items()}

    def _giant(self, T, k, n_giant):
        ctx = self.ctx
        G = {k: T[k]}
        cur = k
        for _ in range(n_giant):
            sq = ctx.square(G[cur])
            G[2 * cur] = ctx.set_mag(ctx.add_const(ctx.add(sq, sq), -1.0), 1.0)
            cur *= 2
        return G

    def _rec(self, coeffs, T, G, k, m):
        ctx = self.ctx
        d = len(coeffs) - 1
        while d >= 0 and abs(coeffs[d]) == 0.0:
            d -= 1
        if d < 0:
            return None, 0.0
        if m == 0 or d < k:
            acc = None
            const = float(coeffs[0])
            lv = T[1].level
            tgt = ctx.dscale[lv] * ctx.dscale[lv]
            for i in range(1, min(d, k) + 1):
                if coeffs[i] == 0.0:
                    continue
                term = ctx.mul_const(T[i], float(coeffs[i]), rescale=False, target_scale=tgt)
                acc = term if acc is None else ctx.add(acc, term)
            if acc is None:
                return None, const
            acc = ctx.rescale(acc)
            acc.scale = ctx.dscale[acc.level]
            return acc, const
        nsplit = k << (m - 1)
        lo = coeffs[:nsplit]
        hi = coeffs[nsplit:]
        q = np.zeros(max(len(hi), 1))
        q[:len(hi)] = hi
        q[0] = q[0] / 2.0
        r = np.array(lo, dtype=np.float64)
        for j in range(1, len(hi)):
            if nsplit - j >= 0:
                r[nsplit - j] -= hi[j]
        qc, qk = self._rec(q, T, G, k, m - 1)
        rc, rk = self._rec(r, T, G, k, m - 1)
        gt = G[nsplit]
        if qc is None:
            prod = ctx.mul_const(gt, 2.0 * qk) if qk != 0.0 else None
        else:
            if qk != 0.0:
                qc = ctx.add_const(qc, qk)
            prod = ctx.mul(qc, gt)
            prod = ctx.add(prod, prod)
        if prod is None:
            return rc, rk
        if rc is not None:
            prod = ctx.add(prod, rc)
        return prod, rk

    def evaluate(self, ct, coeffs, a, b, out_mag=None):
        ctx = self.ctx
        coeffs = np.asarray(coeffs, dtype=np.float64)
        if out_mag is None:
            out_mag = float(np.abs(coeffs[0]) + np.abs(coeffs[1:]).sum())
        d = len(coeffs) - 1
        if d <= 0:
            return ctx.set_mag(ctx.add_const(ctx.mul_const(ct, 0.0), float(coeffs[0])), out_mag)
        t = self._affine(ct, a, b)
        r = int(np.ceil(np.log2(d + 1)))
        kb = 1 << ((r + 1) // 2)
        m = r - ((r + 1) // 2)
        kb = min(kb, d)
        T = self._baby(t, kb)
        G = self._giant(T, kb, max(m - 1, 0)) if m >= 1 else {}
        full = np.zeros(max(kb << m, d + 1))
        full[:d + 1] = coeffs
        out, k0 = self._rec(full, T, G, kb, m)
        if out is None:
            return ctx.set_mag(ctx.add_const(ctx.mul_const(ct, 0.0), float(k0)), out_mag)
        if k0 != 0.0:
            out = ctx.add_const(out, float(k0))
        return ctx.set_mag(out, out_mag)

import numpy as np
from .numbertheory import primitive_root, bit_reverse_table


class Ring:
    def __init__(self, n_ring, primes):
        self.n = int(n_ring)
        self.primes = [int(p) for p in primes]
        self.k = len(self.primes)
        self.q = np.array(self.primes, dtype=np.uint64)
        self.qi = self.q.astype(np.int64)
        self.qf = 1.0 / self.q.astype(np.float64)
        br = bit_reverse_table(self.n)
        psi = np.zeros((self.k, self.n), dtype=np.uint64)
        ipsi = np.zeros((self.k, self.n), dtype=np.uint64)
        self.n_inv = np.zeros(self.k, dtype=np.uint64)
        for i, p in enumerate(self.primes):
            g = primitive_root(p)
            w = pow(g, (p - 1) // (2 * self.n), p)
            iw = pow(w, p - 2, p)
            pw = [0] * self.n
            ipw = [0] * self.n
            a, b = 1, 1
            for j in range(self.n):
                pw[j] = a
                ipw[j] = b
                a = a * w % p
                b = b * iw % p
            psi[i] = np.array([pw[t] for t in br], dtype=np.uint64)
            ipsi[i] = np.array([ipw[t] for t in br], dtype=np.uint64)
            self.n_inv[i] = pow(self.n, p - 2, p)
        self.psi = psi
        self.ipsi = ipsi
        self.psi_f = psi.astype(np.float64) * self.qf.reshape(-1, 1)
        self.ipsi_f = ipsi.astype(np.float64) * self.qf.reshape(-1, 1)
        self._bconv = {}

    def qv(self, idx):
        return self.q[idx].reshape(-1, 1)

    def mulmod(self, a, b, idx):
        q = self.q[idx].reshape(-1, 1)
        qf = self.qf[idx].reshape(-1, 1)
        est = (a.astype(np.float64) * b.astype(np.float64) * qf).astype(np.int64).astype(np.uint64)
        r = (a * b - est * q).astype(np.int64)
        qs = q.astype(np.int64)
        r = np.where(r < 0, r + qs, r)
        r = np.where(r >= qs, r - qs, r)
        return r.astype(np.uint64)

    def mul_scalar(self, a, s, idx):
        return self.mulmod(a, np.asarray(s, dtype=np.uint64).reshape(-1, 1), idx)

    def add(self, a, b, idx):
        q = self.q[idx].reshape(-1, 1)
        r = a + b
        return np.where(r >= q, r - q, r)

    def sub(self, a, b, idx):
        q = self.q[idx].reshape(-1, 1)
        r = a + q - b
        return np.where(r >= q, r - q, r)

    def neg(self, a, idx):
        q = self.q[idx].reshape(-1, 1)
        return np.where(a == 0, a, q - a)

    def ntt(self, a, idx):
        n = self.n
        k = len(idx)
        out = np.array(a, dtype=np.uint64, copy=True)
        psi = self.psi[idx]
        psif = self.psi_f[idx]
        q = self.q[idx].reshape(k, 1, 1)
        qs = q.astype(np.int64)
        m, t = 1, n // 2
        while m < n:
            v = out.reshape(k, m, 2, t)
            s = psi[:, m:2 * m].reshape(k, m, 1)
            sf = psif[:, m:2 * m].reshape(k, m, 1)
            u = v[:, :, 0, :]
            w = v[:, :, 1, :]
            est = (w.astype(np.float64) * sf).astype(np.int64).astype(np.uint64)
            r = (w * s - est * q).astype(np.int64)
            r = np.where(r < 0, r + qs, r)
            r = np.where(r >= qs, r - qs, r).astype(np.uint64)
            x = u + r
            x = np.where(x >= q, x - q, x)
            y = u + q - r
            y = np.where(y >= q, y - q, y)
            v[:, :, 0, :] = x
            v[:, :, 1, :] = y
            m *= 2
            t //= 2
        return out

    def intt(self, a, idx):
        n = self.n
        k = len(idx)
        out = np.array(a, dtype=np.uint64, copy=True)
        ipsi = self.ipsi[idx]
        ipsif = self.ipsi_f[idx]
        q = self.q[idx].reshape(k, 1, 1)
        qs = q.astype(np.int64)
        t, m = 1, n // 2
        while m >= 1:
            v = out.reshape(k, m, 2, t)
            s = ipsi[:, m:2 * m].reshape(k, m, 1)
            sf = ipsif[:, m:2 * m].reshape(k, m, 1)
            u = v[:, :, 0, :]
            w = v[:, :, 1, :]
            x = u + w
            x = np.where(x >= q, x - q, x)
            y = u + q - w
            y = np.where(y >= q, y - q, y)
            est = (y.astype(np.float64) * sf).astype(np.int64).astype(np.uint64)
            r = (y * s - est * q).astype(np.int64)
            r = np.where(r < 0, r + qs, r)
            r = np.where(r >= qs, r - qs, r).astype(np.uint64)
            v[:, :, 0, :] = x
            v[:, :, 1, :] = r.astype(np.uint64)
            m //= 2
            t *= 2
        return self.mulmod(out, self.n_inv[idx].reshape(k, 1), idx)

    def bconv_tables(self, src, dst):
        key = (tuple(src), tuple(dst))
        if key in self._bconv:
            return self._bconv[key]
        prod = 1
        for i in src:
            prod *= self.primes[i]
        inv = []
        mat = np.zeros((len(dst), len(src)), dtype=np.uint64)
        for a, i in enumerate(src):
            qi = self.primes[i]
            hat = prod // qi
            inv.append(pow(hat % qi, qi - 2, qi))
            for b, j in enumerate(dst):
                mat[b, a] = hat % self.primes[j]
        tab = (np.array(inv, dtype=np.uint64).reshape(-1, 1), mat)
        self._bconv[key] = tab
        return tab

    def bconv(self, a, src, dst):
        inv, mat = self.bconv_tables(src, dst)
        t = self.mulmod(a, inv, src)
        tf = t.astype(np.float64)
        out = np.zeros((len(dst), self.n), dtype=np.uint64)
        for b in range(len(dst)):
            j = dst[b]
            qj = np.uint64(self.q[j])
            qjs = np.int64(self.q[j])
            c = mat[b].reshape(-1, 1)
            est = (tf * c.astype(np.float64) * self.qf[j]).astype(np.int64).astype(np.uint64)
            r = (t * c - est * qj).astype(np.int64)
            r = np.where(r < 0, r + qjs, r)
            r = np.where(r >= qjs, r - qjs, r)
            out[b] = np.mod(r.sum(axis=0, dtype=np.int64), qjs).astype(np.uint64)
        return out

    def bconv_exact(self, a, src, dst):
        inv, mat = self.bconv_tables(src, dst)
        t = self.mulmod(a, inv, src)
        tf = t.astype(np.float64)
        qsrc = self.q[src].astype(np.float64).reshape(-1, 1)
        v = np.rint((tf / qsrc).sum(axis=0)).astype(np.int64)
        prod_mod = np.zeros(len(dst), dtype=np.uint64)
        prod = 1
        for i in src:
            prod *= self.primes[i]
        for b, j in enumerate(dst):
            prod_mod[b] = prod % self.primes[j]
        out = np.zeros((len(dst), self.n), dtype=np.uint64)
        for b in range(len(dst)):
            j = dst[b]
            qj = np.uint64(self.q[j])
            qjs = np.int64(self.q[j])
            c = mat[b].reshape(-1, 1)
            est = (tf * c.astype(np.float64) * self.qf[j]).astype(np.int64).astype(np.uint64)
            r = (t * c - est * qj).astype(np.int64)
            r = np.where(r < 0, r + qjs, r)
            r = np.where(r >= qjs, r - qjs, r)
            acc = np.mod(r.sum(axis=0, dtype=np.int64), qjs)
            corr = np.mod(v.astype(np.int64) * np.int64(prod_mod[b]), qjs)
            out[b] = np.mod(acc - corr, qjs).astype(np.uint64)
        return out

    def center(self, a, idx):
        q = self.q[idx].reshape(-1, 1).astype(np.int64)
        ai = a.astype(np.int64)
        return np.where(ai > q // 2, ai - q, ai)

    def perm_table(self, g):
        key = ("perm", int(g))
        if key in self._bconv:
            return self._bconv[key]
        n = self.n
        br = bit_reverse_table(n)
        expo = (2 * br + 1) % (2 * n)
        pos_of = np.zeros(2 * n, dtype=np.int64)
        pos_of[expo] = np.arange(n)
        tgt = (expo * int(g)) % (2 * n)
        tab = pos_of[tgt]
        self._bconv[key] = tab
        return tab

    def permute_ntt(self, a, g):
        return a[:, self.perm_table(g)]

    def to_int(self, res, idx):
        prod = 1
        for i in idx:
            prod *= self.primes[i]
        acc = np.zeros(self.n, dtype=object)
        for a, i in enumerate(idx):
            qi = self.primes[i]
            hat = prod // qi
            g = pow(hat % qi, qi - 2, qi)
            acc = acc + np.array([int(v) for v in res[a]], dtype=object) * (hat * g)
        acc = acc % prod
        half = prod // 2
        return np.array([int(v) - prod if int(v) > half else int(v) for v in acc], dtype=object)

    def from_int(self, coeffs, idx):
        c = np.asarray(coeffs)
        out = np.zeros((len(idx), self.n), dtype=np.uint64)
        if c.dtype == object:
            for a, i in enumerate(idx):
                p = self.primes[i]
                out[a] = np.array([int(v) % p for v in c], dtype=np.uint64)
            return out
        c = c.astype(np.int64)
        for a, i in enumerate(idx):
            p = np.int64(self.primes[i])
            out[a] = np.mod(c, p).astype(np.uint64)
        return out

    def to_int_small(self, res, idx, take=3):
        use = list(idx[:take])
        prod = 1
        for i in use:
            prod *= self.primes[i]
        acc = np.zeros(self.n, dtype=object)
        for a, i in enumerate(use):
            qi = self.primes[i]
            hat = prod // qi
            g = pow(hat % qi, qi - 2, qi)
            acc = acc + np.array(res[a].tolist(), dtype=object) * (hat * g)
        acc = acc % prod
        half = prod // 2
        return np.array([v - prod if v > half else v for v in acc.tolist()], dtype=object)

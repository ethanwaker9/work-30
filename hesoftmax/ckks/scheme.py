import numpy as np
from .rns import Ring
from .encoder import CkksEncoder


class OpCounter:
    FIELDS = ("mult_cc", "mult_pt", "rot", "relin", "rescale", "add", "keyswitch", "boot")

    def __init__(self):
        self.reset()

    def reset(self):
        for f in self.FIELDS:
            setattr(self, f, 0)

    def snapshot(self):
        return {f: getattr(self, f) for f in self.FIELDS}

    def diff(self, before):
        return {f: getattr(self, f) - before[f] for f in self.FIELDS}


class Ciphertext:
    __slots__ = ("c0", "c1", "level", "scale")

    def __init__(self, c0, c1, level, scale):
        self.c0 = c0
        self.c1 = c1
        self.level = level
        self.scale = scale

    def copy(self):
        return Ciphertext(self.c0.copy(), self.c1.copy(), self.level, self.scale)

    def nbytes(self):
        return self.c0.nbytes + self.c1.nbytes


class Plaintext:
    __slots__ = ("m", "level", "scale")

    def __init__(self, m, level, scale):
        self.m = m
        self.level = level
        self.scale = scale


class CkksContext:
    def __init__(self, params, seed=0):
        self.params = params
        self.rng = np.random.default_rng(seed)
        qprimes, pprimes, dscales = params.build_primes()
        self.dscale = list(dscales)
        self.n_q = len(qprimes)
        self.n_p = len(pprimes)
        self.ring = Ring(params.n, qprimes + pprimes)
        self.q_idx = list(range(self.n_q))
        self.p_idx = list(range(self.n_q, self.n_q + self.n_p))
        self.encoder = CkksEncoder(params.n)
        self.scale = float(2 ** params.log_scale)
        self.counter = OpCounter()
        self.max_level = self.n_q - 1
        d = params.dnum
        base = self.n_q // d
        self.digits = []
        for i in range(d):
            lo = i * base
            hi = self.n_q if i == d - 1 else (i + 1) * base
            self.digits.append(list(range(lo, hi)))
        self.p_prod = 1
        for j in self.p_idx:
            self.p_prod *= self.ring.primes[j]
        self._pt_cache = {}
        self._key_seed = 10 ** 6
        self.sk = None
        self.pk = None
        self.rlk = None
        self.rot_keys = {}
        self.conj_key = None

    def set_mag(self, ct, mag):
        return ct

    def level_scale(self, level):
        return self.dscale[level]

    def level_down(self, ct, target=None):
        target = ct.level - 1 if target is None else target
        if target >= ct.level:
            return ct
        return Ciphertext(ct.c0[:target + 1].copy(), ct.c1[:target + 1].copy(), target,
                          self.dscale[target])

    def match(self, x, y):
        lv = min(x.level, y.level)
        if x.level > lv:
            x = self.level_down(x, lv)
        if y.level > lv:
            y = self.level_down(y, lv)
        return x, y

    def lev_idx(self, level):
        return list(range(level + 1))

    def _sample_ternary(self):
        return self.rng.integers(-1, 2, size=self.params.n)

    def _sample_gauss(self):
        return np.rint(self.rng.normal(0.0, self.params.sigma, size=self.params.n)).astype(np.int64)

    def _sample_uniform(self, idx, rng=None):
        rng = rng or self.rng
        out = np.zeros((len(idx), self.params.n), dtype=np.uint64)
        for a, i in enumerate(idx):
            p = self.ring.primes[i]
            out[a] = rng.integers(0, p, size=self.params.n, dtype=np.int64).astype(np.uint64)
        return out

    def _key_a(self, seed):
        full = self.q_idx + self.p_idx
        return self._sample_uniform(full, np.random.default_rng(seed))

    def keygen(self, rotations=(), conj=False):
        R = self.ring
        n = self.params.n
        s = self._sample_ternary()
        self.sk_coeffs = s
        full = self.q_idx + self.p_idx
        self.s_full = R.ntt(R.from_int(s, full), full)
        self.s_q = self.s_full[:self.n_q]
        a = self._sample_uniform(self.q_idx)
        e = R.ntt(R.from_int(self._sample_gauss(), self.q_idx), self.q_idx)
        b = R.sub(e, R.mulmod(a, self.s_q, self.q_idx), self.q_idx)
        self.pk = (b, a)
        self.rlk = self._switch_key(self._mul_full(self.s_full, self.s_full))
        for r in rotations:
            self.rot_keys[r] = self._switch_key(self._perm_full(self.s_full, self._gal_elt(r)))
        if conj:
            self.conj_key = self._switch_key(self._perm_full(self.s_full, 2 * n - 1))
        return self.sk_coeffs

    def _mul_full(self, x, y):
        full = self.q_idx + self.p_idx
        return self.ring.mulmod(x, y, full)

    def _gal_elt(self, r):
        m = 2 * self.params.n
        return pow(5, r % (self.params.n // 2), m)

    def _perm_full(self, s_ntt, g):
        return self.ring.permute_ntt(s_ntt, g)

    def _switch_key(self, target_full):
        R = self.ring
        full = self.q_idx + self.p_idx
        keys = []
        q_prod = 1
        for i in self.q_idx:
            q_prod *= R.primes[i]
        for dig in self.digits:
            dprod = 1
            for i in dig:
                dprod *= R.primes[i]
            comp = q_prod // dprod
            factor = comp * pow(comp % dprod, -1, dprod) % q_prod
            fac = np.zeros(len(full), dtype=np.uint64)
            for a, i in enumerate(full):
                p = R.primes[i]
                fac[a] = (factor % p) * (self.p_prod % p) % p
            self._key_seed += 1
            seed = self._key_seed
            a_i = self._key_a(seed)
            e_i = R.ntt(R.from_int(self._sample_gauss(), full), full)
            b_i = R.add(R.sub(e_i, R.mulmod(a_i, self.s_full, full), full),
                        R.mulmod(target_full, fac.reshape(-1, 1), full), full)
            keys.append((b_i, seed))
        return keys

    def encrypt_sk(self, values, level=None, scale=None):
        R = self.ring
        level = self.max_level if level is None else level
        scale = self.dscale[level] if scale is None else scale
        idx = self.lev_idx(level)
        pt = self.encoder.encode(values, scale)
        m = R.ntt(R.from_int(pt, idx), idx)
        a = self._sample_uniform(idx)
        e = R.ntt(R.from_int(self._sample_gauss(), idx), idx)
        c0 = R.add(R.sub(e, R.mulmod(a, self.s_q[:level + 1], idx), idx), m, idx)
        return Ciphertext(c0, a, level, scale)

    def encrypt(self, values, level=None, scale=None):
        R = self.ring
        level = self.max_level if level is None else level
        scale = self.dscale[level] if scale is None else scale
        idx = self.lev_idx(level)
        pt = self.encoder.encode(values, scale)
        m = R.ntt(R.from_int(pt, idx), idx)
        v = self._sample_ternary()
        vn = R.ntt(R.from_int(v, idx), idx)
        e0 = R.ntt(R.from_int(self._sample_gauss(), idx), idx)
        e1 = R.ntt(R.from_int(self._sample_gauss(), idx), idx)
        b, a = self.pk[0][:level + 1], self.pk[1][:level + 1]
        c0 = R.add(R.add(R.mulmod(vn, b, idx), e0, idx), m, idx)
        c1 = R.add(R.mulmod(vn, a, idx), e1, idx)
        return Ciphertext(c0, c1, level, scale)

    def encode_plain(self, values, level, scale):
        idx = self.lev_idx(level)
        pt = self.encoder.encode(values, scale)
        return Plaintext(self.ring.ntt(self.ring.from_int(pt, idx), idx), level, scale)

    def decrypt(self, ct):
        R = self.ring
        idx = self.lev_idx(ct.level)
        m = R.add(ct.c0, R.mulmod(ct.c1, self.s_q[:ct.level + 1], idx), idx)
        coeffs = R.to_int_small(R.intt(m, idx), idx, take=min(4, len(idx)))
        return self.encoder.decode(coeffs, ct.scale)

    def add(self, x, y):
        x, y = self.match(x, y)
        self.counter.add += 1
        lv = min(x.level, y.level)
        idx = self.lev_idx(lv)
        return Ciphertext(self.ring.add(x.c0[:lv + 1], y.c0[:lv + 1], idx),
                          self.ring.add(x.c1[:lv + 1], y.c1[:lv + 1], idx), lv, x.scale)

    def sub(self, x, y):
        x, y = self.match(x, y)
        self.counter.add += 1
        lv = min(x.level, y.level)
        idx = self.lev_idx(lv)
        return Ciphertext(self.ring.sub(x.c0[:lv + 1], y.c0[:lv + 1], idx),
                          self.ring.sub(x.c1[:lv + 1], y.c1[:lv + 1], idx), lv, x.scale)

    def _const_limbs(self, c, scale, idx):
        v = int(np.rint(float(np.real(c)) * float(scale)))
        out = np.empty((len(idx), 1), dtype=np.uint64)
        for a, i in enumerate(idx):
            p = self.ring.primes[i]
            out[a, 0] = v % p
        return out

    def add_const(self, x, c):
        idx = self.lev_idx(x.level)
        m = self._const_limbs(c, x.scale, idx)
        self.counter.add += 1
        q = self.ring.q[idx].reshape(-1, 1)
        r = x.c0 + m
        r = np.where(r >= q, r - q, r)
        return Ciphertext(r, x.c1.copy(), x.level, x.scale)

    def mul_const(self, x, c, rescale=True, target_scale=None):
        idx = self.lev_idx(x.level)
        lv = x.level
        if target_scale is not None:
            s_pt = target_scale / x.scale
        elif rescale and lv > 0:
            s_pt = self.dscale[lv - 1] * self.ring.primes[lv] / x.scale
        else:
            s_pt = self.dscale[lv]
        m = self._const_limbs(c, s_pt, idx)
        self.counter.mult_pt += 1
        out = Ciphertext(self.ring.mulmod(x.c0, m, idx), self.ring.mulmod(x.c1, m, idx),
                         lv, x.scale * s_pt)
        if not rescale:
            return out
        r = self.rescale(out)
        r.scale = self.dscale[r.level]
        return r

    def mul_plain(self, x, pt, rescale=True):
        idx = self.lev_idx(x.level)
        m = pt.m[:x.level + 1]
        self.counter.mult_pt += 1
        out = Ciphertext(self.ring.mulmod(x.c0, m, idx), self.ring.mulmod(x.c1, m, idx),
                         x.level, x.scale * pt.scale)
        return self.rescale(out) if rescale else out

    def rescale(self, ct):
        R = self.ring
        lv = ct.level
        if lv == 0:
            raise ValueError("no level left to rescale")
        self.counter.rescale += 1
        idx = self.lev_idx(lv)
        low = self.lev_idx(lv - 1)
        top = [lv]
        ql = R.primes[lv]
        out = []
        for comp in (ct.c0, ct.c1):
            t = R.intt(comp[lv:lv + 1], top)
            tc = R.center(t, top)
            tt = R.ntt(R.from_int(tc[0], low), low)
            d = R.sub(comp[:lv], tt, low)
            invq = np.array([pow(ql % R.primes[i], -1, R.primes[i]) for i in low], dtype=np.uint64)
            out.append(R.mulmod(d, invq.reshape(-1, 1), low))
        return Ciphertext(out[0], out[1], lv - 1, ct.scale / ql)

    def _key_switch(self, c1, level, keys):
        R = self.ring
        self.counter.keyswitch += 1
        idx = self.lev_idx(level)
        ext = idx + self.p_idx
        acc_b = np.zeros((len(ext), R.n), dtype=np.uint64)
        acc_a = np.zeros((len(ext), R.n), dtype=np.uint64)
        for di, dig in enumerate(self.digits):
            src = [i for i in dig if i <= level]
            if not src:
                continue
            dst = [i for i in ext if i not in src]
            pos = [ext.index(i) for i in src]
            part = R.intt(c1[[idx.index(i) for i in src]], src)
            up = np.zeros((len(ext), R.n), dtype=np.uint64)
            up[pos] = R.ntt(part, src)
            conv = R.ntt(R.bconv(part, src, dst), dst)
            for a, i in enumerate(dst):
                up[ext.index(i)] = conv[a]
            kb, seed = keys[di]
            ka = self._key_a(seed)
            sel = ext
            acc_b = R.add(acc_b, R.mulmod(up, kb[sel], ext), ext)
            acc_a = R.add(acc_a, R.mulmod(up, ka[sel], ext), ext)
        outs = []
        for acc in (acc_b, acc_a):
            hi = R.intt(acc[len(idx):], self.p_idx)
            back = R.ntt(R.bconv_exact(hi, self.p_idx, idx), idx)
            d = R.sub(acc[:len(idx)], back, idx)
            invp = np.array([pow(self.p_prod % R.primes[i], -1, R.primes[i]) for i in idx],
                            dtype=np.uint64)
            outs.append(R.mulmod(d, invp.reshape(-1, 1), idx))
        return outs[0], outs[1]

    def mul(self, x, y, rescale=True):
        R = self.ring
        x, y = self.match(x, y)
        lv = min(x.level, y.level)
        idx = self.lev_idx(lv)
        self.counter.mult_cc += 1
        a0, a1 = x.c0[:lv + 1], x.c1[:lv + 1]
        b0, b1 = y.c0[:lv + 1], y.c1[:lv + 1]
        d0 = R.mulmod(a0, b0, idx)
        d1 = R.add(R.mulmod(a0, b1, idx), R.mulmod(a1, b0, idx), idx)
        d2 = R.mulmod(a1, b1, idx)
        self.counter.relin += 1
        kb, ka = self._key_switch(d2, lv, self.rlk)
        out = Ciphertext(R.add(d0, kb, idx), R.add(d1, ka, idx), lv, x.scale * y.scale)
        if not rescale:
            return out
        r = self.rescale(out)
        r.scale = self.dscale[r.level]
        return r

    def square(self, x, rescale=True):
        return self.mul(x, x, rescale=rescale)

    def _apply_perm(self, ct, g):
        R = self.ring
        return R.permute_ntt(ct.c0, g), R.permute_ntt(ct.c1, g)

    def rotate(self, ct, r):
        r = r % self.encoder.slots
        if r == 0:
            return ct.copy()
        if r not in self.rot_keys:
            raise KeyError("missing rotation key %d" % r)
        R = self.ring
        self.counter.rot += 1
        idx = self.lev_idx(ct.level)
        g = self._gal_elt(r)
        p0, p1 = self._apply_perm(ct, g)
        kb, ka = self._key_switch(p1, ct.level, self.rot_keys[r])
        return Ciphertext(R.add(p0, kb, idx), ka, ct.level, ct.scale)

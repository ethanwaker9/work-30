import numpy as np


class CkksEncoder:
    def __init__(self, n_ring):
        self.n = int(n_ring)
        self.slots = self.n // 2
        self.m = 2 * self.n
        rot = np.empty(self.slots, dtype=np.int64)
        v = 1
        for i in range(self.slots):
            rot[i] = v
            v = v * 5 % self.m
        self.rot_group = rot
        j = np.arange(self.m)
        self.ksi = np.exp(2j * np.pi * j / self.m)

    def _bitrev(self, a):
        size = a.shape[-1]
        w = size.bit_length() - 1
        idx = np.arange(size)
        rev = np.zeros(size, dtype=np.int64)
        for b in range(w):
            rev |= ((idx >> b) & 1) << (w - 1 - b)
        return a[..., rev]

    def fft_special(self, vals):
        size = vals.shape[-1]
        a = self._bitrev(np.asarray(vals, dtype=np.complex128))
        length = 2
        while length <= size:
            lenh = length >> 1
            lenq = length << 2
            j = np.arange(lenh)
            idx = (self.rot_group[j] % lenq) * (self.m // lenq)
            tw = self.ksi[idx]
            a = a.reshape(-1, size // length, 2, lenh)
            u = a[..., 0, :]
            v = a[..., 1, :] * tw
            a = np.stack([u + v, u - v], axis=-2).reshape(-1, size)
            length <<= 1
        return a.reshape(np.shape(vals))

    def fft_special_inv(self, vals):
        size = vals.shape[-1]
        a = np.asarray(vals, dtype=np.complex128).reshape(-1, size).copy()
        length = size
        while length >= 2:
            lenh = length >> 1
            lenq = length << 2
            j = np.arange(lenh)
            idx = (lenq - (self.rot_group[j] % lenq)) * (self.m // lenq)
            tw = self.ksi[idx]
            a = a.reshape(-1, size // length, 2, lenh)
            u = a[..., 0, :]
            v = a[..., 1, :]
            a = np.stack([u + v, (u - v) * tw], axis=-2).reshape(-1, size)
            length >>= 1
        a = self._bitrev(a) / size
        return a.reshape(np.shape(vals))

    def encode(self, values, scale):
        z = np.zeros(self.slots, dtype=np.complex128)
        vals = np.asarray(values, dtype=np.complex128).ravel()
        z[:vals.size] = vals
        w = self.fft_special_inv(z[None, :])[0]
        c = np.zeros(self.n, dtype=np.int64)
        c[:self.slots] = np.rint(scale * w.real).astype(np.int64)
        c[self.slots:] = np.rint(scale * w.imag).astype(np.int64)
        return c

    def decode(self, coeffs, scale):
        c = np.asarray(coeffs)
        re = np.array([float(v) for v in c[:self.slots]])
        im = np.array([float(v) for v in c[self.slots:]])
        w = (re + 1j * im) / scale
        return self.fft_special(w[None, :])[0]

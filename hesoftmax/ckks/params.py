from dataclasses import dataclass
from .numbertheory import ntt_primes, is_prime


@dataclass
class CkksParams:
    log_n: int
    log_scale: int
    log_q0: int
    n_levels: int
    log_p: int
    n_special: int
    dnum: int
    sigma: float = 3.2
    name: str = "P"

    @property
    def n(self):
        return 1 << self.log_n

    @property
    def slots(self):
        return 1 << (self.log_n - 1)

    def _near_prime(self, target, used):
        n = self.n
        step = 2 * n
        base = int(target) // step
        for off in range(1, 1 << 22):
            for cand in ((base + off) * step + 1, (base - off) * step + 1):
                if cand > 1 << 20 and is_prime(cand) and cand not in used:
                    return cand
        raise ValueError("no prime near target")

    def build_primes(self):
        used = set()
        q0 = ntt_primes(self.log_q0, self.n, 1)[0]
        used.add(q0)
        delta = float(2 ** self.log_scale)
        d = [0.0] * (self.n_levels + 1)
        qs = [0] * (self.n_levels + 1)
        d[self.n_levels] = delta
        for lv in range(self.n_levels, 0, -1):
            target = d[lv] * d[lv] / delta
            q = self._near_prime(target, used)
            used.add(q)
            qs[lv] = q
            d[lv - 1] = d[lv] * d[lv] / q
        chain = [q0] + [qs[i] for i in range(1, self.n_levels + 1)]
        ps = ntt_primes(self.log_p, self.n, self.n_special, skip=used)
        return chain, ps, [delta] + [d[i] for i in range(1, self.n_levels + 1)]

    def log_pq(self):
        return self.log_q0 + self.log_scale * self.n_levels + self.log_p * self.n_special


PARAM_MAIN = CkksParams(log_n=16, log_scale=40, log_q0=50, n_levels=30,
                        log_p=45, n_special=11, dnum=3, name="H128")

PARAM_TEST = CkksParams(log_n=13, log_scale=40, log_q0=50, n_levels=16,
                        log_p=45, n_special=9, dnum=3, name="TEST")

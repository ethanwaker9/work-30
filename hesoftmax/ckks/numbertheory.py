import numpy as np

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def ntt_primes(bits, n_ring, count, skip=()):
    out = []
    k = (1 << bits) // (2 * n_ring)
    while len(out) < count:
        cand = k * 2 * n_ring + 1
        if cand.bit_length() == bits and is_prime(cand) and cand not in skip:
            out.append(cand)
        k -= 1
        if k <= 0:
            raise ValueError("not enough NTT primes")
    return out


def factorize(n):
    fs, d = [], 2
    while d * d <= n:
        if n % d == 0:
            fs.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        fs.append(n)
    return fs


def primitive_root(p):
    fs = factorize(p - 1)
    for g in range(2, 1 << 20):
        if all(pow(g, (p - 1) // f, p) != 1 for f in fs):
            return g
    raise ValueError("no primitive root")


def bit_reverse_table(n):
    w = n.bit_length() - 1
    idx = np.arange(n)
    out = np.zeros(n, dtype=np.int64)
    for b in range(w):
        out |= ((idx >> b) & 1) << (w - 1 - b)
    return out

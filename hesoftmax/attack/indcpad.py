import argparse
import json
import os
import time
import numpy as np

from ..ckks.params import CkksParams
from ..ckks.scheme import CkksContext


def exact_key_recovery(params, seed=0):
    ctx = CkksContext(params, seed=seed)
    ctx.keygen()
    R = ctx.ring
    slots = ctx.encoder.slots
    rng = np.random.default_rng(seed + 5)
    msg = rng.uniform(-0.5, 0.5, size=slots)
    ct = ctx.encrypt_sk(msg)
    idx = ctx.lev_idx(ct.level)
    t0 = time.time()
    released = ctx.decrypt(ct)
    pt = R.ntt(R.from_int(ctx.encoder.encode(released, ct.scale), idx), idx)
    rhs = R.sub(pt, ct.c0, idx)
    inv = np.zeros_like(ct.c1)
    for a, i in enumerate(idx):
        p = R.primes[i]
        inv[a] = np.array([pow(int(v), p - 2, p) for v in ct.c1[a]], dtype=np.uint64)
    s_hat = R.mulmod(rhs, inv, idx)
    coeffs = R.to_int_small(R.intt(s_hat, idx), idx, take=min(3, len(idx)))
    dt = time.time() - t0
    rec = np.array([int(v) for v in coeffs])
    ok = bool(np.array_equal(rec, np.asarray(ctx.sk_coeffs, dtype=np.int64)))
    return {"recovered": ok, "queries": 1, "time_s": dt,
            "hamming": int(np.sum(rec != np.asarray(ctx.sk_coeffs, dtype=np.int64)))}


def residual_noise(t, sigma1, sigma2):
    return sigma1 * sigma2 / np.sqrt(t ** 2 * sigma1 ** 2 + sigma2 ** 2)


def flooding_rules(sigma1, t, kappa=128, n=2 ** 16, static=None):
    avg = sigma1 * np.sqrt(t)
    wc = (static if static is not None else sigma1) * t * 8.0 * n * 2.0 ** (kappa / 2)
    return {"average": avg, "worst": wc}


def sweep(sigma1=3.2, ts=(1, 4, 16, 64, 256, 1024, 4096), kappa=128, n=2 ** 16):
    out = []
    for t in ts:
        r = flooding_rules(sigma1, t, kappa, n)
        out.append({"t": int(t),
                    "avg_sigma": float(r["average"]),
                    "avg_residual": float(residual_noise(t, sigma1, r["average"])),
                    "wc_sigma": float(r["worst"]),
                    "wc_residual": float(residual_noise(t, sigma1, r["worst"]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/attack.json")
    a = ap.parse_args()
    prm = CkksParams(log_n=13, log_scale=40, log_q0=50, n_levels=4, log_p=45, n_special=4,
                     dnum=1, name="ATK")
    res = {"exact": exact_key_recovery(prm), "sweep": sweep()}
    print(json.dumps(res["exact"]))
    for r in res["sweep"]:
        print(r)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

import argparse
import json
import os
import time
import numpy as np
import psutil

from ..ckks.params import CkksParams
from ..ckks.scheme import CkksContext
from ..softmax.common import Packing, Runtime, softmax_ref
from ..softmax import ours as O
from ..softmax import baselines as B

METHODS = ["HETAL", "NEXUS", "THOR", "Cho24", "ACS"]


def build_context(params, n, seed=7, need_neg=False):
    ctx = CkksContext(params, seed=seed)
    pk = Packing(ctx.encoder.slots, n)
    steps = []
    for j in range(int(np.log2(n))):
        steps.append(pk.batch * (2 ** j))
        if need_neg:
            steps.append(-pk.batch * (2 ** j))
    steps = sorted(set(s % ctx.encoder.slots for s in steps))
    ctx.keygen(rotations=steps)
    return ctx, pk, steps


def evk_bytes(ctx, n_rot):
    n_limb = ctx.n_q + ctx.n_p
    per_key = ctx.params.dnum * n_limb * ctx.params.n * 8
    return per_key * (1 + n_rot)


def run_method(name, ctx, pk, X, spread, alpha, refresh_at):
    rt = Runtime(ctx, pk, refresh_at=refresh_at)
    ct = ctx.encrypt_sk(pk.pack(X))
    before = ctx.counter.snapshot()
    lvl0 = ct.level
    t0 = time.time()
    full = 2.0 * spread
    if name == "ACS":
        cfg = O.plan(pk.n, spread, alpha)
        out, _ = O.acs_softmax(rt, ct, cfg)
    elif name == "Cho24":
        cfg = B.cho_plan(pk.n, full, alpha)
        out, _ = B.cho_softmax(rt, ct, cfg, spread)
    elif name == "THOR":
        cfg = B.thor_plan(pk.n, full, alpha)
        out, _ = B.thor_softmax(rt, ct, cfg, 0.0)
    elif name == "NEXUS":
        cfg = B.nexus_plan(pk.n, full, alpha)
        out, _ = B.nexus_softmax(rt, ct, cfg, spread)
    elif name == "HETAL":
        cfg = B.hetal_plan(pk.n, full, alpha)
        out, _ = B.hetal_softmax(rt, ct, cfg, full)
    else:
        raise ValueError(name)
    dt = time.time() - t0
    got = pk.unpack(ctx.decrypt(out))
    ref = softmax_ref(X)
    d = ctx.counter.diff(before)
    err = float(np.abs(got - ref).max())
    l1 = float(np.abs(got - ref).sum(axis=1).mean())
    return {
        "method": name, "time_s": dt, "err_inf": err, "err_l1": l1,
        "bits": float(-np.log2(max(err, 1e-300))),
        "levels": int(cfg["depth"]),
        "levels_used": int(rt.consumed + lvl0 - out.level),
        "mult_cc": d["mult_cc"], "mult_pt": d["mult_pt"], "rot": d["rot"],
        "keyswitch": d["keyswitch"], "boot": rt.boot,
        "rss_mb": psutil.Process(os.getpid()).memory_info().rss / 2 ** 20,
    }


def worker(args):
    try:
        return _worker(args)
    except Exception as exc:
        return {"method": args[0], "spread": args[3], "error": repr(exc)}


def _worker(args):
    name, pname, n, spread, alpha, seed, refresh_at = args
    params = PARAMS[pname]
    ctx, pk, steps = build_context(params, n, seed, need_neg=(name == "Cho24"))
    rng = np.random.default_rng(1234)
    X = rng.uniform(-spread, spread, size=(pk.batch, n))
    X -= X.mean(axis=1, keepdims=True)
    r = run_method(name, ctx, pk, X, spread, alpha, refresh_at)
    r["evk_mb"] = evk_bytes(ctx, len(steps)) / 2 ** 20
    r["n"] = n
    r["spread"] = spread
    r["batch"] = pk.batch
    return r


PARAMS = {
    "H128": CkksParams(log_n=16, log_scale=40, log_q0=50, n_levels=30, log_p=45, n_special=11,
                       dnum=3, name="H128"),
    "H128s": CkksParams(log_n=14, log_scale=40, log_q0=50, n_levels=30, log_p=45, n_special=11,
                        dnum=3, name="H128s"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--spreads", type=float, nargs="+", default=[16.0, 128.0])
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--param", default="H128")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--out", default="results/microbench.json")
    a = ap.parse_args()
    tasks = [(m, a.param, a.n, s, a.alpha, 7, 1) for s in a.spreads for m in a.methods]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    res = []
    if os.path.exists(a.out):
        try:
            res = json.load(open(a.out))
        except Exception:
            res = []
    done = {(r.get("method"), r.get("spread")) for r in res}
    for t in tasks:
        if (t[0], t[3]) in done:
            continue
        r = worker(t)
        print(json.dumps(r), flush=True)
        res.append(r)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

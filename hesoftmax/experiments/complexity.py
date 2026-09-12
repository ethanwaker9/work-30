import argparse
import json
import os
import numpy as np

from ..softmax import ours as O
from ..softmax import baselines as B
from ..polyapprox.chebyshev import poly_depth


def ps_mults(d):
    if d < 1:
        return 0, 0
    r = int(np.ceil(np.log2(d + 1)))
    kb = 1 << ((r + 1) // 2)
    m = r - ((r + 1) // 2)
    kb = min(kb, d)
    baby = kb - 1
    giant = max(m - 1, 0) + 1
    body = max(2 ** m - 1, 0)
    return baby + giant + body, r


def acs_counts(n, spread, alpha):
    cfg = O.plan(n, spread, alpha)
    ln = int(np.log2(n))
    mults, _ = ps_mults(cfg["deg_exp"])
    rot = ln
    depth = poly_depth(cfg["deg_exp"])
    for hi, dj, tol in cfg["stages"]:
        mm, _ = ps_mults(dj)
        mults += mm + 3
        rot += ln
        depth += 1 + poly_depth(dj) + 2
    mf, _ = ps_mults(cfg["deg_fin"])
    mults += mf + 1
    rot += ln
    depth += poly_depth(cfg["deg_fin"]) + 1
    return {"method": "ACS", "mult": mults, "rot": rot, "depth": depth, "cfg": cfg}


def cho_counts(n, spread, alpha):
    cfg = B.cho_plan(n, 2.0 * spread, alpha)
    ln = int(np.log2(n))
    k = cfg["k"]
    mults, _ = ps_mults(cfg["deg_exp"])
    rot = 0
    depth = poly_depth(cfg["deg_exp"])
    for j in range(k):
        if j == 0:
            d = cfg["deg_first"] if k > 1 else max(cfg["deg_first"], cfg["deg_last"])
        else:
            d = cfg["deg_last"] if j == k - 1 else cfg["deg_mid"]
        mm, _ = ps_mults(d)
        mults += mm + 4
        rot += 2 * ln
        depth += 1 + poly_depth(d) + 3
    return {"method": "Cho24", "mult": mults, "rot": rot, "depth": depth, "cfg": cfg}


def thor_counts(n, spread, alpha):
    cfg = B.thor_plan(n, 2.0 * spread, alpha)
    ln = int(np.log2(n))
    mults, _ = ps_mults(cfg["deg_exp"])
    mults += int(np.log2(cfg["d1"]))
    rot = 0
    depth = cfg["depth"]
    rounds = int(np.log2(cfg["d2"])) + 1
    for j in range(rounds):
        it = cfg["it0"] if j == 0 else cfg["it1"]
        mults += 2 * it + 1
        rot += ln
        if j < rounds - 1:
            mults += 1
    return {"method": "THOR", "mult": mults, "rot": rot, "depth": cfg["depth"], "cfg": cfg}


def nexus_counts(n, spread, alpha):
    cfg = B.nexus_plan(n, 2.0 * spread, alpha)
    ln = int(np.log2(n))
    it = cfg["gold"]
    mults = cfg["r"] + 2 * it + 1
    return {"method": "NEXUS", "mult": mults, "rot": ln, "depth": cfg["depth"], "cfg": cfg}


def quad_counts(n, spread, alpha):
    cfg = B.quad_plan(n, 2.0 * spread, alpha)
    ln = int(np.log2(n))
    mults = 1 + 2 * cfg["gold"] + 1
    return {"method": "2Quad", "mult": mults, "rot": ln, "depth": cfg["depth"], "cfg": cfg}


def hetal_counts(n, spread, alpha):
    cfg = B.hetal_plan(n, 2.0 * spread, alpha)
    ln = int(np.log2(n))
    per_sign, _ = ps_mults(7)
    mults = cfg["rounds"] * ((cfg["df"] + cfg["dg"]) * per_sign + 1)
    me, _ = ps_mults(cfg["deg_exp"])
    mults += me + 2 * cfg["gold"] + 1
    rot = cfg["rounds"] + ln
    return {"method": "HETAL", "mult": mults, "rot": rot, "depth": cfg["depth"], "cfg": cfg}


ALL = {"HETAL": hetal_counts, "NEXUS": nexus_counts, "2Quad": quad_counts,
       "THOR": thor_counts, "Cho24": cho_counts, "ACS": acs_counts}


def table(n, spreads, alpha=16):
    rows = []
    for s in spreads:
        for name, fn in ALL.items():
            r = fn(n, s, alpha)
            r.pop("cfg")
            r["spread"] = s
            r["n"] = n
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--spreads", type=float, nargs="+", default=[16.0, 105.0, 1103.5])
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--out", default="results/complexity.json")
    a = ap.parse_args()
    rows = table(a.n, a.spreads, a.alpha)
    for r in rows:
        print(json.dumps(r))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()

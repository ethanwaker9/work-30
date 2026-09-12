import argparse
import json
import warnings
import numpy as np

from ..softmax import ours as O
from ..softmax import baselines as B
from ..softmax import plaintext as P
from ..softmax.common import softmax_ref


def bits(Y, ref):
    with np.errstate(all="ignore"):
        e = float(np.abs(np.asarray(Y, dtype=float) - ref).max())
    if not np.isfinite(e) or e >= 1.0:
        return None
    return float(-np.log2(max(e, 1e-300)))


def sweep(n=128, alpha=16, spreads=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096), rows=256,
          seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for s in spreads:
        X = rng.uniform(-s, s, size=(rows, n))
        X -= X.mean(axis=1, keepdims=True)
        ref = softmax_ref(X)
        m = 2.0 * s
        runs = {
            "ACS": lambda: P.acs_pt(X, O.plan(n, float(s), alpha)),
            "Cho24": lambda: P.cho_pt(X, B.cho_plan(n, m, alpha), float(s)),
            "THOR": lambda: P.thor_pt(X, B.thor_plan(n, m, alpha), 0.0),
            "HETAL": lambda: P.hetal_pt(X, B.hetal_plan(n, m, alpha), m),
            "NEXUS": lambda: P.nexus_pt(X, B.nexus_plan(n, m, alpha), float(s)),
            "2Quad": lambda: P.quad_pt(X, B.quad_plan(n, m, alpha), float(s)),
        }
        row = {"spread": s}
        for name, fn in runs.items():
            try:
                row[name] = bits(fn(), ref)
            except (ValueError, FloatingPointError, OverflowError):
                row[name] = None
        out.append(row)
        print(row, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/accuracy_sweep.json")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    res = sweep()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

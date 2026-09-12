import argparse
import json
import os
import numpy as np

from ..bounds.gpt2 import load_weights, config
from ..bounds.gpt2_forward import Gpt2
from ..bounds.ranges import model_range_bound
from ..softmax import ours as O
from ..softmax import baselines as B
from ..softmax import masked as M


def calibrate(g, windows):
    cap = {}
    stats = {"max": [0.0] * g.cfg["L"], "spread": [0.0] * g.cfg["L"]}
    for ids in windows:
        cap.clear()
        g.forward(ids, capture=cap)
        for l, att, mask in cap["logits"]:
            z = np.where(mask[None], att, -np.inf)
            mx = float(np.max(z))
            mn = float(np.min(np.where(mask[None], att, np.inf)))
            stats["max"][l] = max(stats["max"][l], mx)
            stats["spread"][l] = max(stats["spread"][l], mx - mn)
    return stats


def violation_rate(g, windows, anchors):
    cap = {}
    bad = 0
    tot = 0
    worst = 0.0
    for ids in windows:
        cap.clear()
        g.forward(ids, capture=cap)
        for l, att, mask in cap["logits"]:
            z = np.where(mask[None], att, -np.inf)
            rmax = z.max(axis=-1)
            over = rmax - anchors[l]
            bad += int((over > 0).sum())
            tot += over.size
            worst = max(worst, float(over.max()))
    return bad / max(tot, 1), worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--calib", type=int, default=8)
    ap.add_argument("--windows", type=int, default=24)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--s-meas", type=float, default=105.0)
    ap.add_argument("--out", default="results/gpt2_accuracy.json")
    a = ap.parse_args()
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    w = load_weights()
    cfg = config()
    g = Gpt2(w, cfg)
    rb = model_range_bound(w, cfg)
    s_prov = float(np.median([b["logit_spread"] for b in rb["blocks"]]))
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text[:600000]).input_ids
    n = a.ctx
    allw = [ids[i * n:(i + 1) * n] for i in range((len(ids) // n))]
    cal = allw[:a.calib]
    test = allw[a.calib:a.calib + a.windows]
    st = calibrate(g, cal)
    anchors = st["max"]
    s_emp = float(a.s_meas)
    vr, worst = violation_rate(g, test, anchors)
    res = {"spread_emp": s_emp, "spread_prov": s_prov, "ctx": n, "windows": a.windows,
           "calib_windows": a.calib, "anchors": anchors,
           "violation_rate": vr, "worst_overshoot": worst}
    base = np.concatenate([g.nll(x) for x in test])
    ppl0 = float(np.exp(base.mean()))
    res["exact"] = {"ppl": ppl0, "depth": None}
    print("exact", ppl0, "viol", vr, "worst", worst, flush=True)
    plans = {
        "ACS-prov": ("ACS", O.plan(n, s_prov, a.alpha), s_prov, None),
        "ACS-emp": ("ACS", O.plan(n, s_emp, a.alpha), s_emp, None),
        "Cho24": ("Cho24", B.cho_plan(n, 2 * s_emp, a.alpha), 2 * s_emp, anchors),
        "THOR": ("THOR", B.thor_plan(n, 2 * s_emp, a.alpha), 2 * s_emp, anchors),
        "NEXUS": ("NEXUS", B.nexus_plan(n, 2 * s_emp, a.alpha), 2 * s_emp, anchors),
        "HETAL": ("HETAL", B.hetal_plan(n, 2 * s_emp, a.alpha), 2 * s_emp, None),
        "2Quad": ("2Quad", B.quad_plan(n, 2 * s_emp, a.alpha), 2 * s_emp, anchors),
    }
    for name, (kind, mcfg, sp, anc) in plans.items():
        if kind == "ACS":
            fn = lambda at, m, l, c=mcfg: M.acs_masked(at, m, c)
        elif kind == "Cho24":
            fn = lambda at, m, l, c=mcfg, an=anc: M.cho_masked(at, m, c, an[l])
        elif kind == "THOR":
            fn = lambda at, m, l, c=mcfg, an=anc: M.thor_masked(at, m, c, an[l] - c["spread"] / 2)
        elif kind == "NEXUS":
            fn = lambda at, m, l, c=mcfg, an=anc: M.nexus_masked(at, m, c, an[l])
        elif kind == "HETAL":
            fn = lambda at, m, l, c=mcfg, s=sp: M.hetal_masked(at, m, c, s)
        else:
            fn = lambda at, m, l, c=mcfg, an=anc: M.quad_masked(at, m, c, an[l])
        v = np.concatenate([g.nll(x, fn) for x in test])
        p = float(np.exp(v.mean()))
        res[name] = {"ppl": p, "depth": int(mcfg["depth"]), "d_ppl": p - ppl0}
        print(name, res[name], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

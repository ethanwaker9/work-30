import argparse
import json
import os
import numpy as np

from ..bounds.gpt2 import load_weights, config
from ..bounds.gpt2_forward import Gpt2
from ..bounds.ranges import model_range_bound


def empirical_spreads(g, windows):
    cap = {}
    out = [0.0] * g.cfg["L"]
    for ids in windows:
        cap.clear()
        g.forward(ids, capture=cap)
        for l, att, mask in cap["logits"]:
            z = np.where(mask[None], att, np.nan)
            mu = np.nanmean(z, axis=-1, keepdims=True)
            dev = np.nanmax(np.abs(z - mu))
            out[l] = max(out[l], float(dev))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--out", default="results/range_bounds.json")
    a = ap.parse_args()
    from transformers import GPT2TokenizerFast
    from datasets import load_dataset
    w = load_weights()
    cfg = config()
    rb = model_range_bound(w, cfg)
    g = Gpt2(w, cfg)
    tok = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text[:400000]).input_ids
    n = a.ctx
    wins = [ids[i * n:(i + 1) * n] for i in range(a.windows)]
    rb["empirical"] = empirical_spreads(g, wins)
    rb["ratio"] = [c / max(e, 1e-9) for c, e in
                     zip([b["logit_spread"] for b in rb["blocks"]], rb["empirical"])]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rb, f, indent=1)
    print("provable median %.1f  empirical max %.1f  ratio median %.1f" % (
        float(np.median([b["logit_spread"] for b in rb["blocks"]])),
        float(np.max(rb["empirical"])), float(np.median(rb["ratio"]))))


if __name__ == "__main__":
    main()

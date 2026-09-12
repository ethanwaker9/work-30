import numpy as np


def spectral(m):
    return float(np.linalg.norm(m, 2))


def ln_bound(gamma, beta, d):
    return float(np.abs(gamma).max() * np.sqrt(d) + np.linalg.norm(beta))


def ln_coord_bound(gamma, beta, d):
    return float((np.abs(gamma) * np.sqrt(d) + np.abs(beta)).max())


def logit_spread_bound(gamma, beta, wq, wk, dh, d):
    proj = np.eye(d) - np.ones((d, d)) / d
    g = np.diag(gamma)
    a = wq @ wk.T / np.sqrt(dh)
    core = proj @ (g @ a @ g) @ proj
    s_core = spectral(core)
    cross = np.linalg.norm(proj @ (g @ a.T @ beta))
    return float(2.0 * d * s_core + 2.0 * np.sqrt(d) * cross)


def block_range_bound(w, cfg, layer, r_in):
    d, h, dh = cfg["d"], cfg["h"], cfg["dh"]
    g1 = w["h.%d.ln_1.weight" % layer]
    b1 = w["h.%d.ln_1.bias" % layer]
    g2 = w["h.%d.ln_2.weight" % layer]
    b2 = w["h.%d.ln_2.bias" % layer]
    cat = w["h.%d.attn.c_attn.weight" % layer]
    bcat = w["h.%d.attn.c_attn.bias" % layer]
    wq, wk, wv = cat[:, :d], cat[:, d:2 * d], cat[:, 2 * d:]
    bv = bcat[2 * d:]
    wo = w["h.%d.attn.c_proj.weight" % layer]
    bo = w["h.%d.attn.c_proj.bias" % layer]
    w1 = w["h.%d.mlp.c_fc.weight" % layer]
    b1f = w["h.%d.mlp.c_fc.bias" % layer]
    w2 = w["h.%d.mlp.c_proj.weight" % layer]
    b2f = w["h.%d.mlp.c_proj.bias" % layer]
    r1 = ln_bound(g1, b1, d)
    spreads = []
    for hh in range(h):
        q = wq[:, hh * dh:(hh + 1) * dh]
        k = wk[:, hh * dh:(hh + 1) * dh]
        spreads.append(logit_spread_bound(g1, b1, q, k, dh, d))
    v_bound = spectral(wv) * r1 + np.linalg.norm(bv)
    attn = spectral(wo) * v_bound + np.linalg.norm(bo)
    r2 = ln_bound(g2, b2, d)
    gelu_in = float(np.max(np.linalg.norm(w1, axis=0) * r2 + np.abs(b1f)))
    v_norm = spectral(w1) * r2 + float(np.linalg.norm(b1f))
    gelu_out = v_norm + 0.17 * np.sqrt(cfg["d_ff"])
    mlp = spectral(w2) * gelu_out + float(np.linalg.norm(b2f))
    return {
        "layer": layer, "r_in": r_in, "ln1": r1, "ln2": r2,
        "logit_spread": float(max(spreads)), "logit_spread_mean": float(np.mean(spreads)),
        "attn_out": float(attn), "gelu_in": gelu_in, "mlp_out": float(mlp),
        "r_out": float(r_in + attn + mlp),
        "var_max": float((r_in + attn + mlp) ** 2 / cfg["d"]),
    }


def model_range_bound(w, cfg):
    d = cfg["d"]
    e = w["wte.weight"]
    p = w["wpe.weight"]
    r0 = float(np.linalg.norm(e, axis=1).max() + np.linalg.norm(p, axis=1).max())
    out = []
    r = r0
    for l in range(cfg["L"]):
        c = block_range_bound(w, cfg, l, r)
        out.append(c)
        r = c["r_out"]
    return {"r0": r0, "blocks": out, "r_final": r}

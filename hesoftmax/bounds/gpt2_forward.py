import numpy as np


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    v = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return g * (x - mu) / np.sqrt(v + eps) + b


def exact_softmax_masked(logits, mask):
    z = np.where(mask, logits, -np.inf)
    m = np.max(z, axis=-1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=-1, keepdims=True)


class Gpt2:
    def __init__(self, w, cfg):
        self.w = {k: v.astype(np.float64) for k, v in w.items()}
        self.cfg = cfg

    def forward(self, ids, softmax_fn=None, capture=None):
        w, cfg = self.w, self.cfg
        d, h, dh, L = cfg["d"], cfg["h"], cfg["dh"], cfg["L"]
        t = len(ids)
        x = w["wte.weight"][ids] + w["wpe.weight"][:t]
        mask = np.tril(np.ones((t, t), dtype=bool))
        for l in range(L):
            u = layer_norm(x, w["h.%d.ln_1.weight" % l], w["h.%d.ln_1.bias" % l])
            qkv = u @ w["h.%d.attn.c_attn.weight" % l] + w["h.%d.attn.c_attn.bias" % l]
            q, k, v = qkv[:, :d], qkv[:, d:2 * d], qkv[:, 2 * d:]
            q = q.reshape(t, h, dh).transpose(1, 0, 2)
            k = k.reshape(t, h, dh).transpose(1, 0, 2)
            v = v.reshape(t, h, dh).transpose(1, 0, 2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
            if capture is not None:
                capture.setdefault("logits", []).append((l, att.copy(), mask.copy()))
            if softmax_fn is None:
                p = exact_softmax_masked(att, mask[None])
            else:
                p = softmax_fn(att, mask[None], l)
            o = (p @ v).transpose(1, 0, 2).reshape(t, d)
            x = x + o @ w["h.%d.attn.c_proj.weight" % l] + w["h.%d.attn.c_proj.bias" % l]
            u2 = layer_norm(x, w["h.%d.ln_2.weight" % l], w["h.%d.ln_2.bias" % l])
            f = gelu(u2 @ w["h.%d.mlp.c_fc.weight" % l] + w["h.%d.mlp.c_fc.bias" % l])
            x = x + f @ w["h.%d.mlp.c_proj.weight" % l] + w["h.%d.mlp.c_proj.bias" % l]
        x = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"])
        return x @ w["wte.weight"].T

    def nll(self, ids, softmax_fn=None):
        lg = self.forward(ids, softmax_fn)
        lg = lg[:-1]
        tgt = np.asarray(ids[1:])
        m = lg.max(axis=-1, keepdims=True)
        ls = lg - m - np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))
        return -ls[np.arange(len(tgt)), tgt]

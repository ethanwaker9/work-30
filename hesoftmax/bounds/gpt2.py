import os
import numpy as np

MODEL = "openai-community/gpt2"


def load_weights(model=MODEL, cache=None):
    from safetensors.numpy import load_file
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model, "model.safetensors", cache_dir=cache)
    return load_file(path)


def config(model=MODEL):
    return {"d": 768, "h": 12, "dh": 64, "L": 12, "d_ff": 3072, "vocab": 50257, "n_ctx": 1024}

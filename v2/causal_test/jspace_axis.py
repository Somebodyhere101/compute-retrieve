"""Exploratory: is the compute-retrieve axis a J-space direction? CPU only.

For a direction a at layer l, the Jacobian-lens score of token t is v_l(t) . a, with
v_l(t) = J_l^T (gamma * W_U[t]); so scores = W_U (gamma * (J_l a)). We list the tokens each
direction promotes and suppresses, and measure the share of the direction's norm that lies in the
span of its own top-k lens vectors, against random directions put through the same procedure.

python jspace_axis.py <model key>   (1.7b | 8b)
"""

import json
import os
import sys

import jlens
import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = sys.argv[1]
MODEL = {"1.7b": "Qwen/Qwen3-1.7B", "8b": "Qwen/Qwen3-8B", "14b": "Qwen/Qwen3-14B"}[KEY]
LENS = {"1.7b": "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt",
        "8b": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt",
        "14b": "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"}[KEY]
K = 25
torch.manual_seed(0)

cal = torch.load(os.path.join(HERE, "res", f"{KEY}_cal.pt"), weights_only=False)
layer = cal["layer"]
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
snap = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json"])
W_U = gamma = None
for f in sorted(os.listdir(snap)):
    if not f.endswith(".safetensors"):
        continue
    with safe_open(os.path.join(snap, f), "pt") as st:
        keys = set(st.keys())
        if "lm_head.weight" in keys:
            W_U = st.get_tensor("lm_head.weight").float()
        elif W_U is None and "model.embed_tokens.weight" in keys:
            W_U = st.get_tensor("model.embed_tokens.weight").float()  # tied embeddings
        if "model.norm.weight" in keys:
            gamma = st.get_tensor("model.norm.weight").float()
lens = jlens.JacobianLens.from_pretrained("neuronpedia/jacobian-lens", filename=LENS)
band = sorted(lens.jacobians)
J = lens.jacobians[layer].float()
d = J.shape[0]
print(f"{MODEL}: axis layer {layer}; lens layers {band[0]}-{band[-1]}; d={d}", flush=True)
GW = W_U * gamma  # rows: gamma * W_U[t]


def scores(a):
    return GW @ (J @ a)


def captured(a, k=K):
    """Share of |a| in the span of the J-lens vectors of a's top-k tokens by |score|."""
    s = scores(a)
    idx = s.abs().topk(k).indices
    V = GW[idx] @ J  # [k, d]: v(t) = J^T (gamma * W_U[t])
    Q, _ = torch.linalg.qr(V.T)
    return float((Q.T @ a).norm() / a.norm())


def top_tokens(a, n=15):
    s = scores(a)
    up = [tok.decode([int(i)]) for i in s.topk(n).indices]
    down = [tok.decode([int(i)]) for i in (-s).topk(n).indices]
    return up, down


out = {"model": MODEL, "layer": layer, "lens_layers": [band[0], band[-1]], "k": K}
for name, vec in (("cr_axis", cal["axis"]), ("length_axis", cal["len_axis"])):
    a = torch.tensor(np.asarray(vec), dtype=torch.float32)
    up, down = top_tokens(a)
    out[name] = dict(captured=captured(a), toward_positive=up, toward_negative=down)
    print(f"\n{name}: share of norm in its top-{K} lens directions = {out[name]['captured']:.3f}")
    print("  promotes (compute / long side):", up)
    print("  suppresses (retrieve / short side):", down)
null = [captured(torch.randn(d)) for _ in range(200)]
out["random_null"] = dict(mean=float(np.mean(null)), p95=float(np.percentile(null, 95)), max=float(np.max(null)))
print(f"\nrandom directions, same procedure: mean {np.mean(null):.3f}, 95th percentile {np.percentile(null, 95):.3f}, max {np.max(null):.3f}")
cos = float(torch.nn.functional.cosine_similarity(torch.tensor(np.asarray(cal['axis']), dtype=torch.float32), torch.tensor(np.asarray(cal['len_axis']), dtype=torch.float32), dim=0))
out["cos_axis_len"] = cos
json.dump(out, open(os.path.join(HERE, "res", f"{KEY}_jspace_axis.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)

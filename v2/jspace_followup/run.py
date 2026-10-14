"""Does the CR axis act through the J-space? (PREREGISTRATION.md)

python run.py <model key> main|pilot
Uses paper 3's Rig (model, Jacobian lens, hook controller) for J-space removal, and paper 4's
calibrated CR axis. Three generation calls per item: A nothing removed (clean, axis, a_J, a_perp,
8 random), B J-space removed (clean, axis), C matched random removal (clean, axis). Resumable.
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")  # the v2/ folder; core.py (Rig) sits next to this script
sys.path.insert(0, os.path.join(ROOT, "causal_test"))
from core import Rig, model_spec  # noqa: E402
from cr4lib import _format_prompt, shown_working, NUM  # noqa: E402

_spec = importlib.util.spec_from_file_location("p4_items", os.path.join(ROOT, "causal_test", "items.py"))
p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p4)

KEY, PHASE = sys.argv[1], sys.argv[2]
N_NEW, K, R = 96, 25, 8
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


MODELS = {"1.7b": dict(model="Qwen/Qwen3-1.7B", lens_file="qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt", gpu_layers=None),
          "8b": dict(model="Qwen/Qwen3-8B", lens_file="qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt", gpu_layers=None)}
rig = Rig(**model_spec(MODELS, KEY))
tok, dev = rig.tok, rig.device
eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
cal = torch.load(os.path.join(ROOT, "causal_test", "res", f"{KEY}_cal.pt"), weights_only=False)
LAYER, GAP = cal["layer"], cal["gap"]
axis = torch.tensor(np.asarray(cal["axis"]), dtype=torch.float32, device=dev)
axis = axis / axis.norm()

# J-space component of the axis: span of the lens vectors of its top-K tokens at LAYER
with torch.no_grad():
    GW = rig.W_U.float() * rig.gamma
    J = (rig.J[LAYER] if LAYER in rig.J else rig.J_cpu[LAYER].to(dev)).float()
    sc = GW @ (J @ axis)
    idx = sc.abs().topk(K).indices
    V = GW[idx] @ J
    Q, _ = torch.linalg.qr(V.T)
    a_J = Q @ (Q.T @ axis)
    a_perp = axis - a_J
    share = float(a_J.norm())
    a_J, a_perp = a_J / a_J.norm(), a_perp / a_perp.norm()
g = torch.Generator().manual_seed(1234)
Rnd = torch.randn(12, rig.d, generator=g, dtype=torch.float64)[:R]
Rnd = (Rnd / Rnd.norm(dim=1, keepdim=True)).float().to(dev)
log(f"{rig.model_name}: layer {LAYER}, gap {GAP:.1f}, band {rig.band[0]}-{rig.band[-1]}, a_J share of axis norm {share:.3f}")

CALLS = {
    "A": [("clean", None), ("axis", -axis), ("a_J", -a_J), ("a_perp", -a_perp)] + [(f"rand{j}", Rnd[j]) for j in range(R)],
    "B": [("clean", None), ("axis", -axis)],
    "C": [("clean", None), ("axis", -axis)],
}
ABL = {"A": None, "B": "jspace", "C": "matched"}


@torch.no_grad()
def generate(user, rows, abl, seed):
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text)["input_ids"]
    det = tok(_format_prompt(user, tok))["input_ids"]
    assert ids[: len(det)] == det
    start = len(det) - 1
    B = len(rows)
    vecs = torch.stack([GAP * v if v is not None else torch.zeros(rig.d, device=dev) for _, v in rows]).to(torch.bfloat16)
    ctl = rig.ctl
    ctl.reset()
    ctl.injects = [{"vecs": {LAYER: vecs}, "span": (start, 10 ** 9)}]
    if abl:
        ctl.ablate = {"layers": set(rig.band), "a_start": start, "mode": abl, "k": K}
    torch.manual_seed(seed)
    x = torch.tensor([ids] * B, device=dev)
    cache = transformers.DynamicCache()
    T, out = x.shape[1], []
    eos_t = torch.tensor(sorted(eos), device=dev)
    fin = torch.zeros(B, dtype=torch.bool, device=dev)
    try:
        for step in range(N_NEW):
            pos = torch.arange(T - x.shape[1], T, device=dev)[None].expand(B, -1)
            o = rig.hf(input_ids=x, past_key_values=cache, use_cache=True, position_ids=pos)
            nxt = o.logits[:, -1].argmax(-1)
            out.append(nxt)
            x = nxt[:, None]
            T += 1
            fin |= torch.isin(nxt, eos_t)
            if bool(fin.all()):
                break
    finally:
        ctl.reset()
    toks = torch.stack(out, 1).tolist()
    res = []
    for b in range(B):
        t = toks[b]
        for i, v in enumerate(t):
            if v in eos:
                t = t[:i]
                break
        res.append(tok.decode(t, skip_special_tokens=True))
    return res


def score(text, it):
    W, _ = shown_working(text, it["a"], it["b"])
    nums = [int(n.replace(",", "")) for n in NUM.findall(text)]
    rest = [n for n in nums if n not in (it["a"], it["b"])]
    return dict(W=W, AF=int(bool(rest) and rest[0] == it["answer"]))


S = p4.build()
items = [it for it in S["main"] if it["sub"] == "novel"] if PHASE == "main" else [it for it in S["pilot"] if it["sub"] == "novel"][:6]
path = os.path.join(RES, f"{KEY}_{PHASE}.jsonl")
done = {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()
todo = [it for it in items if it["id"] not in done]
log(f"{len(todo)} of {len(items)} items to run")
with open(path, "a", encoding="utf-8") as f:
    for n, it in enumerate(todo):
        rec = dict(it, share_aJ=share)
        for call, rows in CALLS.items():
            texts = generate(it["prompt"], rows, ABL[call], seed=1000 + items.index(it))
            rec[call] = [dict(cond=name, text=t, **score(t, it)) for (name, _), t in zip(rows, texts)]
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        if n % 20 == 0:
            log(n, it["id"], {c: [(r["cond"], r["AF"]) for r in rec[c][:4]] for c in rec if c in CALLS})
log("DONE")

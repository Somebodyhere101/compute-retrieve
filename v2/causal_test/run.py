"""Paper 4 runs. python run.py <model key> pilot|main [tag]

For every item, one batch holds all conditions (clean, axis +-k, length axis +-k, random
directions at norms c * gap) on the same prompt, so rows differ only in the added vector.
Direct-answer runs (prefilled "{a} * {b} = ") give the answer, its settling depth and the KL
at the first answer position; free runs give the shown-working flag. Resumable.
"""

import json
import os
import sys
import time

import numpy as np
import torch

from cr4lib import (MODELS, CRDetector, Model, answer_span, error_type, settle_jsd, settle_top1,
                    shown_working)
from items import build

HERE = os.path.dirname(os.path.abspath(__file__))
KEY, PHASE = sys.argv[1], sys.argv[2]
TAG = sys.argv[3] if len(sys.argv) > 3 else PHASE  # config and output name
CFG = json.load(open(os.path.join(HERE, f"config_{TAG}.json"), encoding="utf-8"))
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


m = Model(MODELS[KEY], gpu_layers=int(os.environ["GPU_LAYERS"]) if os.environ.get("GPU_LAYERS") else None)
cal_path = os.path.join(RES, f"{KEY}_cal.pt")
if os.path.exists(cal_path):
    cal = torch.load(cal_path, weights_only=False)
else:
    cal = m.calibrate()
    torch.save(cal, cal_path)
    json.dump({k: v for k, v in cal.items() if not isinstance(v, np.ndarray)},
              open(os.path.join(RES, f"{KEY}_cal.json"), "w"), indent=1)
LAYER, GAP = cal["layer"], cal["gap"]
log(f"{MODELS[KEY]} L={m.L} layer={LAYER} auc={cal['auc']:.3f} gap={GAP:.1f} "
    f"cos(axis,len)={cal['cos_len']:+.3f} median|h|={cal['resid_norm_median']:.0f}")
det = CRDetector(m.hf, m.tok, device=m.dev)
det.axis, det.layer = cal["axis"], LAYER

# ---------- conditions ----------
conds, vecs = [dict(kind="clean")], [np.zeros(m.d)]
for k in CFG["k"]:
    for s in (1, -1):
        conds.append(dict(kind="axis", k=s * k))
        vecs.append(s * k * GAP * cal["axis"])
        conds.append(dict(kind="len", k=s * k))
        vecs.append(s * k * GAP * cal["len_axis"])
g = torch.Generator().manual_seed(1234)
R = torch.randn(CFG["R"], m.d, generator=g, dtype=torch.float64)
R = (R / R.norm(dim=1, keepdim=True)).numpy()
for j in range(CFG["R"]):
    for c in CFG["c"]:
        conds.append(dict(kind="rand", j=j, c=c))
        vecs.append(c * GAP * R[j])
V = torch.tensor(np.array(vecs), dtype=torch.float32)
json.dump(dict(conds=conds, cfg=CFG, layer=LAYER, gap=GAP, L=m.L, model=MODELS[KEY]),
          open(os.path.join(RES, f"{KEY}_{TAG}_conds.json"), "w"), indent=0)
log(f"{len(conds)} conditions per item")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")}


def trunc(toks):
    for i, t in enumerate(toks):
        if t in m.eos:
            return toks[:i]
    return toks


def direct(it, Vx):
    text = m.chat_text(it["prompt"]) + it["prefill"]
    r = m.generate(text, it["prompt"], Vx, LAYER, CFG["n_direct"], lens=True)
    lp0 = r["lp0"]
    kl = (lp0[0].exp() * (lp0[0] - lp0)).sum(-1).clamp_min(0).cpu().tolist()
    clean = trunc(r["tokens"][0].tolist())
    kl_all, kl_ans = m.tf_kl(text, it["prompt"], Vx, LAYER, clean, answer_span(m.tok, clean)[1])
    rows = []
    for b in range(Vx.shape[0]):
        toks = trunc(r["tokens"][b].tolist())
        ans, idx = answer_span(m.tok, toks)
        st = [settle_top1(r["top1"][b, i]) for i in idx]
        sj = [settle_jsd(r["jsd"][b, i]) for i in idx]
        row = dict(text=m.tok.decode(toks), ans=ans, correct=int(ans == it["answer"]), kl_first=kl[b],
                   kl_all=kl_all[b], kl_ans=kl_ans[b], kl=kl_all[b] if CFG.get("kl", "all") == "all" else kl[b],
                   n_dig=len(idx),
                   D=float(np.mean(st)) / m.L if st else None, D1=st[0] / m.L if st else None,
                   Djsd=float(np.mean(sj)) / m.L if sj else None, settle=st)
        row.update(error_type(ans, it["a"], it["b"], it["answer"]))
        rows.append(row)
    return rows


def free(it, Vx):
    r = m.generate(m.chat_text(it["prompt"]), it["prompt"], Vx, LAYER, CFG["n_free"], lens=False)
    rows = []
    for b in range(Vx.shape[0]):
        toks = trunc(r["tokens"][b].tolist())
        text = m.tok.decode(toks, skip_special_tokens=True)
        W, final = shown_working(text, it["a"], it["b"])
        rows.append(dict(text=text, W=W, final=final, correct=int(final == it["answer"]),
                         n_tok=len(toks), ended=int(len(toks) < CFG["n_free"])))
    return rows


def run(items, mode, Vx, tag):
    path = os.path.join(RES, f"{KEY}_{TAG}_{tag}.jsonl")
    done = done_ids(path)
    todo = [it for it in items if it["id"] not in done]
    log(f"{tag}: {len(todo)} of {len(items)} items to run")
    with open(path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            rec = dict(it, P=det.detect(it["prompt"]),
                       P_len=float(det._get_hidden(it["prompt"], LAYER) @ cal["len_axis"]))
            rec["rows"] = (direct if mode == "direct" else free)(it, Vx)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 20 == 0:
                c = rec["rows"][0]
                log(f"{tag} {n}/{len(todo)} {it['id']} clean ans={c['ans'] if mode == 'direct' else c['final']} "
                    f"{'D=%.2f' % c['D'] if mode == 'direct' and c['D'] is not None else ''}")


S = build()
items = S[PHASE]
run(items, "direct", V, "direct")
run(items, "free", V, "free")
if PHASE == "main":
    run(S["gradient"], "direct", V[:1], "gradient_direct")
    run(S["gradient"], "free", V[:1], "gradient_free")
log("DONE")

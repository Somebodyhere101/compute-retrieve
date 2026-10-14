"""J-space x CR axis test: preregistered J1, J2 and the replication check.

python analyze.py <model key> main|pilot  ->  res/<key>_<phase>_summary.json
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NB = 2000


def boot(vals):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)]


def main(key, phase):
    recs = [json.loads(l) for l in open(os.path.join(HERE, "res", f"{key}_{phase}.jsonl"), encoding="utf-8")]
    get = lambda r, call, cond, out: next(x[out] for x in r[call] if x["cond"] == cond)
    out = {"n": len(recs), "share_aJ": recs[0]["share_aJ"]}
    for o in ("AF", "W"):
        dJ = [get(r, "B", "axis", o) - get(r, "B", "clean", o) for r in recs]
        dM = [get(r, "C", "axis", o) - get(r, "C", "clean", o) for r in recs]
        dA = [get(r, "A", "axis", o) - get(r, "A", "clean", o) for r in recs]
        rand = [np.mean([get(r, "A", f"rand{j}", o) for j in range(8)]) - get(r, "A", "clean", o) for r in recs]
        aJ = [get(r, "A", "a_J", o) - get(r, "A", "clean", o) for r in recs]
        aP = [get(r, "A", "a_perp", o) - get(r, "A", "clean", o) for r in recs]
        out[o] = dict(
            J1=boot([a - b for a, b in zip(dJ, dM)]),
            J2=boot([a - b for a, b in zip(aJ, aP)]),
            push_effect_J_removed=boot(dJ), push_effect_random_removed=boot(dM),
            push_effect_intact=boot(dA), random_same_norm=boot(rand), axis_minus_random=boot([a - b for a, b in zip(dA, rand)]),
            a_J_effect=boot(aJ), a_perp_effect=boot(aP),
            rates={f"{c}:{cond}": float(np.mean([get(r, c, cond, o) for r in recs]))
                   for c, conds in (("A", ("clean", "axis", "a_J", "a_perp")), ("B", ("clean", "axis")), ("C", ("clean", "axis"))) for cond in conds},
        )
    json.dump(out, open(os.path.join(HERE, "res", f"{key}_{phase}_summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2])
    f = lambda e: f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]"
    print(f"n = {o['n']}; a_J share of axis norm {o['share_aJ']:.3f}")
    for m in ("AF", "W"):
        s = o[m]
        print(f"\n{m}: rates {json.dumps({k: round(v, 3) for k, v in s['rates'].items()})}")
        for k in ("J1", "J2", "push_effect_intact", "axis_minus_random", "push_effect_J_removed", "push_effect_random_removed", "a_J_effect", "a_perp_effect"):
            print(f"  {k:28s} {f(s[k])}")

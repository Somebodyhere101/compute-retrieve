"""Paper 4 analysis: preregistered tests H1-H5 and S, plus exploratory summaries.

python analyze.py <model key> pilot|main  ->  res/<key>_<phase>_summary.json
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

from cr4lib import NUM

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
NB = 2000
KL = "kl"  # size measure for matching: "kl" (= kl_all, Stage 2), or "kl_first" / "kl_ans" (sensitivity)


def load(key, phase, tag):
    p = os.path.join(RES, f"{key}_{phase}_{tag}.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def boot(vals_fn, n, seed=0):
    """Item bootstrap: vals_fn(idx) -> statistic. Returns (estimate, lo, hi)."""
    rng = np.random.default_rng(seed)
    est = vals_fn(np.arange(n))
    bs = [vals_fn(rng.integers(0, n, n)) for _ in range(NB)]
    bs = [b for b in bs if b is not None and np.isfinite(b)]
    if not bs or est is None or not np.isfinite(est):  # e.g. a constant variable: undefined
        return [None, None, None]
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def mean_ci(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 3:
        return None
    return boot(lambda i: x[i].mean(), len(x)) + [len(x)]


class Item:
    def __init__(self, rec, conds, rec_free=None):
        self.r = rec
        self.rows = rec["rows"]
        self.free = rec_free["rows"] if rec_free else None
        self.conds = conds
        self.idx = {}
        for n, c in enumerate(conds):
            key = (c["kind"], c.get("k"), c.get("j"), c.get("c"))
            self.idx[key] = n

    def row(self, kind, k=None, j=None, c=None):
        return self.idx[(kind, k, j, c)]

    def val(self, n, out):
        if out == "W":
            return None if self.free is None else self.free[n]["W"]
        if out == "AF":  # exploratory: the first number other than a and b is the correct product
            if self.free is None:
                return None
            nums = [int(x.replace(",", "")) for x in NUM.findall(self.free[n]["text"])]
            rest = [x for x in nums if x not in (self.r["a"], self.r["b"])]
            return int(bool(rest) and rest[0] == self.r["answer"])
        if out == "Wcorrect":
            return None if self.free is None else self.free[n]["correct"]
        return self.rows[n][out]

    def delta(self, n, out):
        v, v0 = self.val(n, out), self.val(0, out)
        return None if v is None or v0 is None else v - v0

    def rand_ref(self, kl_star, out, js, cs):
        """Mean over random directions of the outcome change interpolated (in log KL) at kl_star."""
        lk = math.log(kl_star + 1e-9)
        refs = []
        for j in js:
            pts = [(self.rows[self.row("rand", j=j, c=c)][KL], self.delta(self.row("rand", j=j, c=c), out))
                   for c in cs]
            for (k1, d1), (k2, d2) in zip(pts, pts[1:]):
                if k1 <= kl_star <= k2:
                    if d1 is None or d2 is None:
                        break
                    l1, l2 = math.log(k1 + 1e-9), math.log(k2 + 1e-9)
                    w = 0.5 if l2 == l1 else (lk - l1) / (l2 - l1)
                    refs.append(d1 + w * (d2 - d1))
                    break
        return (float(np.mean(refs)) if refs else None), len(refs)


def effect(items, kind, k, out, js, cs):
    """Per item: (intervention change - matched-KL random reference)."""
    diffs, raw, refs, n_out = [], [], [], 0
    for it in items:
        n = it.row(kind, k=k)
        d = it.delta(n, out)
        ref, nref = it.rand_ref(it.rows[n][KL], out, js, cs)
        if d is None or ref is None:
            n_out += 1
            diffs.append(None)
            continue
        diffs.append(d - ref)
        raw.append(d)
        refs.append(ref)
    return diffs, dict(effect=mean_ci(diffs), raw_change=mean_ci(raw), random_ref=mean_ci(refs),
                       n_excluded=n_out, n=len(items))


def paired_diff(a, b):
    x = [u - v for u, v in zip(a, b) if u is not None and v is not None]
    return mean_ci(x)


def partial_spearman(x, y, covs):
    rx, ry = rankdata(x), rankdata(y)
    C = np.column_stack([np.ones(len(x))] + [rankdata(c) for c in covs])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def main(key, phase, kl="kl"):
    global KL
    KL = kl
    meta = json.load(open(os.path.join(RES, f"{key}_{phase}_conds.json"), encoding="utf-8"))
    conds, cfg = meta["conds"], meta["cfg"]
    js, cs = list(range(cfg["R"])), cfg["c"]
    D = load(key, phase, "direct")
    F = {r["id"]: r for r in load(key, phase, "free")}
    items = [Item(r, conds, F.get(r["id"])) for r in D]
    by = lambda f: [it for it in items if f(it.r)]
    facts, Mall = by(lambda r: r["set"] == "F"), by(lambda r: r["set"] == "M")
    novel = by(lambda r: r["sub"] == "novel")
    out = dict(model=meta["model"], layer=meta["layer"], gap=meta["gap"], L=meta["L"], cfg=cfg,
               n=dict(F=len(facts), M=len(Mall), novel=len(novel), free=len(F)))
    cal = json.load(open(os.path.join(RES, f"{key}_cal.json"), encoding="utf-8"))
    out["calibration"] = {k: cal[k] for k in ("auc", "layer", "gap", "cos_len", "len_axis_class_auc",
                                              "resid_norm_median")}

    # ---------- clean descriptives ----------
    def clean(its):
        g = lambda o: [it.rows[0][o] for it in its if it.rows[0][o] is not None]
        return dict(acc=float(np.mean(g("correct"))), D=float(np.mean(g("D"))), D1=float(np.mean(g("D1"))),
                    Djsd=float(np.mean(g("Djsd"))), P=float(np.mean([it.r["P"] for it in its])),
                    W=float(np.mean([it.free[0]["W"] for it in its if it.free])) if its and its[0].free else None,
                    n=len(its))
    out["clean"] = {name: clean(s) for name, s in
                    [("F", facts), ("M familiar", by(lambda r: r["sub"] == "familiar")), ("M novel", novel)] if s}

    # ---------- H1 ----------
    m_ok = [it for it in Mall if it.rows[0]["D"] is not None]
    P = np.array([it.r["P"] for it in m_ok])
    Dc = np.array([it.rows[0]["D"] for it in m_ok])
    lab = np.array([math.log(it.r["a"] * it.r["b"]) for it in m_ok])
    fam = np.array([float(it.r["sub"] == "familiar") for it in m_ok])
    if len(m_ok) > 5:
        out["H1"] = dict(rho=boot(lambda i: spearmanr(P[i], Dc[i])[0], len(m_ok)), n=len(m_ok))
        out["H1b"] = dict(partial_rho=boot(lambda i: partial_spearman(P[i], Dc[i], [lab[i], fam[i]]), len(m_ok)))
        out["H1_expl"] = dict(rho_P_vs_correct=boot(lambda i: spearmanr(P[i], [it.rows[0]["correct"] for it in np.array(m_ok)[i]])[0], len(m_ok)),
                              rho_Plen_vs_D=boot(lambda i: spearmanr(np.array([it.r["P_len"] for it in m_ok])[i], Dc[i])[0], len(m_ok)),
                              rho_P_vs_logab=boot(lambda i: spearmanr(P[i], lab[i])[0], len(m_ok)))

    # ---------- H2, H3, S and exploratory strengths ----------
    res = {}
    for name, its, sign in [("F", facts, 1), ("M novel", novel, -1), ("M all", Mall, -1)]:
        if not its:
            continue
        block = {}
        for k in cfg["k"]:
            for s in (1, -1):
                kk = s * k
                for outc in ("D", "D1", "Djsd", "correct"):
                    da, ea = effect(its, "axis", kk, outc, js, cs)
                    dl, el = effect(its, "len", kk, outc, js, cs)
                    block[f"axis {kk:+g} {outc}"] = ea
                    block[f"len {kk:+g} {outc}"] = el
                    block[f"S axis-len {kk:+g} {outc}"] = paired_diff(da, dl)
                if its[0].free is not None:
                    for outc in ("W", "AF"):
                        da, ea = effect(its, "axis", kk, outc, js, cs)
                        dl, el = effect(its, "len", kk, outc, js, cs)
                        block[f"axis {kk:+g} {outc}"] = ea
                        block[f"len {kk:+g} {outc}"] = el
                        block[f"S axis-len {kk:+g} {outc}"] = paired_diff(da, dl)
        res[name] = block
    out["effects"] = res
    out["H2"] = res.get("F", {}).get("axis +1 D")
    out["H3"] = res.get("M novel", {}).get("axis -1 D")
    out["S_H2"] = res.get("F", {}).get("S axis-len +1 D")
    out["S_H3"] = res.get("M novel", {}).get("S axis-len -1 D")
    out["H5_F"] = res.get("F", {}).get("axis +1 W")
    out["H5_M"] = res.get("M novel", {}).get("axis -1 W")

    # ---------- H4: error types under -1 on novel M ----------
    if novel:
        n_ax = novel[0].row("axis", k=-1)
        kl_ax = np.median([it.rows[n_ax][KL] for it in novel])
        med = {c: np.median([it.rows[it.row("rand", j=j, c=c)][KL] for it in novel for j in js]) for c in cs}
        c_star = min(cs, key=lambda c: abs(math.log(med[c] + 1e-9) - math.log(kl_ax + 1e-9)))

        def shares(idx):
            ax = [novel[i].rows[n_ax] for i in idx]
            rd = [novel[i].rows[novel[i].row("rand", j=j, c=c_star)] for i in idx for j in js]
            sh = lambda rows: (np.mean([r["OR"] for r in rows if not r["correct"] and r["ans"] is not None])
                               if any(not r["correct"] and r["ans"] is not None for r in rows) else np.nan)
            return sh(ax) - sh(rd)
        err = lambda rows, f: dict(n_wrong=int(sum(1 for r in rows if not r["correct"])),
                                   **{t: float(np.mean([r[t] for r in rows if not r["correct"] and r["ans"] is not None]))
                                      if any(not r["correct"] and r["ans"] is not None for r in rows) else None
                                      for t in ("OR", "OP", "DC", "other")})
        out["H4"] = dict(OR_share_diff=boot(shares, len(novel)), c_matched=c_star, kl_axis_median=float(kl_ax),
                         kl_rand_median=float(med[c_star]),
                         axis=err([it.rows[n_ax] for it in novel], None),
                         random=err([it.rows[it.row("rand", j=j, c=c_star)] for it in novel for j in js], None),
                         clean=err([it.rows[0] for it in novel], None))

    # ---------- KL coverage (for Stage 2) ----------
    cov = {}
    for name, its in [("F", facts), ("M", Mall)]:
        if not its:
            continue
        cov[name] = {
            **{f"axis {s * k:+g}": float(np.median([it.rows[it.row('axis', k=s * k)][KL] for it in its]))
               for k in cfg["k"] for s in (1, -1)},
            **{f"len {s * k:+g}": float(np.median([it.rows[it.row('len', k=s * k)][KL] for it in its]))
               for k in cfg["k"] for s in (1, -1)},
            **{f"rand c={c}": float(np.median([it.rows[it.row('rand', j=j, c=c)][KL] for it in its for j in js]))
               for c in cs}}
    out["kl_median"] = cov
    # dose-response: per condition, median KL and mean change from clean (for the page)
    dose = {}
    for name, its in [("F", facts), ("M novel", novel), ("M familiar", by(lambda r: r["sub"] == "familiar"))]:
        if not its:
            continue
        rows_out = []
        for n, c in enumerate(conds[1:], start=1):
            pt = dict(c, kl=float(np.median([it.rows[n][KL] for it in its])))
            for o in ("D", "correct", "W", "AF"):
                v = [it.delta(n, o) for it in its]
                v = [x for x in v if x is not None]
                pt[o] = float(np.mean(v)) if v else None
            rows_out.append(pt)
        # random directions: average over directions at each norm
        agg = {}
        for pt in rows_out:
            if pt["kind"] == "rand":
                agg.setdefault(pt["c"], []).append(pt)
        rows_out = [p for p in rows_out if p["kind"] != "rand"] + [
            dict(kind="rand", c=c, **{k: float(np.mean([p[k] for p in ps if p[k] is not None]))
                                      for k in ("kl", "D", "correct", "W", "AF")}) for c, ps in sorted(agg.items())]
        dose[name] = rows_out
    out["dose"] = dose
    out["parse_fail"] = {name: float(np.mean([r["ans"] is None for it in its for r in it.rows]))
                         for name, its in [("F", facts), ("M", Mall)] if its}

    # ---------- exploratory: familiar vs novel within format ----------
    from sklearn.metrics import roc_auc_score
    if Mall:
        y = np.array([it.r["sub"] == "novel" for it in Mall], int)
        Pm = np.array([it.r["P"] for it in Mall])
        PLm = np.array([it.r["P_len"] for it in Mall])
        auc = lambda p: (lambda i: roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else None)
        out["expl_familiar"] = dict(auc_cr=boot(auc(Pm), len(y)), auc_len=boot(auc(PLm), len(y)))

    # ---------- gradient (exploratory) ----------
    G = load(key, phase, "gradient_direct")
    GF = {r["id"]: r for r in load(key, phase, "gradient_free")}
    if G:
        out["gradient"] = [dict(id=r["id"], P=r["P"], D=r["rows"][0]["D"], correct=r["rows"][0]["correct"],
                                W=GF[r["id"]]["rows"][0]["W"] if r["id"] in GF else None) for r in G]
        # the axis paper's memorisation-likelihood scores for its 24 items (experiments/fig3_gradient.py)
        score = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 5, 5, 7, 7, 7, 9, 9, 9, 9, 9, 10, 10, 10, 10], float)
        Pg = np.array([g["P"] for g in G])
        nd = np.array([len(str(g["a"])) + len(str(g["b"])) for g in G], float)
        X = np.c_[np.ones(len(nd)), nd]
        res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
        out["gradient_stats"] = dict(r_P_score=float(np.corrcoef(Pg, score)[0, 1]), r_P_digits=float(np.corrcoef(Pg, nd)[0, 1]),
                                     partial_r_P_score_given_digits=float(np.corrcoef(res(Pg), res(score))[0, 1]), n=len(G))
    out["kl_measure"] = KL
    sfx = "" if KL == "kl" else "_" + KL
    json.dump(out, open(os.path.join(RES, f"{key}_{phase}_summary{sfx}.json"), "w"), indent=1)
    return out


def fmt(e):
    if not e:
        return "-"
    if isinstance(e, dict):
        e = e.get("effect") or e.get("rho") or e.get("partial_rho")
        if not e:
            return "-"
    return f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]" + (f" n={e[3]}" if len(e) > 3 else "")


if __name__ == "__main__":
    o = main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "kl")
    print(json.dumps(o["calibration"]))
    print("clean", json.dumps(o["clean"]))
    for h in ("H1", "H1b", "H2", "S_H2", "H3", "S_H3", "H5_F", "H5_M"):
        print(h, fmt(o.get(h)))
    if "H4" in o:
        print("H4", fmt(o["H4"]["OR_share_diff"]), o["H4"]["axis"], o["H4"]["random"])
    print("KL", json.dumps(o["kl_median"]))
    print("parse_fail", o["parse_fail"])

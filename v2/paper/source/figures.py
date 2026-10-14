"""Figures for v2 (sections 5-6), drawn from the preregistered test's summaries in ../cr4/res."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "cr4", "res")
OUT = os.path.join(HERE, "figures")
KEYS = ["1.7b", "q25-7b", "8b", "14b"]
LABEL = {"1.7b": "Qwen3-1.7B", "q25-7b": "Qwen2.5-7B-Instruct", "8b": "Qwen3-8B", "14b": "Qwen3-14B"}
BLUE, RED, VIOLET, GREY = "#4878A8", "#C0504D", "#7A5AA6", "#9AA3AD"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
S = {k: json.load(open(os.path.join(RES, f"{k}_main_summary.json"), encoding="utf-8")) for k in KEYS}


def eff(e):
    return e["effect"] if isinstance(e, dict) else e


def forest(ax, rows, xlabel, xlim=None):
    """rows: list of (label, [est, lo, hi], color, hollow) or (heading,)"""
    y = 0
    ticks, labels = [], []
    for r in rows:
        if len(r) == 1:
            y -= 0.4
            ax.text(-0.02, y, r[0], transform=ax.get_yaxis_transform(), ha="right", va="center", fontweight="bold", fontsize=8)
            y -= 1
            continue
        name, e, col, hollow = r
        ax.plot([e[1], e[2]], [y, y], color=col, lw=1.4)
        ax.plot([e[0]], [y], "o", ms=4.5, color=col, mfc="white" if hollow else col, mew=1.3)
        ticks.append(y)
        labels.append(name)
        y -= 1
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel(xlabel)
    if xlim:
        ax.set_xlim(*xlim)
    ax.tick_params(axis="y", length=0)


# Figure: tracking (H1)
fig, ax = plt.subplots(figsize=(3.4, 3.3))
rows = []
for k in KEYS:
    rows += [(LABEL[k],), ("CR axis (H1)", S[k]["H1"]["rho"], BLUE, False), ("length axis", S[k]["H1_expl"]["rho_Plen_vs_D"], VIOLET, True)]
forest(ax, rows, "Spearman $\\rho$ with settling depth (240 items)", (-0.5, 0.5))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_tracking.pdf"))

# Figure: settling depth under pushes (H2, H3), in layers
fig, ax = plt.subplots(figsize=(3.4, 4.0))
rows = []
for k in KEYS:
    L = S[k]["L"]
    lay = lambda e: [x * L for x in eff(e)[:3]]
    E = S[k]["effects"]
    rows += [(LABEL[k],), ("H2: facts, +1 gap", lay(S[k]["H2"]), BLUE, False), ("H3: novel, $-$1 gap", lay(S[k]["H3"]), RED, False),
             ("length axis, $-$1 gap", lay(E["M novel"]["len -1 D"]), VIOLET, True)]
forest(ax, rows, "change in settling depth vs. random pushes (layers)", (-1.2, 0.6))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_settling.pdf"))

# Figure: reply format (answers first), novel items
fig, ax = plt.subplots(figsize=(3.4, 3.6))
rows = []
for k in KEYS:
    E = S[k]["effects"]["M novel"]
    rows += [(LABEL[k],), ("axis, $-$1 gap", eff(E["axis -1 AF"]), RED, False), ("length axis, $-$1 gap", eff(E["len -1 AF"]), VIOLET, True),
             ("axis, +1 gap", eff(E["axis +1 AF"]), BLUE, False)]
forest(ax, rows, "change in answer-first replies vs. random pushes", (-0.45, 0.6))
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v2_format.pdf"))

# Figure: dose-response, Qwen2.5-7B-Instruct, novel items
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
pts = S["q25-7b"]["dose"]["M novel"]
for ax, (o, yl) in zip(axes, (("AF", "change in answer-first replies"), ("D", "change in settling depth (fraction)"))):
    rnd = sorted([p for p in pts if p["kind"] == "rand"], key=lambda p: p["kl"])
    ax.plot([p["kl"] for p in rnd], [p[o] for p in rnd], "-o", color=GREY, ms=3, lw=1.2, label="random (mean of 12)")
    for kind, col, hollow, lab in (("axis", None, False, "CR axis"), ("len", VIOLET, True, "length axis")):
        ps = [p for p in pts if p["kind"] == kind]
        for p in ps:
            c = col or (BLUE if p["k"] > 0 else RED)
            ax.plot([p["kl"]], [p[o]], "o", ms=5, color=c, mfc="white" if hollow else c, mew=1.3)
            ax.annotate(f"{p['k']:+g}", (p["kl"], p[o]), xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=c)
    ax.set_xscale("log")
    ax.axhline(0, color="0.4", lw=0.7, ls="--")
    ax.set_xlabel("KL from the unpushed model (log)")
    ax.set_ylabel(yl)
handles = [plt.Line2D([], [], color=GREY, marker="o", ms=3, label="random (mean of 12)"),
           plt.Line2D([], [], color=BLUE, marker="o", ls="", label="axis toward compute"),
           plt.Line2D([], [], color=RED, marker="o", ls="", label="axis toward retrieve"),
           plt.Line2D([], [], color=VIOLET, marker="o", mfc="white", ls="", label="length axis")]
fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=7.5)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "v2_dose_q25.pdf"))
print("figures written")

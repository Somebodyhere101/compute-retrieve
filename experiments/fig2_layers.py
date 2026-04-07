"""Figure 2: Full layer sweep -- AUC at every layer on Gemma 4 E2B.

Shows the CR axis exists at nearly every layer, not just one special depth.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from experiments.plot_style import plt
from cr_axis.detector import _find_layers, _format_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")

with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    CAL_C = [l.strip() for l in f if l.strip()][:20]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    CAL_R = [l.strip() for l in f if l.strip()][:20]

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto').eval()
layers = _find_layers(model)
n_layers = len(layers)

@torch.no_grad()
def get_h(prompt, li):
    cap = {}
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap['h'] = h[0, -1].float().cpu().numpy()
    handle = layers[li].register_forward_hook(hook)
    text = _format_prompt(prompt, tok)
    ids = tok(text, return_tensors='pt').to(next(model.parameters()).device)
    model(ids['input_ids'])
    handle.remove()
    return cap['h']

# sweep every layer
print(f"Sweeping {n_layers} layers...")
aucs = []
ds = []
for li in range(n_layers):
    hc = np.array([get_h(p, li) for p in CAL_C])
    hr = np.array([get_h(p, li) for p in CAL_R])
    ax_vec = hc.mean(0) - hr.mean(0)
    n = np.linalg.norm(ax_vec)
    if n < 1e-10:
        aucs.append(0.5); ds.append(0); continue
    ax_vec /= n
    pc = hc @ ax_vec; pr = hr @ ax_vec
    auc = roc_auc_score(np.concatenate([np.ones(len(pc)), np.zeros(len(pr))]),
                         np.concatenate([pc, pr]))
    d = (pc.mean() - pr.mean()) / max(np.sqrt((pc.std()**2 + pr.std()**2)/2), 1e-10)
    aucs.append(auc); ds.append(d)
    print(f"  L{li:2d}: AUC={auc:.3f} d={d:.2f}")
fig, ax1 = plt.subplots(figsize=(5.5, 3))
x = np.arange(n_layers)

# AUC line
ax1.plot(x, aucs, 'o-', color='k', markersize=3, linewidth=1.0,
         markeredgewidth=0.3, markeredgecolor='k', label='AUC', zorder=3)
ax1.set_xlabel('Layer')
ax1.set_ylabel('AUC')
ax1.set_ylim(0.85, 1.01)
ax1.axhline(1.0, color='0.7', linewidth=0.4, linestyle='--')

# Cohen's d bars on twin axis
ax2 = ax1.twinx()
ax2.bar(x, ds, alpha=0.22, color='#C0504D', edgecolor='none', label="Cohen's $d$")
ax2.set_ylabel("Cohen's $d$", color='#C0504D')
ax2.tick_params(axis='y', colors='#C0504D')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_linewidth(0.6)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend = [Line2D([0], [0], color='k', marker='o', markersize=3,
                 linewidth=1.0, label='AUC'),
          Patch(facecolor='#C0504D', alpha=0.22, label="Cohen's $d$")]
ax1.legend(handles=legend, loc='lower left', frameon=False, fontsize=8)

plt.tight_layout()
out = os.path.join(BASE, 'figures/fig2_layers.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"\nSaved to {out}")

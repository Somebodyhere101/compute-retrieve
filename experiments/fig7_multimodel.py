import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from experiments.plot_style import plt, COLORS
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.dirname(__file__))

mm = os.path.join(BASE, 'data/results/multimodel.json')
with open(mm) as f:
    raw = json.load(f)
models = [(name, r['calibration_auc'], r['gradient_correlation'])
          for name, r in sorted(raw.items(), key=lambda x: x[1]['n_layers'])]

names = [m[0].split('/')[-1] for m in models]
aucs = [m[1] for m in models]
rs = [m[2] for m in models]

families = []
for m in models:
    n = m[0].lower()
    if 'pythia' in n: families.append(COLORS['blue'])
    elif 'qwen' in n: families.append(COLORS['red'])
    elif 'llama' in n: families.append(COLORS['orange'])
    else: families.append(COLORS['green'])

y = np.arange(len(names))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

ax1.barh(y, aucs, color=families, edgecolor='white', linewidth=0.4, height=0.55)
ax1.set_xlim(0.95, 1.005)
ax1.set_xlabel('Calibration AUC')
ax1.set_yticks(y)
ax1.set_yticklabels(names, fontsize=8)
ax1.axvline(1.0, color='0.7', linewidth=0.4, linestyle='--')
for i, v in enumerate(aucs):
    ax1.text(v - 0.003, i, f'{v:.3f}', va='center', ha='right',
             fontsize=7, color='white', fontweight='bold')

ax2.barh(y, rs, color=families, edgecolor='white', linewidth=0.4, height=0.55)
ax2.set_xlim(0, 1)
ax2.set_xlabel('Gradient correlation ($r$)')
for i, v in enumerate(rs):
    ax2.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=7)

ax2.legend(handles=[Patch(facecolor=COLORS['blue'], label='Pythia'),
                     Patch(facecolor=COLORS['red'], label='Qwen'),
                     Patch(facecolor=COLORS['green'], label='Gemma'),
                     Patch(facecolor=COLORS['orange'], label='LLaMA')],
           loc='lower right', fontsize=8, frameon=False)

plt.subplots_adjust(wspace=0.08)
out = os.path.join(BASE, 'figures/fig7_multimodel.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved to {out}")

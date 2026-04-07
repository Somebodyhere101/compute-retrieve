import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from experiments.plot_style import plt
from matplotlib.lines import Line2D

BASE = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(BASE, 'data/results/robustness.json')) as f:
    data = json.load(f)

rob = data['gap1_robustness']
sizes = [5, 10, 15, 20, 30]
means = [rob[str(s)]['mean_cos_to_full'] for s in sizes]
stds = [rob[str(s)]['std'] for s in sizes]

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.errorbar(sizes, means, yerr=[1.96*s for s in stds], fmt='o-',
            color='k', capsize=3, capthick=0.6, markersize=3.5,
            linewidth=0.9, markeredgewidth=0.3, elinewidth=0.6)
ax.axhline(0.95, color='0.7', linewidth=0.4, linestyle='--')
ax.axhline(0.90, color='0.7', linewidth=0.4, linestyle=':')
ax.set_xlabel('Calibration prompts per class')
ax.set_ylabel('Cosine to full-data axis')
ax.set_ylim(0.6, 1.02)
ax.set_xticks(sizes)

legend = [Line2D([0], [0], color='0.7', linewidth=0.4, linestyle='--', label='cos = 0.95'),
          Line2D([0], [0], color='0.7', linewidth=0.4, linestyle=':', label='cos = 0.90')]
ax.legend(handles=legend, loc='lower right', frameon=False, fontsize=8)

out = os.path.join(BASE, 'figures/fig6_robustness.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved to {out}")

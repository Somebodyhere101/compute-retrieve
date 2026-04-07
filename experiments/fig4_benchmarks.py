# GSM8K vs MMLU projection histograms
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from experiments.plot_style import plt, COLORS

BASE = os.path.dirname(os.path.dirname(__file__))

b320 = os.path.join(BASE, 'data/results/benchmark_320.json')
with open(b320) as f:
    data = json.load(f)
tc = [d['projection'] for d in data['test_compute']]
tr = [d['projection'] for d in data['test_retrieve']]

fig, ax = plt.subplots(figsize=(4.5, 3))
bins = np.linspace(min(min(tc), min(tr)) - 2, max(max(tc), max(tr)) + 2, 25)
ax.hist(tc, bins=bins, alpha=0.55, color=COLORS['red'],
        label=f'GSM8K ($n$={len(tc)})', edgecolor='white', linewidth=0.4)
ax.hist(tr, bins=bins, alpha=0.55, color=COLORS['blue'],
        label=f'MMLU ($n$={len(tr)})', edgecolor='white', linewidth=0.4)
ax.axvline(0, color='k', linewidth=0.4, linestyle='--', alpha=0.3)
ax.set_xlabel('CR projection')
ax.set_ylabel('Count')
ax.legend(frameon=False, fontsize=8)
ax.annotate('AUC = 1.000', xy=(0.97, 0.93), xycoords='axes fraction',
            ha='right', va='top', fontsize=8)

out = os.path.join(BASE, 'figures/fig4_benchmarks.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved to {out}")

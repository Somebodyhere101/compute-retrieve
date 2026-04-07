import matplotlib.patches as patches
import os
from experiments.plot_style import plt, COLORS

BASE = os.path.dirname(os.path.dirname(__file__))

fig, ax = plt.subplots(figsize=(4.5, 3.5))

c_blue = COLORS['blue']
c_green = COLORS['green']
c_red = COLORS['red']
c_orange = COLORS['orange']

W, H = 1.8, 1.2
gap = 0.08

cells = [
    (-W - gap/2, gap/2,    W, H, c_blue,   'Accurate Recall'),
    (gap/2,      gap/2,    W, H, c_green,  'Reliable Reasoning'),
    (-W - gap/2, -H - gap/2, W, H, c_red,  'Retrieval Error'),
    (gap/2,      -H - gap/2, W, H, c_orange, 'Reasoning Error'),
]

for x, y, w, h, color, label in cells:
    rect = patches.Rectangle((x, y), w, h, linewidth=1.2,
                               edgecolor=color, facecolor=color, alpha=0.15)
    ax.add_patch(rect)
    rect2 = patches.Rectangle((x, y), w, h, linewidth=1.2,
                                edgecolor=color, facecolor='none')
    ax.add_patch(rect2)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=color)

arrow_kw = dict(arrowstyle='->', lw=1.5, color='black')
margin = 0.35
ax.annotate('', xy=(W + gap/2 + margin, 0), xytext=(-W - gap/2 - margin, 0),
            arrowprops=arrow_kw)
ax.annotate('', xy=(0, H + gap/2 + margin), xytext=(0, -H - gap/2 - margin),
            arrowprops=arrow_kw)

ax.text(W + gap/2 + margin + 0.08, 0, 'Compute', fontsize=9, va='center', ha='left')
ax.text(-W - gap/2 - margin - 0.08, 0, 'Retrieve', fontsize=9, va='center', ha='right')
ax.text(0, H + gap/2 + margin + 0.08, 'Correct', fontsize=9, ha='center', va='bottom')
ax.text(0, -H - gap/2 - margin - 0.08, 'Wrong', fontsize=9, ha='center', va='top')

pad = 0.6
ax.set_xlim(-W - gap/2 - margin - pad, W + gap/2 + margin + pad)
ax.set_ylim(-H - gap/2 - margin - pad, H + gap/2 + margin + pad)
ax.set_aspect('equal')
ax.axis('off')

out = os.path.join(BASE, 'figures/fig1_concept.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Saved to {out}")

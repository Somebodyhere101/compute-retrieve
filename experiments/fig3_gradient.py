"""fig3: CR projection vs arithmetic difficulty."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from experiments.plot_style import plt
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")

GRADIENT = [
    ("What is 1 + 1?", "2", 1), ("What is 2 + 2?", "4", 1),
    ("What is 5 + 5?", "10", 1), ("What is 7 * 8?", "56", 2),
    ("What is 9 * 9?", "81", 2), ("What is 6 * 7?", "42", 2),
    ("What is 12 * 12?", "144", 3), ("What is 15 * 15?", "225", 3),
    ("What is 11 * 11?", "121", 3), ("What is 13 * 14?", "182", 5),
    ("What is 17 * 6?", "102", 5), ("What is 19 * 8?", "152", 5),
    ("What is 23 * 17?", "391", 7), ("What is 29 * 13?", "377", 7),
    ("What is 37 * 11?", "407", 7), ("What is 47 * 38?", "1786", 9),
    ("What is 63 * 29?", "1827", 9), ("What is 87 * 43?", "3741", 9),
    ("What is 347 + 289?", "636", 9), ("What is 891 - 456?", "435", 9),
    ("What is 123 * 45?", "5535", 10), ("What is 789 + 654?", "1443", 10),
    ("What is 1847 * 3?", "5541", 10), ("What is 2468 + 1357?", "3825", 10),
]

with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    CAL_C = [l.strip() for l in f if l.strip()]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    CAL_R = [l.strip() for l in f if l.strip()]

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto')
model.eval()

det = CRDetector(model, tok)
det.calibrate(CAL_C, CAL_R)

scores, projs, correct = [], [], []
for prompt, expected, score in GRADIENT:
    proj = det.detect(prompt)
    with torch.no_grad():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors='pt').to(next(model.parameters()).device)
        out = model.generate(ids['input_ids'], max_new_tokens=20, do_sample=False)
        resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
    ok = expected in resp
    scores.append(score); projs.append(proj); correct.append(ok)
    print(f"  {'OK' if ok else 'X ':>2s} score={score:2d} proj={proj:+6.1f} | {prompt}")

s, p = np.array(scores), np.array(projs)
r = np.corrcoef(s, p)[0, 1]
print(f"\nr = {r:.3f}")
fig, ax = plt.subplots(figsize=(4.5, 3.2))

c_ok, c_bad = '#5A9E6F', '#C0504D'
colors = [c_ok if c else c_bad for c in correct]
ax.scatter(s, p, c=colors, s=20, edgecolors='k', linewidths=0.3, zorder=3)

z = np.polyfit(s, p, 1)
xs = np.linspace(0.5, 10.5, 100)
ax.plot(xs, np.polyval(z, xs), '--', color='0.65', linewidth=0.7)

ax.set_xlabel('Memorization likelihood (1 = trivial, 10 = novel)')
ax.set_ylabel('CR projection (compute $\\rightarrow$)')
ax.set_xticks([1, 2, 3, 5, 7, 9, 10])

from matplotlib.lines import Line2D
legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c_ok,
                  markeredgecolor='k', markeredgewidth=0.3, markersize=4, label='Correct'),
          Line2D([0], [0], marker='o', color='w', markerfacecolor=c_bad,
                  markeredgecolor='k', markeredgewidth=0.3, markersize=4, label='Incorrect')]
ax.legend(handles=legend, loc='upper left', frameon=False, fontsize=8)

ax.annotate(f'$r$ = {r:.3f}', xy=(0.97, 0.05), xycoords='axes fraction',
            ha='right', fontsize=9)

out_png = os.path.join(BASE, 'figures/fig3_gradient.png')
out_pdf = os.path.join(BASE, 'figures/fig3_gradient.pdf')
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.savefig(out_pdf, dpi=300, bbox_inches='tight')
print(f"Saved to {out_png}")

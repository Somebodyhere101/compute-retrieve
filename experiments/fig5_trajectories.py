"""fig5: per-token CR trajectories during generation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from experiments.plot_style import plt
from cr_axis import CRDetector
from cr_axis.detector import _find_layers, _format_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
COLORS_COMPUTE = ['#C0504D', '#E8913A', '#4878A8', '#5A9E6F']
COLORS_RETRIEVE = ['#4878A8', '#E8913A', '#5A9E6F', '#C0504D']

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")

PROMPTS = [
    ("What is 347 + 289?", "compute"),
    ("What is 23 times 17?", "compute"),
    ("All blorks are snargs. Ted is a blork. Is Ted a snarg?", "compute"),
    ("If it takes 3 painters 6 hours, how long for 9 painters?", "compute"),
    ("The capital of France is", "retrieve"),
    ("Shakespeare wrote Romeo and", "retrieve"),
    ("The speed of light is approximately", "retrieve"),
    ("To be or not to be, that is the", "retrieve"),
]

# load + calibrate
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
layers = _find_layers(model)
@torch.no_grad()
def generate_trajectory(prompt, max_tokens=20):
    text = _format_prompt(prompt, tok)
    ids = tok(text, return_tensors='pt').to(next(model.parameters()).device)['input_ids']
    trajectory = []
    captured = {}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured['h'] = h[0, -1].float().cpu().numpy()

    handle = layers[det.layer].register_forward_hook(hook)

    for step in range(max_tokens):
        captured.clear()
        out = model(ids)
        next_id = out.logits[0, -1].argmax().item()
        token_str = tok.decode([next_id])

        proj = float(captured['h'] @ det.axis) if 'h' in captured else 0
        trajectory.append({"token": token_str, "proj": proj, "step": step})

        if next_id == tok.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=ids.device)], dim=1)

    handle.remove()
    return trajectory
# run all trajectories
print("Generating trajectories...")
all_traj = []
for prompt, mode in PROMPTS:
    traj = generate_trajectory(prompt, 20)
    all_traj.append({"prompt": prompt, "mode": mode, "trajectory": traj})
    tokens = [t['token'] for t in traj[:8]]
    projs_t = [t['proj'] for t in traj[:8]]
    print(f"  [{mode:>7s}] {prompt[:40]:40s} -> {' '.join(tokens[:5])}")
fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=False)

for i, mode in enumerate(['compute', 'retrieve']):
    ax = axes[i]
    palette = COLORS_COMPUTE if mode == 'compute' else COLORS_RETRIEVE
    subset = [t for t in all_traj if t['mode'] == mode]
    for j, entry in enumerate(subset):
        traj_short = entry['trajectory'][:10]
        projs_t = [t['proj'] for t in traj_short]
        tokens_t = [t['token'].strip() for t in traj_short]
        lbl = entry['prompt'][:30] + '...' if len(entry['prompt']) > 30 else entry['prompt']
        ax.plot(projs_t, marker='o', markersize=3, alpha=0.8, linewidth=0.9,
                color=palette[j % len(palette)], markeredgecolor='k',
                markeredgewidth=0.3, label=lbl)
        # label first token
        if projs_t:
            ax.annotate(tokens_t[0], (0, projs_t[0]), fontsize=6, alpha=0.5,
                       textcoords="offset points", xytext=(5, 5))

    ax.axhline(0, color='black', linewidth=0.4, linestyle='-')
    ax.set_ylabel('CR projection')
    ax.annotate(f'{mode.capitalize()} tasks', xy=(0.02, 0.95),
                xycoords='axes fraction', fontsize=9, fontweight='bold',
                va='top', ha='left')
    ax.legend(fontsize=7, loc='lower right' if mode == 'compute' else 'upper right',
              frameon=False)

axes[1].set_xlabel('Generation step')
plt.tight_layout()

out = os.path.join(BASE, 'figures/fig5_trajectories.pdf')
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f"Saved to {out}")

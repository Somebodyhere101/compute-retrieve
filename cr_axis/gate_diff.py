"""Gate map diffing for GELU-gated MLPs.

Compares binary gate patterns (sign of gate_proj output) between prompts.
Only works on gated architectures (LLaMA, Qwen, Gemma, Mistral).
Returns empty dict on non-gated models (Pythia, GPT-2).
"""

import torch
import numpy as np
from .detector import _format_prompt


@torch.no_grad()
def extract_gates(model, tokenizer, prompt, layers, device=None):
    if device is None:
        device = next(model.parameters()).device

    gates = {}
    hooks = []

    def make_hook(li):
        def fn(mod, inp, out):
            gates[li] = (out[0, -1] > 0).cpu().numpy()
        return fn

    for li, layer in enumerate(layers):
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_hook(li)))

    if not hooks:
        return {}

    text = _format_prompt(prompt, tokenizer)
    ids = tokenizer(text, return_tensors='pt').to(device)
    try:
        model(ids['input_ids'])
    finally:
        for h in hooks:
            h.remove()
    return gates


def gate_diff(gates_a, gates_b):
    results = {}
    for li in gates_a:
        if li not in gates_b:
            continue
        flips = gates_a[li] != gates_b[li]
        results[li] = {
            "n_flips": int(flips.sum()),
            "pct": flips.sum() / len(gates_a[li]),
            "flip_indices": np.where(flips)[0],
            "on_to_off": int(((gates_a[li] == 1) & (gates_b[li] == 0)).sum()),
            "off_to_on": int(((gates_a[li] == 0) & (gates_b[li] == 1)).sum()),
        }
    return results


def jaccard(diff_a, diff_b):
    set_a = {(li, int(idx)) for li, d in diff_a.items() for idx in d["flip_indices"]}
    set_b = {(li, int(idx)) for li, d in diff_b.items() for idx in d["flip_indices"]}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

"""Table 1: CR axis across 12 models, 5 architecture families."""
import sys, os, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    COMPUTE = [l.strip() for l in f if l.strip()]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    RETRIEVE = [l.strip() for l in f if l.strip()]

GRADIENT = [
    ("What is 1 + 1?", "2", 1), ("What is 2 + 2?", "4", 1),
    ("What is 5 + 5?", "10", 1), ("What is 7 * 8?", "56", 2),
    ("What is 9 * 9?", "81", 2), ("What is 6 * 7?", "42", 2),
    ("What is 12 * 12?", "144", 3), ("What is 11 * 11?", "121", 3),
    ("What is 13 * 14?", "182", 5), ("What is 17 * 6?", "102", 5),
    ("What is 23 * 17?", "391", 7), ("What is 29 * 13?", "377", 7),
    ("What is 47 * 38?", "1786", 9), ("What is 87 * 43?", "3741", 9),
    ("What is 347 + 289?", "636", 9), ("What is 891 - 456?", "435", 9),
    ("What is 123 * 45?", "5535", 10), ("What is 789 + 654?", "1443", 10),
]

MODELS = [
    ("EleutherAI/pythia-70m", False),
    ("EleutherAI/pythia-160m", False),
    ("EleutherAI/pythia-410m", False),
    ("EleutherAI/pythia-1b", False),
    ("Qwen/Qwen2.5-0.5B-Instruct", False),
    ("Qwen/Qwen2.5-1.5B-Instruct", False),
    ("Qwen/Qwen2.5-3B", False),
    ("Qwen/Qwen2.5-7B-Instruct", True),
    ("google/gemma-4-e2b-it", False),
    ("google/gemma-4-E4B", False),
    ("Qwen/Qwen3-30B-A3B", True),
    ("unsloth/Llama-3.2-1B-Instruct", False),
]


def run_model(name, use_4bit):
    print(f"\n  {name}")
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=bnb, device_map={'': 0}, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, device_map='auto')
    model.eval()

    det = CRDetector(model, tok)
    auc, layer = det.calibrate(COMPUTE, RETRIEVE)

    grad = []
    for prompt, expected, score in GRADIENT:
        grad.append({"score": score, "proj": det.detect(prompt)})

    scores = np.array([g['score'] for g in grad])
    projs = np.array([g['proj'] for g in grad])
    r = np.corrcoef(scores, projs)[0, 1] if len(set(scores)) > 1 else 0
    low = projs[scores <= 3].mean()
    mid = projs[(scores > 3) & (scores <= 7)].mean()
    high = projs[scores > 7].mean()

    print(f"    L{layer} AUC={auc:.3f} r={r:.3f} mono={'Y' if low<mid<high else 'N'}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return {"model": name, "layer": layer, "auc": float(auc),
            "gradient_r": float(r), "monotonic": bool(low < mid < high),
            "gradient_details": grad}


if __name__ == '__main__':
    results = [run_model(n, q) for n, q in MODELS]

    print(f"\n{'Model':<40s} {'L':>3s} {'AUC':>6s} {'r':>6s} {'Mono':>5s}")
    print("-" * 62)
    for r in results:
        print(f"{r['model']:<40s} L{r['layer']:>2d} {r['auc']:6.3f} "
              f"{r['gradient_r']:6.3f} {'Y' if r['monotonic'] else 'N':>5s}")

    out = os.path.join(BASE, 'data/results/table1.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

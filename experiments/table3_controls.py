"""Table 3: Control experiments - format, length, robustness.
~5 min on Gemma 4 E2B.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")

with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    CAL_C = [l.strip() for l in f if l.strip()]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    CAL_R = [l.strip() for l in f if l.strip()]

print(f"Loading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto').eval()

det = CRDetector(model, tok)
det.calibrate(CAL_C, CAL_R)
print(f"Calibrated: L{det.layer}\n")

results = {}

# 1. format control
FORMAT_PAIRS = [
    # same "What is X?" format, one arithmetic one factual
    ("What is 23 times 17?", "What is the capital of France?"),
    ("What is 48 plus 76?", "What is the color of the sky?"),
    ("What is 99 minus 37?", "What is the symbol for gold?"),
    ("What is 144 divided by 12?", "What is the formula for water?"),
    ("What is 256 plus 389?", "What is the largest ocean?"),
    ("What is 15 times 17?", "What is the currency of Japan?"),
    ("What is 999 minus 572?", "What is the boiling point of water?"),
]

print("1. Format control")
n_sep = sum(1 for pc, pr in FORMAT_PAIRS if det.detect(pc) > det.detect(pr))
print(f"   {n_sep}/{len(FORMAT_PAIRS)} separated")
results['format_control'] = {"separated": n_sep, "total": len(FORMAT_PAIRS)}

# 2. within-arithmetic
TRIVIAL = ["What is 1 + 1?", "What is 2 + 2?", "What is 5 + 5?",
           "What is 2 * 2?", "What is 3 * 3?", "What is 10 + 10?",
           "What is 100 - 50?", "What is 10 * 10?", "What is 6 + 6?",
           "What is 1 + 0?", "What is 0 + 0?", "What is 5 * 2?",
           "What is 10 - 1?", "What is 3 + 3?", "What is 4 * 4?"]
HARD = ["What is 347 + 289?", "What is 23 * 19?", "What is 891 - 456?",
        "What is 17 * 23?", "What is 256 + 389?", "What is 15 * 17?",
        "What is 999 - 572?", "What is 48 * 13?", "What is 167 + 894?",
        "What is 73 * 8?", "What is 512 - 237?", "What is 29 * 31?",
        "What is 444 + 777?", "What is 63 * 7?", "What is 1000 - 618?"]

tp = np.array(det.detect_batch(TRIVIAL))
hp = np.array(det.detect_batch(HARD))
wa_auc = roc_auc_score(np.concatenate([np.zeros(len(tp)), np.ones(len(hp))]),
                        np.concatenate([tp, hp]))
wa_d = (hp.mean() - tp.mean()) / np.sqrt((hp.std()**2 + tp.std()**2) / 2)
print(f"\n2. Within-arithmetic: AUC={wa_auc:.2f} d={wa_d:.2f}")
results['within_arithmetic'] = {"auc": float(wa_auc), "d": float(wa_d)}

# 3. reversed length (short compute, long retrieve)
REVERSED = [
    ("47*38=", "According to established historical records, the capital city of the country of France is widely known to be"),
    ("891-456=", "In the well-documented field of chemistry, the molecular formula used to represent water is commonly written as"),
    ("23*19=", "As any student of physics would know, the approximate speed of light traveling through a vacuum is"),
    ("73*8=", "The famous English playwright William Shakespeare is well known for having written the tragedy of Romeo and"),
    ("347+289=", "In basic geography that most people learn in school, the single largest ocean on the surface of planet Earth is the"),
    ("15*17=", "Looking at the periodic table of elements, one can see that the standard chemical symbol used for the element gold is"),
    ("999-572=", "It is a well established fact that the very first president of the United States of America was a man named George"),
    ("48*13=", "According to basic science that everyone learns, the standard boiling point of pure water at normal sea level atmospheric pressure is"),
]

n_rev = sum(1 for pc, pr in REVERSED if det.detect(pc) > det.detect(pr))
print(f"\n3. Reversed length: {n_rev}/{len(REVERSED)} separated")
results['reversed_length'] = {"separated": n_rev, "total": len(REVERSED)}

# 4. calibration robustness
np.random.seed(42)
hc_all = np.array([det._get_hidden(p, det.layer) for p in CAL_C])
hr_all = np.array([det._get_hidden(p, det.layer) for p in CAL_R])
full_axis = hc_all.mean(0) - hr_all.mean(0)
full_axis /= np.linalg.norm(full_axis)

print(f"\n4. Calibration robustness")
robustness = {}
for size in [5, 10, 15, 20, 30]:
    cosines = []
    for _ in range(20):
        ic = np.random.choice(len(CAL_C), size=min(size, len(CAL_C)), replace=False)
        ir = np.random.choice(len(CAL_R), size=min(size, len(CAL_R)), replace=False)
        ax = hc_all[ic].mean(0) - hr_all[ir].mean(0)
        ax /= np.linalg.norm(ax) + 1e-10
        cosines.append(float(np.dot(ax, full_axis)))
    m, s = np.mean(cosines), np.std(cosines)
    print(f"   n={size:2d}: cos={m:.4f} +/- {s:.4f}")
    robustness[size] = {"mean": m, "std": s}
results['robustness'] = robustness

out = os.path.join(BASE, 'data/results/table3.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}")

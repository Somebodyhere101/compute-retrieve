# benchmark CR test
import sys, os, json, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

random.seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_gsm8k():
    local = os.environ.get("GSM8K_PATH")
    if local and os.path.exists(local):
        return [d['q'] for d in load_jsonl(local)]
    from datasets import load_dataset
    return [d['question'] for d in load_dataset("gsm8k", "main", split="train")]


def load_mmlu():
    local = os.environ.get("MMLU_PATH")
    if local and os.path.exists(local):
        with open(local) as f:
            raw = json.load(f)
        return [q['q'] for s in raw.values() for q in s], raw
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    raw = {}
    for d in ds:
        raw.setdefault(d['subject'], []).append({"q": d['question']})
    return [q['q'] for s in raw.values() for q in s], raw


def load_obqa():
    local = os.environ.get("OBQA_PATH")
    if local and os.path.exists(local):
        return [d['q'] for d in load_jsonl(local)]
    from datasets import load_dataset
    return [d['question_stem'] for d in load_dataset("allenai/openbookqa", "main", split="train")]


def load_aqua():
    local = os.environ.get("AQUA_PATH")
    if local and os.path.exists(local):
        return [d['q'] for d in load_jsonl(local)]
    from datasets import load_dataset
    return [d['question'] for d in load_dataset("deepmind/aqua_rat", split="train")]


print("Loading benchmarks...")
gsm8k = load_gsm8k()
mmlu, mmlu_raw = load_mmlu()
obqa = load_obqa()
aqua = load_aqua()
print(f"  GSM8K={len(gsm8k)} MMLU={len(mmlu)} OBQA={len(obqa)} AQUA={len(aqua)}")

cal_c = random.sample(gsm8k, 30)
cal_r = random.sample(mmlu, 30)
test_c = random.sample([q for q in gsm8k if q not in cal_c], 50)
test_r = random.sample([q for q in mmlu if q not in cal_r], 50)

print(f"\nLoading {MODEL}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto').eval()

det = CRDetector(model, tok)
auc_cal, layer = det.calibrate(cal_c, cal_r)
print(f"Calibrated: L{layer} AUC={auc_cal:.3f}")

# held-out
print("\nHeld-out (50+50)...")
details = []
for q in test_c:
    details.append({"prompt": q[:200], "proj": det.detect(q), "label": "compute"})
for q in test_r:
    details.append({"prompt": q[:200], "proj": det.detect(q), "label": "retrieve"})

tc = np.array([d['proj'] for d in details if d['label'] == 'compute'])
tr = np.array([d['proj'] for d in details if d['label'] == 'retrieve'])
auc_ho = roc_auc_score(np.concatenate([np.ones(len(tc)), np.zeros(len(tr))]),
                        np.concatenate([tc, tr]))
d_ho = (tc.mean() - tr.mean()) / np.sqrt((tc.std()**2 + tr.std()**2) / 2)
print(f"  AUC={auc_ho:.3f} d={d_ho:.2f}")

# cross-benchmark
print("\nCross-benchmark (AQUA vs OBQA)...")
xc = np.array([det.detect(q) for q in random.sample(aqua, 30)])
xr = np.array([det.detect(q) for q in random.sample(obqa, 30)])
auc_x = roc_auc_score(np.concatenate([np.ones(len(xc)), np.zeros(len(xr))]),
                       np.concatenate([xc, xr]))
d_x = (xc.mean() - xr.mean()) / np.sqrt((xc.std()**2 + xr.std()**2) / 2)
print(f"  AUC={auc_x:.3f} d={d_x:.2f}")

# MMLU reasoning vs factual
print("\nMMLU split...")
reason_subj = ['high_school_physics', 'high_school_mathematics',
               'abstract_algebra', 'machine_learning', 'college_physics']
fact_subj = ['world_religions', 'high_school_geography', 'anatomy',
             'college_biology', 'high_school_world_history']

mr = [det.detect(q['q']) for s in reason_subj if s in mmlu_raw for q in mmlu_raw[s][:10]]
mf = [det.detect(q['q']) for s in fact_subj if s in mmlu_raw for q in mmlu_raw[s][:10]]
mr, mf = np.array(mr), np.array(mf)
auc_m = roc_auc_score(np.concatenate([np.ones(len(mr)), np.zeros(len(mf))]),
                       np.concatenate([mr, mf]))
d_m = (mr.mean() - mf.mean()) / np.sqrt((mr.std()**2 + mf.std()**2) / 2)
print(f"  AUC={auc_m:.3f} d={d_m:.2f}")

print(f"\n{'Test':<40s} {'N':>4s} {'AUC':>6s} {'d':>6s}")
print("-" * 58)
print(f"{'Calibration (GSM8K + MMLU)':<40s} {60:>4d} {auc_cal:6.3f}     -")
print(f"{'Held-out (GSM8K + MMLU)':<40s} {100:>4d} {auc_ho:6.3f} {d_ho:6.2f}")
print(f"{'Cross-benchmark (AQUA + OBQA)':<40s} {60:>4d} {auc_x:6.3f} {d_x:6.2f}")
print(f"{'MMLU reasoning vs factual':<40s} {len(mr)+len(mf):>4d} {auc_m:6.3f} {d_m:6.2f}")

out = os.path.join(BASE, 'data/results/table2.json')
with open(out, 'w') as f:
    json.dump({"calibration": {"auc": auc_cal, "layer": layer},
               "held_out": {"auc": float(auc_ho), "d": float(d_ho)},
               "cross_benchmark": {"auc": float(auc_x), "d": float(d_x)},
               "mmlu_split": {"auc": float(auc_m), "d": float(d_m)},
               "per_prompt": details}, f, indent=2)
print(f"\nSaved to {out}")

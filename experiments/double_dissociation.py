# double dissociation + subspace analysis + multi-step/single-step control
# run on Qwen 7B (gets enough MMLU right to train a truthfulness probe)
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from cr_axis.detector import _format_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
SAVE = os.path.join(BASE, 'data/results/double_dissociation.json')
MMLU_PATH = os.environ.get("MMLU_PATH")

# multi-hop retrieval (looks like it needs reasoning but it's all recall)
MULTIHOP_RETRIEVAL = [
    "What country was the inventor of the telephone born in?",
    "What language is spoken in the country where the Eiffel Tower is?",
    "What continent is the largest ocean's deepest point in?",
    "What century was the author of Hamlet born in?",
    "What element has the same atomic number as the number of planets in our solar system?",
    "What is the capital of the country that hosted the 2020 Olympics?",
    "Who painted the ceiling of the building where the Pope lives?",
    "What ocean borders the country where sushi originated?",
    "What is the national language of the country where the Taj Mahal is?",
    "What is the currency of the country that invented pizza?",
]

# single-step computation (trivially simple but still requires computing)
SINGLESTEP_COMPUTE = [
    "Is 15 greater than 12?",
    "What is 2 times 3?",
    "What is 10 minus 4?",
    "Is 7 an odd number?",
    "What is half of 20?",
    "What is 3 + 4?",
    "Is 100 greater than 99?",
    "What is double 8?",
    "What is 9 minus 5?",
    "Is 6 an even number?",
]

print("Loading Qwen 7B (4-bit)...", flush=True)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct',
    quantization_config=bnb, device_map={'': 0}, low_cpu_mem_usage=True).eval()

with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    cal_c = [l.strip() for l in f if l.strip()]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    cal_r = [l.strip() for l in f if l.strip()]

det = CRDetector(model, tok)
auc_cal, layer = det.calibrate(cal_c, cal_r)
print(f"Calibrated: L{layer} AUC={auc_cal:.3f}\n", flush=True)

results = {}

print("--- Double dissociation ---", flush=True)

with open(MMLU_PATH) as f:
    mmlu = json.load(f)

REASONING = ['high_school_physics', 'high_school_mathematics', 'college_physics',
             'abstract_algebra', 'machine_learning', 'college_chemistry',
             'high_school_chemistry']
FACTUAL = ['world_religions', 'high_school_geography', 'anatomy',
           'college_biology', 'high_school_world_history', 'computer_security',
           'prehistory']

entries = []
for stype, subjects in [('reasoning', REASONING), ('factual', FACTUAL)]:
    for subj in subjects:
        if subj not in mmlu:
            continue
        for q in mmlu[subj][:15]:
            h = det._get_hidden(q['q'], layer)
            cr_proj = float(h @ det.axis)

            text = _format_prompt(q['q'] + "\nAnswer:", tok)
            ids = tok(text, return_tensors='pt').to('cuda')
            with torch.no_grad():
                out = model.generate(ids['input_ids'], max_new_tokens=10, do_sample=False)
            resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            correct_text = q['c'][q['a']] if isinstance(q['a'], int) and q['a'] < len(q['c']) else str(q['a'])
            ok = correct_text.lower()[:20] in resp.lower()[:60]

            entries.append({
                'h': h, 'correct': ok, 'type': stype,
                'cr_proj': cr_proj, 'subject': subj,
            })
    print(f"  {stype}: {sum(1 for e in entries if e['type'] == stype)} questions", flush=True)

n_correct = sum(1 for e in entries if e['correct'])
n_wrong = sum(1 for e in entries if not e['correct'])
print(f"  Total: {len(entries)} ({n_correct} correct, {n_wrong} wrong)", flush=True)

# TEST A: can CR axis predict correctness?
if n_correct >= 5 and n_wrong >= 5:
    cr_scores = np.array([e['cr_proj'] for e in entries])
    correctness_labels = np.array([1 if e['correct'] else 0 for e in entries])
    cr_correctness_auc = roc_auc_score(correctness_labels, cr_scores)
    # AUC could be inverted, take closest to 0.5
    cr_correctness_auc_centered = min(cr_correctness_auc, 1 - cr_correctness_auc)
    print(f"\n  TEST A: CR axis predicts correctness?")
    print(f"    AUC = {cr_correctness_auc:.3f} (distance from chance: {abs(cr_correctness_auc - 0.5):.3f})")
    print(f"    {'PASS: near chance' if abs(cr_correctness_auc - 0.5) < 0.15 else 'FAIL: predicts correctness'}")

# TEST B: can truthfulness probe separate failure types?
correct_h = np.array([e['h'] for e in entries if e['correct']])
incorrect_h = np.array([e['h'] for e in entries if not e['correct']])

if len(correct_h) >= 5 and len(incorrect_h) >= 5:
    truth_axis = correct_h.mean(0) - incorrect_h.mean(0)
    truth_axis /= np.linalg.norm(truth_axis) + 1e-10

    cos_truth_cr = float(np.dot(truth_axis, det.axis))
    print(f"\n  Cosine(truth axis, CR axis) = {cos_truth_cr:.3f}")

    # can truth axis classify failure types?
    wrong_reasoning = [e for e in entries if not e['correct'] and e['type'] == 'reasoning']
    wrong_factual = [e for e in entries if not e['correct'] and e['type'] == 'factual']

    if len(wrong_reasoning) >= 5 and len(wrong_factual) >= 5:
        wr_truth = np.array([e['h'] @ truth_axis for e in wrong_reasoning])
        wf_truth = np.array([e['h'] @ truth_axis for e in wrong_factual])
        truth_failure_auc_raw = roc_auc_score(
            np.concatenate([np.ones(len(wr_truth)), np.zeros(len(wf_truth))]),
            np.concatenate([wr_truth, wf_truth]))
        truth_failure_auc = max(truth_failure_auc_raw, 1 - truth_failure_auc_raw)

        wr_cr = np.array([e['h'] @ det.axis for e in wrong_reasoning])
        wf_cr = np.array([e['h'] @ det.axis for e in wrong_factual])
        cr_failure_auc = roc_auc_score(
            np.concatenate([np.ones(len(wr_cr)), np.zeros(len(wf_cr))]),
            np.concatenate([wr_cr, wf_cr]))

        print(f"\n  TEST B: Truth probe separates failure types?")
        print(f"    Truth axis AUC on failures = {truth_failure_auc:.3f}")
        print(f"    CR axis AUC on failures    = {cr_failure_auc:.3f}")
        print(f"    {'PASS: truth cant separate failures' if truth_failure_auc < 0.65 else 'FAIL: truth separates failures'}")

        # also: can CR axis predict correctness among reasoning-only and factual-only?
        for stype in ['reasoning', 'factual']:
            subset = [e for e in entries if e['type'] == stype]
            if sum(e['correct'] for e in subset) >= 3 and sum(not e['correct'] for e in subset) >= 3:
                s_scores = np.array([e['cr_proj'] for e in subset])
                s_labels = np.array([1 if e['correct'] else 0 for e in subset])
                s_auc = roc_auc_score(s_labels, s_scores)
                print(f"    CR→correctness within {stype}: AUC={s_auc:.3f}")

        results['double_dissociation'] = {
            'cr_correctness_auc': float(cr_correctness_auc),
            'cos_truth_cr': float(cos_truth_cr),
            'truth_failure_auc': float(truth_failure_auc),
            'cr_failure_auc': float(cr_failure_auc),
            'n_correct': n_correct,
            'n_wrong': n_wrong,
            'n_wrong_reasoning': len(wrong_reasoning),
            'n_wrong_factual': len(wrong_factual),
        }

print(f"\n--- Cross-domain subspace ---", flush=True)

# build per-domain axes
domain_prompts = {
    'arithmetic': ["What is 23+45?", "What is 7*8?", "What is 99-37?", "What is 15*13?",
                    "What is 256+389?", "What is 81-29?", "What is 17*6?", "What is 200/8?",
                    "What is 33+78?", "What is 9*11?"],
    'logic': ["All cats purr. Tom is a cat. Therefore Tom",
              "All birds fly. Robin is a bird. Therefore Robin",
              "No fish walk. Sam is a fish. Therefore Sam",
              "If it rains the ground gets wet. It rains. Therefore",
              "If P then Q. P is true. Therefore Q is",
              "All metals conduct. Iron is metal. Therefore iron",
              "No dogs fly. Rex is a dog. Therefore Rex",
              "All squares have four sides. This is a square. Therefore",
              "If you study you pass. You studied. Therefore",
              "All mammals breathe. Whale is mammal. Therefore whale"],
    'causal': ["It rained all day. The ground is probably",
               "Glass fell. It probably", "He studied hard. He probably",
               "Ice left in sun. It probably", "Forgot umbrella. She probably got",
               "Car ran out of gas. It probably", "Ate too much. He probably felt",
               "Alarm rang. Everyone probably", "Road was icy. Car probably",
               "Fire went out. It probably ran out of"],
    'sequence': ["2, 4, 6, 8,", "1, 3, 5, 7,", "10, 20, 30, 40,",
                 "3, 6, 9, 12,", "1, 4, 9, 16,", "5, 10, 15, 20,",
                 "100, 200, 300,", "2, 4, 8, 16,", "1, 1, 2, 3, 5,",
                 "0, 1, 1, 2, 3, 5,"],
}

retrieve_prompts = cal_r[:10]

domain_axes = {}
for domain, prompts in domain_prompts.items():
    hc = np.array([det._get_hidden(p, layer) for p in prompts])
    hr = np.array([det._get_hidden(p, layer) for p in retrieve_prompts])
    ax = hc.mean(0) - hr.mean(0)
    ax /= np.linalg.norm(ax) + 1e-10
    domain_axes[domain] = ax

# pairwise cosines
domains = list(domain_axes.keys())
print(f"\n  Pairwise cosines:")
cos_matrix = {}
for i, d1 in enumerate(domains):
    for d2 in domains[i+1:]:
        c = float(np.dot(domain_axes[d1], domain_axes[d2]))
        cos_matrix[f"{d1}_vs_{d2}"] = c
        print(f"    {d1:12s} vs {d2:12s}: cos = {c:.3f}")

# stack axes, do PCA to find shared subspace
axis_matrix = np.array([domain_axes[d] for d in domains])  # [4, HD]
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(axis_matrix)
explained = np.cumsum(pca.explained_variance_ratio_)
n_dims_90 = int(np.searchsorted(explained, 0.90)) + 1
n_dims_95 = int(np.searchsorted(explained, 0.95)) + 1
print(f"\n  Subspace analysis (4 domain axes):")
print(f"    Variance explained by 1 dim: {pca.explained_variance_ratio_[0]:.1%}")
print(f"    Variance explained by 2 dim: {explained[1]:.1%}")
print(f"    Dims for 90% variance: {n_dims_90}")
print(f"    Dims for 95% variance: {n_dims_95}")

results['subspace'] = {
    'pairwise_cosines': cos_matrix,
    'variance_explained': [float(v) for v in pca.explained_variance_ratio_],
    'dims_90pct': n_dims_90,
    'dims_95pct': n_dims_95,
}

print(f"\n--- Multi-hop retrieval vs single-step computation ---", flush=True)

mh_projs = [det.detect(p) for p in MULTIHOP_RETRIEVAL]
ss_projs = [det.detect(p) for p in SINGLESTEP_COMPUTE]
mh, ss = np.array(mh_projs), np.array(ss_projs)

for p, proj in zip(MULTIHOP_RETRIEVAL, mh_projs):
    print(f"  ret  {proj:+7.1f} | {p[:55]}", flush=True)
for p, proj in zip(SINGLESTEP_COMPUTE, ss_projs):
    print(f"  comp {proj:+7.1f} | {p[:55]}", flush=True)

if len(mh) >= 3 and len(ss) >= 3:
    labels_ms = np.concatenate([np.zeros(len(mh)), np.ones(len(ss))])
    scores_ms = np.concatenate([mh, ss])
    ms_auc = roc_auc_score(labels_ms, scores_ms)
    ms_d = (ss.mean() - mh.mean()) / max(np.sqrt((ss.std()**2 + mh.std()**2) / 2), 1e-10)
    print(f"\n  Multi-hop retrieval: {mh.mean():+.1f} +/- {mh.std():.1f}")
    print(f"  Single-step compute: {ss.mean():+.1f} +/- {ss.std():.1f}")
    print(f"  AUC: {ms_auc:.3f}, d: {ms_d:.2f}")

    results['multistep_control'] = {
        'multihop_retrieval_mean': float(mh.mean()),
        'singlestep_compute_mean': float(ss.mean()),
        'auc': float(ms_auc),
        'd': float(ms_d),
    }

with open(SAVE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n--- Summary ---")
if 'double_dissociation' in results:
    dd = results['double_dissociation']
    print(f"  CR→correctness AUC: {dd['cr_correctness_auc']:.3f} ({'orthogonal' if abs(dd['cr_correctness_auc']-0.5) < 0.15 else 'NOT orthogonal'})")
    print(f"  Truth→failure-type AUC: {dd['truth_failure_auc']:.3f} ({'orthogonal' if dd['truth_failure_auc'] < 0.65 else 'NOT orthogonal'})")
    print(f"  CR→failure-type AUC: {dd['cr_failure_auc']:.3f}")
    print(f"  cos(truth, CR): {dd['cos_truth_cr']:.3f}")
if 'subspace' in results:
    print(f"  Subspace: {results['subspace']['dims_90pct']} dims for 90%")
if 'multistep_control' in results:
    mc = results['multistep_control']
    print(f"  Multi-hop ret vs single-step comp: AUC={mc['auc']:.3f}")
print(f"\nSaved to {SAVE}")

del model; gc.collect(); torch.cuda.empty_cache()

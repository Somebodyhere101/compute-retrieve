# adversarial prompt test + baseline probe comparison
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from cr_axis.detector import _format_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

BASE = os.path.dirname(os.path.dirname(__file__))
SAVE = os.path.join(BASE, 'data/results/acceptance_experiments.json')

MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")

# math-looking but pure retrieval
MATH_LOOKING_RETRIEVAL = [
    "What is the value of pi to 5 decimal places?",
    "What is Avogadro's number?",
    "What is the speed of light in meters per second?",
    "What is Euler's number e to 4 decimal places?",
    "What is the atomic number of gold?",
    "What is the gravitational constant G?",
    "What is the mass of an electron in kilograms?",
    "What is the square root of 2 to 3 decimal places?",
    "What is Planck's constant?",
    "What is the charge of a proton in coulombs?",
    "What is the boiling point of nitrogen in Celsius?",
    "What is the radius of Earth in kilometers?",
    "How many chromosomes do humans have?",
    "What is the atomic mass of carbon?",
    "What is the melting point of iron in Celsius?",
    "What is the distance from Earth to the Sun in kilometers?",
    "What is the half-life of carbon-14 in years?",
    "How many bones are in the adult human body?",
    "What is the pH of pure water?",
    "What is the density of water in kg per cubic meter?",
    "What is the molecular weight of glucose?",
    "What is the escape velocity of Earth in km/s?",
    "How many teeth does an adult human have?",
    "What is the freezing point of mercury in Celsius?",
    "What is the wavelength of red light in nanometers?",
]

# trivia-looking but requires computation
TRIVIA_LOOKING_COMPUTATION = [
    "If a recipe serves 4 and I need to serve 7, how many cups of flour do I need if the recipe calls for 3 cups?",
    "How many seconds are in 3 hours and 45 minutes?",
    "If I drive 65 mph for 2 hours and 30 minutes, how far do I go?",
    "A shirt is 40% off and originally costs $85. What do I pay?",
    "If 3 painters can paint a house in 8 hours, how long for 6 painters?",
    "I have $250 and spend 15% on lunch. How much is left?",
    "A pool fills at 3 gallons per minute. How long to fill 450 gallons?",
    "If I read 35 pages per hour, how long to finish a 280-page book?",
    "A train leaves at 9:15 AM and arrives at 1:40 PM. How long is the trip?",
    "If apples cost $1.20 each and I buy 8, how much change from $20?",
    "My rent is $1,450 per month. What is my yearly rent?",
    "A 15% tip on a $78 dinner bill is how much?",
    "If I save $125 per week, how many weeks to save $3,000?",
    "A rectangular room is 12 feet by 15 feet. What is the area?",
    "I bought 3 items at $12.50, $8.75, and $15.99. What is the total?",
    "A car gets 32 miles per gallon. How many gallons for 480 miles?",
    "If 1 inch is 2.54 cm, how many cm in 5 feet 8 inches?",
    "A flight from NYC to LA takes 5.5 hours. If I leave at 10 AM EST, what time do I arrive PST?",
    "I need to split $340 evenly among 8 people. How much each?",
    "A 20% raise on a $52,000 salary is how much total?",
    "If a ball is thrown at 20 m/s straight up, how high does it go?",
    "How many tiles do I need for a 10x12 foot floor if each tile is 1 sq ft?",
    "A bacteria population doubles every 30 minutes. Starting from 100, how many after 3 hours?",
    "If I mix 2 liters of 80% solution with 3 liters of 40% solution, what percentage is the result?",
    "A ladder leans against a wall. The base is 6 feet from the wall and the ladder is 10 feet long. How high up the wall?",
]

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map='auto').eval()

with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
    cal_c = [l.strip() for l in f if l.strip()]
with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
    cal_r = [l.strip() for l in f if l.strip()]

det = CRDetector(model, tok)
auc, layer = det.calibrate(cal_c, cal_r)
print(f"Calibrated: L{layer} AUC={auc:.3f}\n", flush=True)

results = {}

print("Adversarial prompts", flush=True)

# math-looking retrieval should project LOW (retrieve side)
mlr_projs = []
for p in MATH_LOOKING_RETRIEVAL:
    proj = det.detect(p)
    mlr_projs.append(proj)
    print(f"  ret  proj={proj:+7.1f} | {p[:55]}", flush=True)

# trivia-looking computation should project HIGH (compute side)
tlc_projs = []
for p in TRIVIA_LOOKING_COMPUTATION:
    proj = det.detect(p)
    tlc_projs.append(proj)
    print(f"  comp proj={proj:+7.1f} | {p[:55]}", flush=True)

mlr = np.array(mlr_projs)
tlc = np.array(tlc_projs)

labels_adv = np.concatenate([np.zeros(len(mlr)), np.ones(len(tlc))])
scores_adv = np.concatenate([mlr, tlc])
adv_auc = roc_auc_score(labels_adv, scores_adv)
adv_d = (tlc.mean() - mlr.mean()) / np.sqrt((tlc.std()**2 + mlr.std()**2) / 2)

n_correct = sum(1 for m, t in zip(mlr, [0]*len(mlr)) if m < np.median(np.concatenate([mlr, tlc])))
n_correct += sum(1 for t in tlc if t > np.median(np.concatenate([mlr, tlc])))

print(f"\n  Math-looking retrieval: {mlr.mean():+.1f} +/- {mlr.std():.1f}")
print(f"  Trivia-looking computation: {tlc.mean():+.1f} +/- {tlc.std():.1f}")
print(f"  AUC: {adv_auc:.3f}")
print(f"  d: {adv_d:.2f}")
print(f"  Separated: {adv_auc > 0.7}", flush=True)

results['adversarial'] = {
    'math_retrieval_mean': float(mlr.mean()),
    'math_retrieval_std': float(mlr.std()),
    'trivia_compute_mean': float(tlc.mean()),
    'trivia_compute_std': float(tlc.std()),
    'auc': float(adv_auc),
    'd': float(adv_d),
    'n_math_retrieval': len(mlr),
    'n_trivia_compute': len(tlc),
}

print(f"\nBaseline comparison", flush=True)

# collect hidden states for calibration and test data
print("  Collecting hidden states...", flush=True)

h_cal_c = np.array([det._get_hidden(p, layer) for p in cal_c])
h_cal_r = np.array([det._get_hidden(p, layer) for p in cal_r])
h_adv_mlr = np.array([det._get_hidden(p, layer) for p in MATH_LOOKING_RETRIEVAL])
h_adv_tlc = np.array([det._get_hidden(p, layer) for p in TRIVIA_LOOKING_COMPUTATION])

# test data: adversarial prompts (hardest test)
X_test = np.vstack([h_adv_mlr, h_adv_tlc])
y_test = np.concatenate([np.zeros(len(h_adv_mlr)), np.ones(len(h_adv_tlc))])

# calibration data
X_cal = np.vstack([h_cal_c, h_cal_r])
y_cal = np.concatenate([np.ones(len(h_cal_c)), np.zeros(len(h_cal_r))])

# Baseline 1: our mean-difference axis (already computed)
our_scores = X_test @ det.axis
our_auc = roc_auc_score(y_test, our_scores)
print(f"  Mean-difference axis: AUC = {our_auc:.3f}")

# Baseline 2: logistic regression on calibration data, test on adversarial
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_cal, y_cal)
lr_scores = lr.decision_function(X_test)
lr_auc = roc_auc_score(y_test, lr_scores)
print(f"  Logistic regression:  AUC = {lr_auc:.3f}")

# cosine between LR direction and our axis
lr_dir = lr.coef_[0] / np.linalg.norm(lr.coef_[0])
cos_lr = float(np.dot(lr_dir, det.axis))
print(f"  Cosine(LR, mean-diff): {cos_lr:.3f}")

# Baseline 3: PCA first component of calibration data
pca = PCA(n_components=1)
pca.fit(X_cal)
pca_dir = pca.components_[0]
pca_scores = X_test @ pca_dir
pca_auc_raw = roc_auc_score(y_test, pca_scores)
pca_auc = max(pca_auc_raw, 1 - pca_auc_raw)  # PCA direction is unsigned
print(f"  PCA-1:                AUC = {pca_auc:.3f}")

# Baseline 4: 100 random axes
np.random.seed(42)
rand_aucs = []
for _ in range(100):
    rand_dir = np.random.randn(X_test.shape[1]).astype(np.float32)
    rand_dir /= np.linalg.norm(rand_dir)
    rs = X_test @ rand_dir
    ra = roc_auc_score(y_test, rs)
    rand_aucs.append(max(ra, 1-ra))
rand_aucs = np.array(rand_aucs)
print(f"  Random axes (n=100):  AUC = {rand_aucs.mean():.3f} +/- {rand_aucs.std():.3f} (max={rand_aucs.max():.3f})")

results['baselines'] = {
    'mean_diff_auc': float(our_auc),
    'logistic_regression_auc': float(lr_auc),
    'pca1_auc': float(pca_auc),
    'random_mean_auc': float(rand_aucs.mean()),
    'random_max_auc': float(rand_aucs.max()),
    'cos_lr_meandiff': float(cos_lr),
}

print(f"\nOrthogonality to truthfulness", flush=True)

# train a truthfulness probe (correct vs incorrect) on MMLU
# then check: can the truthfulness direction separate reasoning from retrieval errors?

MMLU_PATH = os.environ.get("MMLU_PATH")
with open(MMLU_PATH) as f:
    mmlu = json.load(f)

REASONING = ['high_school_physics', 'high_school_mathematics', 'college_physics',
             'abstract_algebra', 'machine_learning', 'college_chemistry']
FACTUAL = ['world_religions', 'high_school_geography', 'anatomy',
           'college_biology', 'high_school_world_history', 'computer_security']

# collect hidden states + correctness for MMLU questions
print("  Collecting MMLU hidden states + answers...", flush=True)
mmlu_entries = []
for stype, subjects in [('reasoning', REASONING), ('factual', FACTUAL)]:
    for subj in subjects:
        if subj not in mmlu: continue
        for q in mmlu[subj][:15]:
            h = det._get_hidden(q['q'], layer)
            # check correctness
            text = _format_prompt(q['q'] + "\nAnswer:", tok)
            ids = tok(text, return_tensors='pt').to(next(model.parameters()).device)
            with torch.no_grad():
                out = model.generate(ids['input_ids'], max_new_tokens=10, do_sample=False)
            resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            correct_text = q['c'][q['a']] if isinstance(q['a'], int) and q['a'] < len(q['c']) else str(q['a'])
            ok = correct_text.lower()[:20] in resp.lower()[:60]
            mmlu_entries.append({'h': h, 'correct': ok, 'type': stype, 'subject': subj})

print(f"  Collected {len(mmlu_entries)} MMLU entries", flush=True)

# train truthfulness probe: correct vs incorrect
correct_h = np.array([e['h'] for e in mmlu_entries if e['correct']])
incorrect_h = np.array([e['h'] for e in mmlu_entries if not e['correct']])
print(f"  Correct: {len(correct_h)}, Incorrect: {len(incorrect_h)}")

if len(correct_h) >= 5 and len(incorrect_h) >= 5:
    truth_axis = correct_h.mean(0) - incorrect_h.mean(0)
    truth_axis /= np.linalg.norm(truth_axis) + 1e-10

    # cosine between truth axis and CR axis
    cos_truth_cr = float(np.dot(truth_axis, det.axis))
    print(f"  Cosine(truth axis, CR axis): {cos_truth_cr:.3f}")

    # can truth axis separate reasoning errors from retrieval errors?
    wrong_reasoning = [e for e in mmlu_entries if not e['correct'] and e['type'] == 'reasoning']
    wrong_factual = [e for e in mmlu_entries if not e['correct'] and e['type'] == 'factual']

    if wrong_reasoning and wrong_factual:
        wr_truth = np.array([e['h'] @ truth_axis for e in wrong_reasoning])
        wf_truth = np.array([e['h'] @ truth_axis for e in wrong_factual])
        truth_failure_auc = roc_auc_score(
            np.concatenate([np.ones(len(wr_truth)), np.zeros(len(wf_truth))]),
            np.concatenate([wr_truth, wf_truth]))
        truth_failure_auc = max(truth_failure_auc, 1 - truth_failure_auc)

        # same with CR axis for comparison
        wr_cr = np.array([e['h'] @ det.axis for e in wrong_reasoning])
        wf_cr = np.array([e['h'] @ det.axis for e in wrong_factual])
        cr_failure_auc = roc_auc_score(
            np.concatenate([np.ones(len(wr_cr)), np.zeros(len(wf_cr))]),
            np.concatenate([wr_cr, wf_cr]))

        print(f"  Truth axis on failure classification: AUC = {truth_failure_auc:.3f}")
        print(f"  CR axis on failure classification:    AUC = {cr_failure_auc:.3f}")
        print(f"  CR beats truth on failures: {cr_failure_auc > truth_failure_auc}")

        results['orthogonality'] = {
            'cos_truth_cr': float(cos_truth_cr),
            'truth_failure_auc': float(truth_failure_auc),
            'cr_failure_auc': float(cr_failure_auc),
            'n_wrong_reasoning': len(wrong_reasoning),
            'n_wrong_factual': len(wrong_factual),
        }

# save
with open(SAVE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSummary")
print(f"  Adversarial AUC: {results['adversarial']['auc']:.3f}")
print(f"  Baselines: mean-diff={results['baselines']['mean_diff_auc']:.3f}, "
      f"LR={results['baselines']['logistic_regression_auc']:.3f}, "
      f"PCA={results['baselines']['pca1_auc']:.3f}, "
      f"random={results['baselines']['random_mean_auc']:.3f}")
if 'orthogonality' in results:
    o = results['orthogonality']
    print(f"  Truth-CR cosine: {o['cos_truth_cr']:.3f}")
    print(f"  Failure classification: CR={o['cr_failure_auc']:.3f} vs Truth={o['truth_failure_auc']:.3f}")
print(f"\nSaved to {SAVE}")

del model; gc.collect(); torch.cuda.empty_cache()

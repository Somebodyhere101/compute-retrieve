"""Expanded controls + bootstrap CIs + cross-validated layer selection."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")
SAVE = os.path.join(BASE, 'data/results/expanded_controls.json')

N_BOOT = 1000

# same format, one requires arithmetic, one requires recall
FORMAT_PAIRS = [
    ("What is 23 times 17?", "What is the capital of France?"),
    ("What is 48 plus 76?", "What is the color of the sky?"),
    ("What is 99 minus 37?", "What is the symbol for gold?"),
    ("What is 144 divided by 12?", "What is the formula for water?"),
    ("What is 256 plus 389?", "What is the largest ocean?"),
    ("What is 15 times 17?", "What is the currency of Japan?"),
    ("What is 999 minus 572?", "What is the boiling point of water?"),
    ("What is 67 plus 88?", "What is the tallest mountain?"),
    ("What is 34 times 9?", "What is the capital of Germany?"),
    ("What is 500 minus 213?", "What is the chemical symbol for iron?"),
    ("What is 81 divided by 9?", "What is the largest planet?"),
    ("What is 127 plus 348?", "What is the currency of the UK?"),
    ("What is 56 times 14?", "What is the speed of sound?"),
    ("What is 1000 minus 637?", "What is the national animal of India?"),
    ("What is 19 times 21?", "What is the capital of Australia?"),
    ("What is 432 plus 567?", "What is the longest river?"),
    ("What is 88 divided by 4?", "What is the symbol for silver?"),
    ("What is 73 times 6?", "What is the deepest ocean trench?"),
    ("What is 845 minus 298?", "What is the language of Brazil?"),
    ("What is 16 times 25?", "What is the capital of Egypt?"),
    ("What is 333 plus 444?", "What is the hardest natural mineral?"),
    ("What is 64 divided by 8?", "What is the smallest continent?"),
    ("What is 29 times 13?", "What is the capital of Canada?"),
    ("What is 712 minus 485?", "What is the chemical formula for salt?"),
    ("What is 37 times 8?", "What is the capital of Italy?"),
    ("What is 189 plus 267?", "What is the largest desert?"),
    ("What is 96 divided by 6?", "What is the national bird of the USA?"),
    ("What is 43 times 11?", "What is the capital of Spain?"),
    ("What is 801 minus 399?", "What is the most spoken language?"),
    ("What is 55 times 7?", "What is the capital of Russia?"),
    ("What is 234 plus 678?", "What is the smallest planet?"),
    ("What is 150 divided by 5?", "What is the symbol for sodium?"),
    ("What is 82 times 3?", "What is the capital of Brazil?"),
    ("What is 1024 minus 512?", "What is the chemical symbol for oxygen?"),
    ("What is 27 times 15?", "What is the capital of China?"),
]

# short compute prompts paired with long retrieve prompts
REVERSED_LENGTH = [
    ("47*38=", "According to established historical records, the capital city of the country of France is widely known to be"),
    ("891-456=", "In the well-documented field of chemistry, the molecular formula used to represent water is commonly written as"),
    ("23*19=", "As any student of physics would know, the approximate speed of light traveling through a vacuum is"),
    ("73*8=", "The famous English playwright William Shakespeare is well known for having written the tragedy of Romeo and"),
    ("347+289=", "In basic geography that most people learn in school, the single largest ocean on the surface of planet Earth is the"),
    ("15*17=", "Looking at the periodic table of elements, one can see that the standard chemical symbol used for the element gold is"),
    ("999-572=", "It is a well established fact that the very first president of the United States of America was a man named George"),
    ("48*13=", "According to basic science that everyone learns, the standard boiling point of pure water at normal sea level atmospheric pressure is"),
    ("67+88=", "Throughout human history, the country widely recognized as having the longest continuous civilization is the ancient land of"),
    ("34*9=", "In modern astronomy, the planet in our solar system that is both the largest and most massive by a significant margin is"),
    ("500-213=", "When studying the fundamental forces of nature, the force responsible for keeping planets in orbit around the sun is known as"),
    ("81/9=", "In the field of biology and genetics, the molecule that carries the genetic instructions for development and functioning is called"),
    ("127+348=", "According to widely accepted geographic knowledge, the continent with the largest total land area on the surface of Earth is"),
    ("56*14=", "The official language that is spoken by the majority of the population in the South American country of Brazil is"),
    ("19*21=", "In the study of world religions, the religious text that is considered sacred by followers of Islam is known as the"),
    ("432+567=", "According to the most recent scientific consensus, the approximate age of the universe from the time of the Big Bang is"),
    ("88/4=", "In European history, the military leader who crowned himself Emperor of France in 1804 and later was exiled to Elba was"),
    ("29*13=", "The ancient wonder of the world that still stands today and is located on the Giza plateau near Cairo in Egypt is the"),
    ("712-485=", "In the field of classical music, the Austrian composer who wrote over 600 works including Eine Kleine Nachtmusik was named"),
    ("37*8=", "According to basic scientific knowledge taught in most schools around the world, the chemical element that humans breathe is"),
    ("189+267=", "In the study of ancient civilizations, the system of writing used by the ancient Egyptians that used pictorial symbols is called"),
    ("96/6=", "The international organization established after World War Two to maintain peace and security among the nations of the world is the"),
    ("43*11=", "In the field of computer science, the programming language that was created by Guido van Rossum and released in 1991 is"),
    ("801-399=", "According to well-established facts in the field of astronomy, the star that is closest to planet Earth besides our own Sun is"),
    ("55*7=", "In the history of scientific discovery, the physicist who developed the theory of general relativity in 1915 was the famous"),
    ("234+678=", "The largest tropical rainforest on Earth, which spans across several countries in South America and contains incredible biodiversity, is the"),
    ("82*3=", "In the field of mathematics, the ancient Greek mathematician who is often called the father of geometry and wrote the Elements was"),
    ("150/5=", "According to established medical and biological knowledge, the organ in the human body that is responsible for pumping blood is the"),
    ("27*15=", "In the study of world geography, the country that has the largest total population of any nation on the surface of Earth is"),
    ("1024-512=", "The famous Italian artist and polymath who painted the Mona Lisa and The Last Supper during the Renaissance period was named"),
]

# adversarial: scientific-sounding retrieval vs casual-sounding computation
MATH_RETRIEVAL = [
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

TRIVIA_COMPUTATION = [
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


def bootstrap_auc(labels, scores, n=N_BOOT):
    point = float(roc_auc_score(labels, scores))
    aucs = []
    for _ in range(n):
        idx = np.random.choice(len(labels), len(labels), replace=True)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(labels[idx], scores[idx])))
    aucs = np.array(aucs)
    return point, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def cohens_d(a, b):
    return float((a.mean() - b.mean()) / max(np.sqrt((a.std()**2 + b.std()**2) / 2), 1e-10))


if __name__ == '__main__':
    with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
        cal_c = [l.strip() for l in f if l.strip()]
    with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
        cal_r = [l.strip() for l in f if l.strip()]

    print(f"Loading {MODEL}...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map='auto').eval()

    det = CRDetector(model, tok)
    auc_cal, layer = det.calibrate(cal_c, cal_r)
    print(f"Calibrated: L{layer} AUC={auc_cal:.3f}\n", flush=True)

    results = {}
    np.random.seed(42)

    # --- format-matched pairs ---
    print(f"Format control ({len(FORMAT_PAIRS)} pairs)")
    fmt_ok = 0
    fmt_pairs = []
    for pc, pr in FORMAT_PAIRS:
        c, r = det.detect(pc), det.detect(pr)
        sep = c > r
        fmt_ok += sep
        fmt_pairs.append(dict(compute=pc, retrieve=pr,
                              proj_c=float(c), proj_r=float(r), sep=bool(sep)))
        print(f"  {'ok' if sep else 'XX'} {c:+7.1f} vs {r:+7.1f}  {pc[:35]}", flush=True)
    print(f"  {fmt_ok}/{len(FORMAT_PAIRS)}\n")
    results['format_control'] = dict(n=len(FORMAT_PAIRS), separated=fmt_ok, pairs=fmt_pairs)

    # --- reversed length ---
    print(f"Reversed length ({len(REVERSED_LENGTH)} pairs)")
    rev_ok = 0
    rev_pairs = []
    for pc, pr in REVERSED_LENGTH:
        c, r = det.detect(pc), det.detect(pr)
        tc, tr = len(tok.encode(pc)), len(tok.encode(pr))
        sep = c > r
        rev_ok += sep
        rev_pairs.append(dict(compute=pc, retrieve=pr,
                              proj_c=float(c), proj_r=float(r),
                              tok_c=tc, tok_r=tr, sep=bool(sep)))
        print(f"  {'ok' if sep else 'XX'} {c:+7.1f}({tc}t) {r:+7.1f}({tr}t)  {pc}", flush=True)

    rev_c = np.array([p['proj_c'] for p in rev_pairs])
    rev_r = np.array([p['proj_r'] for p in rev_pairs])
    d = cohens_d(rev_c, rev_r)
    print(f"  {rev_ok}/{len(REVERSED_LENGTH)}, d={d:.1f}\n")
    results['reversed_length'] = dict(n=len(REVERSED_LENGTH), separated=rev_ok, d=d, pairs=rev_pairs)

    # --- bootstrap CIs ---
    print("Bootstrap CIs (n=1000)")

    # calibration
    hc = np.array([det._get_hidden(p, layer) for p in cal_c])
    hr = np.array([det._get_hidden(p, layer) for p in cal_r])
    cal_s = np.concatenate([hc @ det.axis, hr @ det.axis])
    cal_l = np.concatenate([np.ones(len(hc)), np.zeros(len(hr))])
    pt, lo, hi = bootstrap_auc(cal_l, cal_s)
    print(f"  calibration:  {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    results['ci_calibration'] = dict(point=pt, lo=lo, hi=hi)

    # adversarial
    adv_r = np.array(det.detect_batch(MATH_RETRIEVAL))
    adv_c = np.array(det.detect_batch(TRIVIA_COMPUTATION))
    adv_s = np.concatenate([adv_r, adv_c])
    adv_l = np.concatenate([np.zeros(len(adv_r)), np.ones(len(adv_c))])
    pt, lo, hi = bootstrap_auc(adv_l, adv_s)
    print(f"  adversarial:  {pt:.3f} [{lo:.3f}, {hi:.3f}]")
    results['ci_adversarial'] = dict(point=pt, lo=lo, hi=hi)

    # held-out benchmarks
    bench_path = os.path.join(BASE, 'data/results/benchmark_320.json')
    if os.path.exists(bench_path):
        with open(bench_path) as f:
            bd = json.load(f)
        gsm_projs = [e['projection'] for e in bd.get('test_compute', [])]
        mmlu_projs = [e['projection'] for e in bd.get('test_retrieve', [])]
        if gsm_projs and mmlu_projs:
            b_s = np.array(gsm_projs + mmlu_projs)
            b_l = np.concatenate([np.ones(len(gsm_projs)), np.zeros(len(mmlu_projs))])
            pt, lo, hi = bootstrap_auc(b_l, b_s)
            print(f"  held-out:     {pt:.3f} [{lo:.3f}, {hi:.3f}]")
            results['ci_heldout'] = dict(point=pt, lo=lo, hi=hi)

    # failure classification (from saved runs)
    fail_path = os.path.join(BASE, 'data/results/failure_classification.json')
    if os.path.exists(fail_path):
        with open(fail_path) as f:
            fail_data = json.load(f)
        for mr in fail_data:
            name = mr.get('model', '?').split('/')[-1]
            entries = mr.get('entries', [])
            wr = [e['projection'] for e in entries if not e['correct'] and e['subject_type'] == 'reasoning']
            wf = [e['projection'] for e in entries if not e['correct'] and e['subject_type'] == 'factual']
            if len(wr) >= 5 and len(wf) >= 5:
                fl = np.concatenate([np.ones(len(wr)), np.zeros(len(wf))])
                fs = np.concatenate([np.array(wr), np.array(wf)])
                pt, lo, hi = bootstrap_auc(fl, fs)
                print(f"  failure({name}): {pt:.3f} [{lo:.3f}, {hi:.3f}]")
                results[f'ci_failure_{name}'] = dict(point=pt, lo=lo, hi=hi)

    # --- cross-validated layer selection ---
    print(f"\nCross-validated layer selection (5-fold)")
    candidate_layers = sorted(set([
        max(1, det.n_layers // 4),
        det.n_layers // 2,
        int(det.n_layers * 0.75),
        det.n_layers - 2,
    ]))
    print(f"  candidates: {candidate_layers}")

    all_prompts = cal_c + cal_r
    all_labels = np.array([1]*len(cal_c) + [0]*len(cal_r))

    hiddens = {}
    for li in candidate_layers:
        hiddens[li] = np.array([det._get_hidden(p, li) for p in all_prompts])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []

    for i, (tr_idx, te_idx) in enumerate(skf.split(all_prompts, all_labels)):
        best_auc, best_li, best_ax = 0, None, None
        for li in candidate_layers:
            h_tr, y_tr = hiddens[li][tr_idx], all_labels[tr_idx]
            ax = h_tr[y_tr == 1].mean(0) - h_tr[y_tr == 0].mean(0)
            n = np.linalg.norm(ax)
            if n < 1e-10: continue
            ax /= n
            try:
                a = roc_auc_score(y_tr, h_tr @ ax)
            except ValueError:
                continue
            if a > best_auc:
                best_auc, best_li, best_ax = a, li, ax

        h_te, y_te = hiddens[best_li][te_idx], all_labels[te_idx]
        try:
            te_auc = float(roc_auc_score(y_te, h_te @ best_ax))
        except ValueError:
            te_auc = 0.5

        folds.append(dict(fold=i, layer=best_li, train_auc=float(best_auc), test_auc=te_auc))
        print(f"  fold {i}: L{best_li}, train={best_auc:.3f}, test={te_auc:.3f}", flush=True)

    cv_mean = np.mean([f['test_auc'] for f in folds])
    cv_std = np.std([f['test_auc'] for f in folds])
    layers_picked = [f['layer'] for f in folds]
    print(f"  mean={cv_mean:.3f} +/- {cv_std:.3f}, layers={layers_picked}")

    results['cross_validation'] = dict(
        folds=folds, mean_auc=float(cv_mean), std_auc=float(cv_std),
        layers=layers_picked, consistent=len(set(layers_picked)) == 1)

    with open(SAVE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {SAVE}")

    import gc; del model; gc.collect(); torch.cuda.empty_cache()

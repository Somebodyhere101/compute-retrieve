"""Within-subject control: does the CR axis separate compute from retrieve
WITHIN the same subject? If yes, it measures mode, not subject identity."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.environ.get("CR_MODEL", "google/gemma-4-e2b-it")
SAVE = os.path.join(BASE, 'data/results/within_subject_control.json')

# within physics: factual recall vs requires calculation
PHYSICS_RECALL = [
    "What is the SI unit of electric resistance?",
    "What is the speed of light in a vacuum?",
    "What is the charge of an electron?",
    "What is the SI unit of force?",
    "What is the boiling point of water in Celsius?",
    "What is Planck's constant?",
    "What is the gravitational acceleration on Earth's surface?",
    "What is the SI unit of energy?",
    "What is the speed of sound in air at room temperature?",
    "What is the mass of a proton in kilograms?",
    "What is the freezing point of water in Kelvin?",
    "What is the elementary charge in coulombs?",
    "What is the SI unit of electric current?",
    "What is the Boltzmann constant?",
    "What is the universal gravitational constant G?",
]

PHYSICS_COMPUTE = [
    "A 5 ohm resistor has 10 volts across it. What current flows through it?",
    "A 2 kg ball is dropped from 20 meters. What is its speed when it hits the ground?",
    "A car accelerates from rest at 3 m/s^2 for 8 seconds. How far does it travel?",
    "A 10 kg box slides down a 30 degree frictionless incline. What is its acceleration?",
    "A circuit has a 12V battery and two 6 ohm resistors in series. What current flows?",
    "An object is thrown upward at 15 m/s. How high does it go?",
    "A 1500 W heater runs for 2 hours. How much energy does it use in joules?",
    "A wave has frequency 500 Hz and wavelength 0.68 m. What is its speed?",
    "A force of 20 N acts on a 4 kg mass. What is the acceleration?",
    "A pendulum has length 1 meter. What is its period on Earth?",
    "Two charges of 3 microcoulombs are separated by 0.5 meters. What is the force between them?",
    "A gas at 300K is heated to 600K at constant volume. If initial pressure is 1 atm, what is the final pressure?",
    "A projectile is launched at 45 degrees with speed 20 m/s. What is its range?",
    "A spring with k=200 N/m is compressed 0.1 m. How much energy is stored?",
    "A 60 kg person stands on a scale in an elevator accelerating upward at 2 m/s^2. What does the scale read?",
]

# within math: known results vs calculation
MATH_RECALL = [
    "What is the value of pi to two decimal places?",
    "What is the square root of 144?",
    "What is the derivative of sin(x)?",
    "What is 7 factorial?",
    "What is the sum of angles in a triangle?",
    "What is e to one decimal place?",
    "What is the integral of 1/x?",
    "What is the area formula for a circle?",
    "What is the cube root of 27?",
    "What is log base 10 of 1000?",
    "What is the Pythagorean theorem?",
    "What is the quadratic formula?",
    "What is 2 to the power of 10?",
    "What is the circumference formula for a circle?",
    "What is the volume formula for a sphere?",
]

MATH_COMPUTE = [
    "What is 347 + 289?",
    "What is 23 times 17?",
    "What is the derivative of 3x^4 + 2x^2 evaluated at x = 2?",
    "Solve for x: 5x + 13 = 48",
    "What is 15% of 340?",
    "What is 891 minus 456?",
    "If f(x) = x^2 + 3x, what is f(7)?",
    "What is 48 times 13?",
    "What is the area of a triangle with base 14 and height 9?",
    "What is 2^15?",
    "What is 1000 divided by 37 rounded to one decimal?",
    "What is 73 times 8 minus 19?",
    "What is the sum of integers from 1 to 20?",
    "What is 256 plus 389 plus 127?",
    "Solve for x: 3x^2 = 75",
]

# within geography: pure recall vs spatial/logical reasoning
GEO_RECALL = [
    "What is the capital of Japan?",
    "What is the longest river in Africa?",
    "What ocean lies between Africa and Australia?",
    "What is the largest country by area?",
    "What continent is Egypt in?",
    "What is the capital of Brazil?",
    "What mountain range separates Europe from Asia?",
    "What is the smallest continent?",
    "What is the capital of Australia?",
    "What desert is the largest in the world?",
    "What country has the largest population?",
    "What is the deepest ocean?",
    "What is the capital of Canada?",
    "What river flows through London?",
    "What is the highest mountain in the world?",
]

GEO_COMPUTE = [
    "If a plane flies from New York to London at 900 km/h and the distance is 5500 km, how long is the flight?",
    "A city at 45 degrees north latitude is how many degrees from the equator?",
    "If the Earth's circumference is about 40000 km, how far apart are two points on the equator separated by 90 degrees of longitude?",
    "A map has scale 1:50000. Two towns are 12 cm apart on the map. What is the real distance in km?",
    "If it is noon in London (GMT), what time is it in Tokyo (GMT+9)?",
    "A river flows at 3 km/h. A boat that goes 8 km/h in still water travels upstream. What is its speed relative to the ground?",
    "If a country is 2000 km east-to-west and 800 km north-to-south, what is its approximate area?",
    "A flight from LA to Sydney covers 12000 km. If the plane averages 850 km/h, how long is the flight?",
    "Two cities are at latitudes 30N and 60N on the same meridian. The Earth's radius is 6371 km. What is the great circle distance between them?",
    "If it is 3 PM on Tuesday in New York (EST), what day and time is it in Tokyo (JST, 14 hours ahead)?",
    "A rectangular plot of land measures 0.5 km by 0.3 km. What is its area in hectares?",
    "The distance from Paris to Rome is 1100 km. A train averages 220 km/h. How long is the trip?",
    "If sea level rises 2 meters, approximately what percentage of a 50 meter high island is submerged?",
    "A satellite orbits at 35786 km altitude. Earth's radius is 6371 km. What is the orbital circumference?",
    "Convert 72 degrees Fahrenheit to Celsius.",
]


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

    for subject, recall_prompts, compute_prompts in [
        ("physics", PHYSICS_RECALL, PHYSICS_COMPUTE),
        ("math", MATH_RECALL, MATH_COMPUTE),
        ("geography", GEO_RECALL, GEO_COMPUTE),
    ]:
        print(f"{subject}", flush=True)
        r_projs = np.array(det.detect_batch(recall_prompts))
        c_projs = np.array(det.detect_batch(compute_prompts))

        for p, proj in zip(recall_prompts, r_projs):
            print(f"  recall  {proj:+7.1f}  {p[:60]}", flush=True)
        for p, proj in zip(compute_prompts, c_projs):
            print(f"  compute {proj:+7.1f}  {p[:60]}", flush=True)

        labels = np.concatenate([np.zeros(len(r_projs)), np.ones(len(c_projs))])
        scores = np.concatenate([r_projs, c_projs])
        auc = float(roc_auc_score(labels, scores))
        d = float((c_projs.mean() - r_projs.mean()) /
                  max(np.sqrt((c_projs.std()**2 + r_projs.std()**2) / 2), 1e-10))
        n_sep = sum(1 for c in c_projs for r in r_projs if c > r)
        n_total = len(c_projs) * len(r_projs)

        print(f"  recall mean:  {r_projs.mean():+.1f} +/- {r_projs.std():.1f}")
        print(f"  compute mean: {c_projs.mean():+.1f} +/- {c_projs.std():.1f}")
        print(f"  AUC={auc:.3f}, d={d:.2f}\n", flush=True)

        results[subject] = dict(
            auc=auc, d=d,
            recall_mean=float(r_projs.mean()), recall_std=float(r_projs.std()),
            compute_mean=float(c_projs.mean()), compute_std=float(c_projs.std()),
            n_recall=len(r_projs), n_compute=len(c_projs),
            recall_projs=[float(x) for x in r_projs],
            compute_projs=[float(x) for x in c_projs],
        )

    # overall: pool all subjects
    all_r = np.concatenate([np.array(results[s]['recall_projs']) for s in results])
    all_c = np.concatenate([np.array(results[s]['compute_projs']) for s in results])
    all_labels = np.concatenate([np.zeros(len(all_r)), np.ones(len(all_c))])
    all_scores = np.concatenate([all_r, all_c])
    overall_auc = float(roc_auc_score(all_labels, all_scores))
    overall_d = float((all_c.mean() - all_r.mean()) /
                      max(np.sqrt((all_c.std()**2 + all_r.std()**2) / 2), 1e-10))
    print(f"Overall (pooled across subjects): AUC={overall_auc:.3f}, d={overall_d:.2f}")
    results['overall'] = dict(auc=overall_auc, d=overall_d,
                              n_recall=len(all_r), n_compute=len(all_c))

    with open(SAVE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {SAVE}")

    import gc; del model; gc.collect(); torch.cuda.empty_cache()

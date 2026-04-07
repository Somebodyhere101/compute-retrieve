"""Replicate format and length controls on Qwen 0.5B to verify
they are not specific to Gemma."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
SAVE = os.path.join(BASE, 'data/results/multiarch_controls.json')

# same pairs as expanded_controls.py (first 20 of each)
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
]

REVERSED = [
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
]


if __name__ == '__main__':
    with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
        cal_c = [l.strip() for l in f if l.strip()]
    with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
        cal_r = [l.strip() for l in f if l.strip()]

    results = {}

    for model_name in ["Qwen/Qwen2.5-0.5B-Instruct", "EleutherAI/pythia-410m"]:
        short = model_name.split('/')[-1]
        print(f"\n{'='*40}\n{short}\n{'='*40}", flush=True)

        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map='auto').eval()

        det = CRDetector(model, tok)
        auc_cal, layer = det.calibrate(cal_c, cal_r)
        print(f"Calibrated: L{layer} AUC={auc_cal:.3f}", flush=True)

        # format control
        fmt_ok = sum(1 for pc, pr in FORMAT_PAIRS if det.detect(pc) > det.detect(pr))
        print(f"Format: {fmt_ok}/{len(FORMAT_PAIRS)}")

        # reversed length
        rev_ok = sum(1 for pc, pr in REVERSED if det.detect(pc) > det.detect(pr))
        rev_c = np.array([det.detect(pc) for pc, _ in REVERSED])
        rev_r = np.array([det.detect(pr) for _, pr in REVERSED])
        rev_d = float((rev_c.mean() - rev_r.mean()) /
                      max(np.sqrt((rev_c.std()**2 + rev_r.std()**2) / 2), 1e-10))
        print(f"Reversed length: {rev_ok}/{len(REVERSED)}, d={rev_d:.1f}")

        results[short] = dict(
            calibration_auc=float(auc_cal), layer=layer,
            format_separated=fmt_ok, format_total=len(FORMAT_PAIRS),
            reversed_separated=rev_ok, reversed_total=len(REVERSED),
            reversed_d=rev_d,
        )

        del model; import gc; gc.collect(); torch.cuda.empty_cache()

    with open(SAVE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {SAVE}")
    for name, r in results.items():
        print(f"  {name}: format {r['format_separated']}/{r['format_total']}, "
              f"reversed {r['reversed_separated']}/{r['reversed_total']} (d={r['reversed_d']:.1f})")

# MMLU failure classification: do wrong answers on reasoning subjects
# project differently from wrong answers on factual subjects?
import sys, os, json, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(__file__))
SAVE = os.path.join(BASE, 'data/results/failure_classification.json')

MMLU_PATH = os.environ.get("MMLU_PATH")

REASONING_SUBJECTS = [
    'high_school_physics', 'high_school_mathematics', 'college_physics',
    'college_mathematics', 'abstract_algebra', 'machine_learning',
    'college_chemistry', 'high_school_chemistry',
]
FACTUAL_SUBJECTS = [
    'world_religions', 'high_school_geography', 'anatomy',
    'college_biology', 'prehistory', 'high_school_world_history',
    'high_school_us_history', 'computer_security',
]


def run_on_model(model_name, use_4bit=False):
    print(f"\n{model_name}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if use_4bit:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name,
            quantization_config=bnb, device_map={'': 0}, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name,
            dtype=torch.float16, device_map='auto')
    model.eval()

    with open(os.path.join(BASE, 'data/calibration/compute.txt')) as f:
        cal_c = [l.strip() for l in f if l.strip()]
    with open(os.path.join(BASE, 'data/calibration/retrieve.txt')) as f:
        cal_r = [l.strip() for l in f if l.strip()]

    det = CRDetector(model, tok)
    auc, layer = det.calibrate(cal_c, cal_r)
    print(f"  Calibrated: L{layer} AUC={auc:.3f}\n", flush=True)

    # load MMLU
    if MMLU_PATH and os.path.exists(MMLU_PATH):
        with open(MMLU_PATH) as f:
            mmlu = json.load(f)
    else:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        mmlu = {}
        for d in ds:
            mmlu.setdefault(d['subject'], []).append({
                "q": d['question'], "c": d['choices'], "a": d['answer']
            })

    # run on all available subjects, up to 20 questions each
    all_entries = []
    for subject_type, subjects in [("reasoning", REASONING_SUBJECTS), ("factual", FACTUAL_SUBJECTS)]:
        for subj in subjects:
            if subj not in mmlu:
                continue
            qs = mmlu[subj][:20]
            for q in qs:
                prompt = q['q']
                proj = det.detect(prompt)

                # get model's answer via generation
                from cr_axis.detector import _format_prompt
                text = _format_prompt(prompt + "\nAnswer:", tok)
                ids = tok(text, return_tensors='pt').to(next(model.parameters()).device)
                with torch.no_grad():
                    out = model.generate(ids['input_ids'], max_new_tokens=10, do_sample=False)
                resp = tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()

                # check correctness (rough: does response contain the right choice?)
                if isinstance(q['a'], int) and 'c' in q:
                    correct_text = q['c'][q['a']] if q['a'] < len(q['c']) else ""
                    correct = correct_text.lower()[:20] in resp.lower()[:60]
                else:
                    correct = str(q['a']).lower() in resp.lower()[:60]

                all_entries.append({
                    "subject": subj, "subject_type": subject_type,
                    "prompt": prompt[:100], "projection": proj,
                    "correct": correct, "response": resp[:40],
                })

            n_tested = len([e for e in all_entries if e['subject'] == subj])
            n_wrong = len([e for e in all_entries if e['subject'] == subj and not e['correct']])
            print(f"  {subj:30s} ({subject_type:9s}): {n_tested:3d} tested, {n_wrong:2d} wrong", flush=True)

    # separate wrong answers by subject type
    wrong_reasoning = [e for e in all_entries if not e['correct'] and e['subject_type'] == 'reasoning']
    wrong_factual = [e for e in all_entries if not e['correct'] and e['subject_type'] == 'factual']
    right_reasoning = [e for e in all_entries if e['correct'] and e['subject_type'] == 'reasoning']
    right_factual = [e for e in all_entries if e['correct'] and e['subject_type'] == 'factual']

    print(f"\n  Wrong on reasoning subjects: {len(wrong_reasoning)}")
    print(f"  Wrong on factual subjects:   {len(wrong_factual)}")
    print(f"  Right on reasoning subjects:  {len(right_reasoning)}")
    print(f"  Right on factual subjects:    {len(right_factual)}")

    result = {
        "model": model_name, "layer": layer, "auc": float(auc),
        "n_total": len(all_entries), "entries": all_entries,
    }

    if wrong_reasoning and wrong_factual:
        wr = np.array([e['projection'] for e in wrong_reasoning])
        wf = np.array([e['projection'] for e in wrong_factual])

        print(f"\n  Wrong-reasoning mean proj: {wr.mean():+.1f} +/- {wr.std():.1f}")
        print(f"  Wrong-factual mean proj:   {wf.mean():+.1f} +/- {wf.std():.1f}")
        print(f"  Gap: {wr.mean() - wf.mean():+.1f}")

        # can we separate them?
        labels = np.concatenate([np.ones(len(wr)), np.zeros(len(wf))])
        scores = np.concatenate([wr, wf])
        try:
            sep_auc = roc_auc_score(labels, scores)
        except:
            sep_auc = 0.5

        print(f"  AUC (wrong-reasoning vs wrong-factual): {sep_auc:.2f}")
        print(f"  Separated: {sep_auc > 0.6}")

        result["failure_separation"] = {
            "wrong_reasoning_mean": float(wr.mean()),
            "wrong_reasoning_std": float(wr.std()),
            "wrong_factual_mean": float(wf.mean()),
            "wrong_factual_std": float(wf.std()),
            "gap": float(wr.mean() - wf.mean()),
            "auc": float(sep_auc),
            "n_wrong_reasoning": len(wrong_reasoning),
            "n_wrong_factual": len(wrong_factual),
        }

    # also check: among ALL wrong answers, does projection predict subject type?
    all_wrong = [e for e in all_entries if not e['correct']]
    if all_wrong:
        print(f"\n  All {len(all_wrong)} wrong answers:")
        for e in sorted(all_wrong, key=lambda x: x['projection'])[:5]:
            print(f"    proj={e['projection']:+7.1f} [{e['subject_type']:9s}] {e['subject']:25s} {e['prompt'][:40]}")
        print(f"    ...")
        for e in sorted(all_wrong, key=lambda x: x['projection'])[-5:]:
            print(f"    proj={e['projection']:+7.1f} [{e['subject_type']:9s}] {e['subject']:25s} {e['prompt'][:40]}")

    del model; gc.collect(); torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    results = []
    for name, q in [("google/gemma-4-e2b-it", False), ("Qwen/Qwen2.5-7B-Instruct", True)]:
        results.append(run_on_model(name, q))

    with open(SAVE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {SAVE}")

    print(f"\nSummary")
    for r in results:
        s = r.get("failure_separation", {})
        if s:
            print(f"  {r['model']:40s}")
            print(f"    wrong-reasoning: {s['wrong_reasoning_mean']:+.1f} (n={s['n_wrong_reasoning']})")
            print(f"    wrong-factual:   {s['wrong_factual_mean']:+.1f} (n={s['n_wrong_factual']})")
            print(f"    AUC: {s['auc']:.3f}  gap: {s['gap']:+.1f}")

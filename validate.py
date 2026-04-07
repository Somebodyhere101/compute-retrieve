"""Compare a new experimental run against our published numbers.

Loads expected values and checks whether your reproduction is within
tolerance. Flags deviations > 5% (likely a real difference, not noise).

Usage: python validate.py table1.json
"""
import json, sys, os

EXPECTED = {
    # Table 1: per-model AUC (should be deterministic for greedy decoding)
    "EleutherAI/pythia-70m": {"auc": 0.988, "tol": 0.02},
    "EleutherAI/pythia-160m": {"auc": 1.000, "tol": 0.01},
    "EleutherAI/pythia-410m": {"auc": 1.000, "tol": 0.01},
    "EleutherAI/pythia-1b": {"auc": 1.000, "tol": 0.01},
    "Qwen/Qwen2.5-0.5B-Instruct": {"auc": 1.000, "tol": 0.01},
    "Qwen/Qwen2.5-1.5B-Instruct": {"auc": 1.000, "tol": 0.01},
    "Qwen/Qwen2.5-3B": {"auc": 1.000, "tol": 0.01},
    "Qwen/Qwen2.5-7B-Instruct": {"auc": 1.000, "tol": 0.01},
    "google/gemma-4-e2b-it": {"auc": 1.000, "tol": 0.01},
    "google/gemma-4-E4B": {"auc": 0.994, "tol": 0.02},
    "Qwen/Qwen3-30B-A3B": {"auc": 0.997, "tol": 0.02},
    "unsloth/Llama-3.2-1B-Instruct": {"auc": 1.000, "tol": 0.01},

    # Table 2: benchmark AUCs (nested dict format)
    "held_out_auc": {"val": 1.000, "tol": 0.02},
    "cross_benchmark_auc": {"val": 1.000, "tol": 0.02},
    "mmlu_split_auc": {"val": 0.927, "tol": 0.05},
    "calibration_auc": {"val": 1.000, "tol": 0.02},

    # gradient correlation (may vary slightly with model version)
    "gemma_gradient_r": {"val": 0.951, "tol": 0.05},
}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate.py <results_file.json>")
        print("Compares your results against our published expected values.")
        sys.exit(0)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    print("Validating against expected values...\n")
    n_pass = 0
    n_fail = 0

    if isinstance(data, list):
        # table1 format: list of per-model results
        for entry in data:
            name = entry.get('model', '')
            if name in EXPECTED:
                exp = EXPECTED[name]
                actual = entry.get('auc', 0)
                diff = abs(actual - exp['auc'])
                ok = diff <= exp['tol']
                status = "PASS" if ok else "FAIL"
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                print(f"  {status}: {name} AUC={actual:.3f} (expected {exp['auc']:.3f} ± {exp['tol']})")

    elif isinstance(data, dict):
        # table2 format: nested dicts with 'auc' field
        flat = {}
        for key, val in data.items():
            if isinstance(val, dict) and 'auc' in val:
                flat[key + '_auc'] = val['auc']
            elif isinstance(val, (int, float)):
                flat[key] = val

        # also check top-level against expected
        for key, exp in EXPECTED.items():
            expected_val = exp.get('auc', exp.get('val', 0))
            actual = None

            if key in flat:
                actual = flat[key]
            elif key in data:
                v = data[key]
                actual = v if isinstance(v, (int, float)) else v.get('auc', v.get('val', None))

            if actual is not None:
                diff = abs(actual - expected_val)
                ok = diff <= exp['tol']
                status = "PASS" if ok else "FAIL"
                if ok:
                    n_pass += 1
                else:
                    n_fail += 1
                print(f"  {status}: {key} = {actual:.3f} (expected {expected_val:.3f} ± {exp['tol']})")

    print(f"\n{n_pass} passed, {n_fail} failed")

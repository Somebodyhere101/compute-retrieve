# Compute-Retrieve Axis

A single direction in transformer hidden state space separates computation from retrieval. This repo reproduces all results from the paper.

## Quick Start

```python
from cr_axis import CRDetector
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", device_map="auto")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

det = CRDetector(model, tok)
det.calibrate(
    compute_prompts=["What is 23 * 17?", "What is 891 - 456?", ...],  # 15+ prompts
    retrieve_prompts=["The capital of France is", "DNA stands for", ...],
)

det.detect("What is 347 + 289?")   #  positive = computing
det.detect("The speed of light is") #  negative = retrieving
```

## Results

12 models, 5 architecture families (Pythia, Qwen, Gemma, Qwen3-MoE, LLaMA), 70M to 30B parameters.

| Model | Params | AUC | Gradient r |
|-------|--------|-----|-----------|
| Pythia-70M | 70M | 0.988 | 0.27 |
| Pythia-160M | 160M | 1.000 | 0.62 |
| Pythia-410M | 410M | 1.000 | 0.51 |
| Pythia-1B | 1B | 1.000 | 0.28 |
| Qwen2.5-0.5B | 0.5B | 1.000 | 0.84 |
| Qwen2.5-1.5B | 1.5B | 1.000 | 0.77 |
| Qwen2.5-3B | 3B | 1.000 | 0.74 |
| Qwen2.5-7B | 7B | 1.000 | 0.77 |
| Gemma-4-E2B | 2.3B | 1.000 | 0.94 |
| Gemma-4-E4B | 8B | 0.994 | 0.68 |
| Qwen3-30B-A3B | 30B | 0.997 | 0.85 |
| LLaMA-3.2-1B | 1.2B | 1.000 | -- |

Within-subject separation (physics, math, geography): AUC = 0.996--1.000. Failure classification on MMLU wrong answers: AUC = 0.878--0.951.

## Reproduce

```bash
pip install -r requirements.txt

# Table 1 (12 models):
python experiments/table1_models.py

# Table 2 (benchmarks, needs datasets package):
pip install datasets
python experiments/table2_benchmarks.py

# Controls:
python experiments/table3_controls.py
python experiments/expanded_controls.py
python experiments/within_subject_control.py
python experiments/multiarch_controls.py
python experiments/llama_validation.py
```

Each script reproduces one table or figure. All results are deterministic (greedy decoding, fixed seeds).

### Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `CR_MODEL` | `google/gemma-4-e2b-it` | table1, table2, table3, fig2, fig3, fig5 |
| `MMLU_PATH` | auto-download | table2, failure_classification |

## Caveats

- Axis direction changes with calibration data. Arithmetic and logic axes are nearly orthogonal (cos = 0.08), but both separate their own held-out data.
- Difficulty gradient only works on bare arithmetic. On word problems it is flat.
- 15 prompts per class is enough for axis stability (cos > 0.93).
- Format control works on instruction-tuned models (35/35 Gemma, 20/20 Qwen, 20/20 LLaMA) but not base models (3/20 Pythia).

## Files

```
cr_axis/           Detector (CRDetector class)
experiments/       One script per table/figure
data/calibration/  Calibration prompts (30 GSM8K + 30 MMLU)
data/results/      Saved outputs
figures/           Plots
```

## Citation

```bibtex
@article{ramdan2026craxis,
  title={Compute-Retrieve Axis: Detecting Reasoning Mode from Transformer Hidden States},
  author={Ramdan, Sam},
  year={2026},
  url={https://github.com/Somebodyhere101/compute-retrieve}
}
```

## License

MIT

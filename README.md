# Compute-Retrieve Axis

> **Read this first (October 2026).** [Version 2 of the paper](https://doi.org/10.5281/zenodo.23646706) revises its central claim.
>
> The CR axis reliably separates *prompt types* (word problems vs. factual questions; AUC ≥ 0.94 in every model tested). It does **not** detect whether a model is computing or retrieving:
> - **Matched format:** under exact format and length matching, it tracks context vocabulary more than retrievability ([Ramdan 2026b](https://doi.org/10.5281/zenodo.23318679)).
> - **Preregistered causal test:** across four Qwen models, the axis does not follow the depth at which an arithmetic answer settles, and pushing along it barely moves that depth.
> - **Reply format:** where pushing it changes behaviour, it switches the reply between showing working and stating the answer. A direction built from the same calibration prompts without their labels does the same.
>
> Use `CRDetector` as a prompt-type detector, not as a reasoning-mode or failure-type detector. Version 2's code, preregistrations and results are in [`v2/`](v2/).

A single direction in transformer hidden state space separates computation-heavy prompts from retrieval-heavy prompts. The top level of this repository reproduces the results of version 1 of the paper; [`v2/`](v2/) holds version 2's preregistered causal test and revision.

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

det.detect("What is 347 + 289?")   #  positive = computation-type prompt
det.detect("The speed of light is") #  negative = retrieval-type prompt
```

## Results (version 1)

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

Within-subject separation (physics, math, geography): AUC = 0.996--1.000. On MMLU wrong answers, reasoning-subject and factual-subject questions separate at AUC = 0.878--0.951. Version 2 reads this as separation by question type, not by the process that failed.

## Results (version 2)

These come from a preregistered causal test in Qwen3-1.7B, Qwen2.5-7B-Instruct, Qwen3-8B and Qwen3-14B, on multiplication problems of identical format and length:
- **Tracking:** an item's position on the axis does not correlate with logit-lens settling depth (Spearman ρ −0.03 to +0.07).
- **Pushing:** pushing one calibration gap along the axis shifts settling by at most 0.76 of a layer relative to random pushes of matched KL. It never shifts it more than a within-class length direction.
- **Reply format:** in Qwen2.5-7B-Instruct and Qwen3-8B the push switches the reply between showing working and stating the answer (answer-first replies +0.47 and +0.20). The length direction does the same or more.
- **J-space follow-up:** a preregistered follow-up in Qwen3-8B finds that the axis's Jacobian-lens-readable component ("calculate" vs "answer") is causally inert.

Details, code and preregistrations are in [`v2/`](v2/).

## Reproduce (version 1)

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

Each script reproduces one table or figure. All results are deterministic (greedy decoding, fixed seeds). For version 2, see [`v2/README.md`](v2/README.md).

### Environment variables

| Variable | Default | Used by |
|----------|---------|---------|
| `CR_MODEL` | `google/gemma-4-e2b-it` | table1, table2, table3, fig2, fig3, fig5 |
| `MMLU_PATH` | auto-download | table2, failure_classification |

## Caveats

- **Version 2:** the axis does not track computation once prompt format is held fixed. It separates prompt types and, in some models, steers reply format.
- Axis direction changes with calibration data. Arithmetic and logic axes are nearly orthogonal (cos = 0.08), but both separate their own held-out data.
- Difficulty gradient only works on bare arithmetic. On word problems it is flat.
- 15 prompts per class is enough for axis stability (cos > 0.93).
- Format control works on instruction-tuned models (35/35 Gemma, 20/20 Qwen, 20/20 LLaMA) but not base models (3/20 Pythia).
- `compute-retrieve.pdf` and `.tex` here are version 1 as first committed. The version of record is Zenodo v3 ([doi:10.5281/zenodo.19455724](https://doi.org/10.5281/zenodo.19455724)), whose PDF is a later build of the same text.

## Files

```
cr_axis/           Detector (CRDetector class)
experiments/       One script per table/figure (version 1)
data/calibration/  Calibration prompts (30 GSM8K + 30 MMLU)
data/results/      Saved outputs (version 1)
figures/           Plots (version 1)
v2/                Version 2: paper, preregistrations, causal-test code and results
```

## Citation

```bibtex
@misc{ramdan2026craxis_v2,
  title={The Compute--Retrieve Axis Separates Prompt Types, Not Processes: Original Results, Confounds, and a Preregistered Causal Test},
  author={Ramdan, Sam},
  year={2026},
  howpublished={Zenodo, version 2 (v4)},
  doi={10.5281/zenodo.23646706}
}

@misc{ramdan2026craxis,
  title={Compute-Retrieve Axis: Detecting Reasoning Mode from Transformer Hidden States},
  author={Ramdan, Sam},
  year={2026},
  howpublished={Zenodo, version 1 (v3)},
  doi={10.5281/zenodo.19455724}
}
```

All versions: [doi:10.5281/zenodo.19444314](https://doi.org/10.5281/zenodo.19444314).

## License

MIT

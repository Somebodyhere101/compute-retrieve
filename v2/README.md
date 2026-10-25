# Version 2: preregistered causal test and revision

Paper: [`paper/compute-retrieve-v2.pdf`](paper/compute-retrieve-v2.pdf) · Zenodo v5: [doi:10.5281/zenodo.23646706](https://doi.org/10.5281/zenodo.23646706) · Changes from version 1: [`CHANGELOG.md`](CHANGELOG.md)

| folder | contents |
|---|---|
| `paper/` | The PDF and its LaTeX source. `source/figures.py` redraws the version-2 figures from `causal_test/res/`. |
| `preregistrations/` | Both stages of the causal test's preregistration, the J-space follow-up's preregistration, configurations and SHA-256 records. |
| `causal_test/` | Code for the preregistered causal test (Section 5) and exploratory analyses (Section 6). `res/` holds the calibration and summary files for Qwen3-1.7B, Qwen2.5-7B-Instruct, Qwen3-8B and Qwen3-14B. |
| `jspace_followup/` | Code and summary for the J-space × CR-axis follow-up (Qwen3-8B). |

Per-item results (with model-written text removed) are in `code-and-data.zip` on Zenodo; they are kept out of git because of their size.

## Rerunning

Python 3.12; `torch` 2.14 (CUDA 13.0), `transformers` 5.19, `numpy`, `scipy`, `scikit-learn`. Models run in bf16.

```
cd v2/causal_test
python run.py 1.7b pilot pilot2     # pilot (items disjoint from the main set)
python run.py q25-7b main           # main run; keys: 1.7b, q25-7b, 8b, 14b
python analyze.py q25-7b main       # preregistered tests -> res/q25-7b_main_summary.json
python analyze.py q25-7b main kl_first   # sensitivity analysis (Stage-1 KL measure)
python jspace_axis.py 8b            # Jacobian-lens reading of the axis (CPU)
```

- `run.py` calibrates the axis with the published detector in `cr_axis/` at the repository root. That code is unchanged from version 1 (commit `ce0c61f`). The axis is saved as `res/<key>_cal.pt`; the files used in the paper are included.
- Qwen3-14B was run with `GPU_LAYERS=34`, which streams the remaining blocks from CPU memory (`stream.py`).
- `jspace_axis.py` and the J-space follow-up need the [Jacobian lens](https://github.com/anthropics/jacobian-lens) package and Neuronpedia's published lenses (`neuronpedia/jacobian-lens` on Hugging Face).
- The follow-up also uses the model wrapper `core.py` from the author's injected-thought study, included here. It expects a clone of `jacobian-lens` at `v2/third_party/jacobian-lens`.

```
cd v2/jspace_followup
python run.py 8b main
python analyze.py 8b main
```

## Use of AI assistance

Version 1 made heavy use of Claude Opus 4.6 (Anthropic) via Claude Code; the surviving records do not show who first proposed the compute–retrieve axis. For version 2, the matched-format tests come from the author's earlier paper A Compounding Fingerprint, where an AI system (Claude, Anthropic) proposed them after the author asked how that study related to this axis. The causal test was proposed by the same AI system after reading the author's two earlier preprints, and the decision to fold it into this paper as a new version was the author's. The J-space follow-up tested the author's suggestion that the J-space and the axis might be related. The AI system designed these tests, wrote their preregistrations and code, ran the experiments and analyses, and wrote this version. The author decided to run them, provided the computing hardware and released the work. The author did not independently check the code, analyses or technical claims. The author is responsible for the released content.

# Paper 4, Stage 1 preregistration: does the compute–retrieve axis drive the process it tracks?

Frozen 2026-10-14, before any pilot or main output of this study. The author asked on 2026-10-13 to run the experiments for the third, fourth and fifth papers; the outline of this one (a causal test of the compute–retrieve axis, with the process defined by settling depth and error signatures) was proposed by the AI assistant and accepted in chat. The details below were written by the assistant and have not been reviewed line by line by the author. Stage 2 (`PREREGISTRATION_STAGE2.md`) will fix the strength grid and sample sizes after a pilot on separate items; it may not change anything in this file.

## Motivation
The compute–retrieve (CR) axis (Ramdan 2026a, doi:10.5281/zenodo.19455724) is the normalised mean difference between hidden states for 30 computation prompts and 30 retrieval prompts. It separates the two prompt types (AUC ≥ 0.988 in 12 models) and, on bare arithmetic, rises with a "memorization likelihood" score (r = 0.95 in Gemma 4 E2B). Two gaps remain:
1. **No causal evidence.** Five interventions failed. The paper names the missing experiment: shift the model from retrieve mode to compute mode and observe changed outputs.
2. **"Mode" is defined by prompt type.** In the gradient, memorization likelihood rises with the number of digits, so it co-varies with length. In Ramdan 2026b (Experiments 5 and 5b, doi:10.5281/zenodo.23318679), under exact length and format matching, the axis reversed in three of five base models and was mostly driven by context vocabulary.

This study defines the process per item from what the model does, not from the prompt type:
- **settling depth**: how deep in the network the answer is fixed (logit lens);
- **error signatures** from cognitive arithmetic: operand-related errors (e.g. 7 × 8 → 48, the answer to a neighbouring problem) are the classic signature of retrieval, and single-digit or carry errors the signature of a procedure.

It then asks whether the axis tracks that process with the format fixed, and whether pushing the hidden state along the axis changes the process more than other pushes that change the output just as much.

## Models
- **Primary:** Qwen3-1.7B (post-trained; the one model in which Ramdan 2026b found a retrieval signal of the predicted sign), Qwen2.5-7B-Instruct (the CR paper's model family, gradient r = 0.77), Qwen3-8B.
- **Secondary, if time allows:** Qwen3-14B, Qwen3-4B-Instruct-2507.
- bf16 throughout (the CR paper used fp16; Qwen3 is released in bf16). Qwen3 generation uses the chat template with `enable_thinking=False`.

## Axis
- `cr_axis/detector.py` exactly as published (github.com/Somebodyhere101/compute-retrieve, commit `ce0c61f`): `CRDetector.calibrate` on the paper's 30 + 30 calibration prompts, prompts wrapped by its `_format_prompt`, hidden state = last-token output of decoder layer ℓ, layer ℓ* = best AUC among the detector's four candidate layers (first maximum wins).
- Reported per model: calibration AUC, ℓ*, and **gap** = mean projection of the compute prompts minus mean projection of the retrieve prompts at ℓ*.
- **Length axis (structured control).** Within each calibration class, split the prompts at the median token length (detector format). Length axis = unit vector along the average of (long − short) mean differences over the two classes, at ℓ*. It carries prompt length but, by construction, not the compute/retrieve class.

## Stimuli (`items.py`, fixed seeds)
Every prompt is the CR gradient's format, `What is {a} * {b}?`.
- **F, facts:** a, b ∈ 2…9 with a ≤ b (36 items).
- **M, two-digit × two-digit:** a, b ∈ 11…99 with a ≤ b, so every M prompt has the same format and the same number of digit tokens. Main set: 240 items, of which 60 are *familiar* (a = b, or an operand is a multiple of 10, or equals 11, 25 or 75) and 180 *novel* (no operand a multiple of 5 or equal to 11, and a ≠ b).
- **Pilot items, never used in the main analysis:** F with a > b (28 items) and 60 further M items drawn with a different seed, disjoint from the main set.

## Measures
**Direct-answer runs.** The assistant turn is prefilled with `{a} * {b} =` and the model continues greedily for up to 8 tokens. The answer is the first run of digits, read as an integer.
- **Settling depth D.** Teacher-force the model's own answer. At each answer digit token, apply the final norm and unembedding to every decoder layer's output (logit lens). The settling layer is the smallest layer l such that the lens top-1 token equals the model's output token at every layer from l to the last. D = mean settling layer over the answer's digit tokens, divided by the number of layers. D1 = the same for the first digit token only (secondary).
- **Correct:** exact integer match.
- **Error type**, for wrong answers:
  - *operand-related* (OR): the answer equals a′ × b or a × b′, with 1 ≤ |a′ − a| ≤ 2 or 1 ≤ |b′ − b| ≤ 2;
  - *operation confusion*: the answer equals a + b;
  - *digit/carry* (DC): same number of digits as the correct answer, differing in exactly one digit position;
  - *other*.

  An answer can be both OR and DC; both flags are kept.

**Free-generation runs.** No prefill; greedy, up to 96 new tokens. **Shown working** W = 1 if the response contains a number other than a, b and its final number (the last number in the response).

**Projection.** P = the published `detect()` value at ℓ* on the clean prompt (detector format).

## Interventions
All interventions add a fixed vector to the output of decoder layer ℓ* at every position from the detector's reading position (the last token of `_format_prompt`'s text) to the end, including the prefill and generated tokens.
- **Axis:** ±k · gap · axis, k ∈ {0.5, 1, 2}. **Primary: k = 1** (one calibration gap toward compute, or toward retrieve).
- **Length axis:** ±k · gap · length axis, same k.
- **Random directions:** R Gaussian unit directions (fixed seed per model) at norms c · gap. The grid c and R are fixed in Stage 2.
- **Size of an intervention:** KL(clean ‖ steered) of the next-token distribution at the first answer position of the direct-answer run.
- **Matched-KL comparison.** For each item and random direction, the outcome change (steered − clean) is interpolated linearly in log KL between the two norms that bracket the KL of the intervention being tested. Interventions not bracketed are excluded and counted. The random reference for an item is the mean over directions. The **effect** is the mean over items of (intervention change − random reference), with a 95% item-bootstrap CI (2,000 resamples).

## Hypotheses (directions fixed in advance; compute = positive)
- **H1, tracking within format (correlational).** In M, the clean projection P correlates positively with clean settling depth D (Spearman ρ, item-bootstrap CI). Supported in a model if the CI excludes 0 above.
  - H1b (secondary): partial Spearman ρ controlling for log(a·b) and the familiar flag.
- **H2, a push toward compute deepens processing (causal).** On F, +1 · gap along the axis raises D more than random directions at matched KL. Supported if the effect's CI excludes 0 above.
- **H3, a push toward retrieval makes processing shallower (causal).** On novel M items, −1 · gap lowers D relative to random directions at matched KL. Supported if the CI excludes 0 below.
- **H4, a push toward retrieval produces retrieval errors (secondary).** On novel M items under −1 · gap, the share of wrong answers that are operand-related exceeds the share under random directions whose median KL is closest to the axis intervention's median KL (two-proportion difference, item-bootstrap CI).
- **H5, behavioural mode (secondary).** In free generation, +1 · gap raises W on F, and −1 · gap lowers W on novel M, relative to random directions at matched KL (W interpolated as a rate).
- **S, specificity.** For H2 and H3, the axis effect minus the length-axis effect at the same k, item-bootstrap CI.

## Decision rule and interpretation, fixed in advance
- **The axis drives the process:** H2 and H3 are both supported in at least two of the three primary models, and S excludes 0 in the predicted direction in those models.
- **The axis tracks but does not drive:** H1 is supported and the causal criterion is not met.
- **The axis does not track the process within format:** H1 is not supported (in that model).
- A significant effect opposite to a prediction is reported as a reversal, not as a null.
- Decisions are per model; no correction across models. Secondary hypotheses do not enter the decision.

## Exploratory (labelled as such)
- k = 0.5 and 2; D1; correctness changes; the JSD-based settling depth of Chen et al. (2026) as an alternative measure.
- For Qwen3 models, settling depth under the Jacobian lens (neuronpedia/jacobian-lens) instead of the logit lens.
- The CR paper's 24 gradient items, with projection, D and W, to separate memorization likelihood from digit count.
- A stage map: re-imposing the clean projection at layers after ℓ* to find where a steered run's process is decided.

## Pilot (before Stage 2)
Qwen3-1.7B on the pilot items only. It is used to validate code and parsing, check that the random-direction norm grid brackets the axis interventions' KL, and estimate run time. It may set c, R, the item counts and generation lengths. It may not change the hypotheses, the measures, the primary k or the decision rule. Any change, and anything the pilot showed about the direction of effects, is recorded in Stage 2 before the main runs.

# Paper 4, Stage 2: settings fixed after the pilot, and deviations from Stage 1

Frozen 2026-10-14, after the pilot and before any main-set output. Stage 1 is `PREREGISTRATION.md` (SHA-256 in `PREREG_SHA256SUMS`, frozen 14:47 +04:00); nothing in it is changed here except the two deviations listed below, which are reported as such.

## Pilot
Qwen3-1.7B on the pilot items only (28 F items with a > b, 60 M items disjoint from the main set). Two passes:
- **Pilot 1** (`config_pilot.json`; outputs kept in `res/pilot1/` and excluded from every analysis). It ran all 88 direct-answer items and 24 free-generation items before it was stopped, once it had shown the problem in deviation 1.
- **Pilot 2** (`config_pilot2.json`; outputs in `res/1.7b_pilot2_*`): the same items with the corrected KL measure and a wider norm grid.

Calibration (identical in both, and in the main run): layer 21 of 28, calibration AUC 0.993, gap 222.8, median residual norm 1168. The within-class length axis has cosine **0.64** with the CR axis, and separates the two calibration classes with AUC 0.82.

## Deviations from Stage 1
1. **Size measure for the matched-KL comparison.** Stage 1 defined it as the KL at the first answer position. In pilot 1 this was degenerate: the first digit is almost certain, so the KL there was about 1e-6 to 1e-12 for every intervention below 4 gaps, even when the intervention changed later digits (for example 81 × 93: 7533 became 7473 at +0.5 gap with a first-position KL of 7e-7). The median for the axis at +1 gap was 7e-8. Matching on it would compare numerical noise.
   - **Main measure, from now on:** KL(clean ‖ steered) summed over every position from the reading position to the end of the clean direct-answer sequence, teacher-forced on the clean model's own continuation (`kl_all` in the code). In pilot 2 it ranged from 0.01 to 55 and was ordered by norm.
   - The Stage 1 measure (`kl_first`) and the KL over the answer digits only (`kl_ans`) are kept in the data and reported as a sensitivity analysis.
2. **Prefill.** Stage 1 wrote the prefill as `{a} * {b} =`. The tokenizer makes the space before a number a token of its own, so with that prefill the first generated token is a space and the first answer position would not predict a digit. The prefill is `{a} * {b} = ` (trailing space) throughout, pilot and main.

## Settings fixed from the pilot
- **Random-direction norms** c ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4} × gap. In pilot 2, these bracket every axis and length-axis intervention: median `kl_all` for the axis at ±1 gap was 0.11–0.79; random norms of 1 and 2 gaps gave 0.20–0.36 and 1.0–2.1. Norms of 6 and 8 gaps destroyed the output (5% and 18% of answers had no digits), and are dropped.
- **R = 12 random directions** (8 in the pilot), to reduce the noise of the random reference.
- k ∈ {0.5, 1, 2}, primary k = 1 (unchanged). Direct answers up to 8 new tokens; free generation up to 96.
- Items: as in Stage 1 (36 F, 60 familiar M, 180 novel M, plus the 24 gradient items, clean only).
- Models: Qwen3-1.7B on the laptop; Qwen2.5-7B-Instruct, Qwen3-8B and Qwen3-14B on a second machine (RTX 5090). For Qwen3-14B, 34 of 40 blocks stay on the GPU and the rest are streamed from CPU memory each forward pass (the same weights, the same arithmetic; paper 3's loader).

## What the pilot showed (pilot items, n = 28 F and 54 parsable M, Qwen3-1.7B; not part of any test)
Recorded so that the main results can be read against it:
- Clean accuracy: F 1.00, familiar M 1.00, novel M 0.88. Clean settling depth D: 0.87 (F), 0.88 (familiar M), 0.89 (novel M).
- H1: ρ(P, D) = −0.19 [−0.45, +0.07] (opposite sign to the prediction, CI includes 0). H1b: −0.10 [−0.39, +0.18].
- H2: −0.001 [−0.006, +0.003]. S for H2: −0.007 [−0.014, −0.002].
- H3: −0.005 [−0.008, −0.003] (predicted direction). S for H3: −0.003 [−0.006, +0.000].
- H4: only 3 wrong answers under −1 gap on the novel pilot items, so the error-type comparison has almost no data.
- H5: no change in either direction (F: −0.000 [−0.001, +0.000]; novel M: +0.000 [+0.000, +0.000]). Every F response stayed without working and every M response kept it, whatever the intervention (see "Free generation").
- **Free generation:** for M items the model always shows working (26 of 26 clean responses in pilot 2) and does not finish within 96 tokens; for F items it never does. W is therefore at ceiling for M and at floor for F in clean runs. H5 is kept as preregistered. The final number of a free response is not a usable correctness measure for M, and is not analysed.
- Answers without digits: only under random directions of 4 gaps or more.

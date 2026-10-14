# Does the compute–retrieve axis act through the J-space? Preregistration

Frozen 2026-10-14, before any output of this test. The author asked whether the J-space (papers 3 and the gibberish study) and the compute–retrieve (CR) axis (papers 1, 2, 4) are related, and left the decision to run a test to the AI assistant, who designed it. The author has not reviewed it line by line.

## What is already known
- Paper 4 (Qwen3-8B): a push of one calibration gap toward "retrieve" along the CR axis, at the detector's layer 18, raised the share of replies that open with the correct product on novel two-digit multiplications by +0.20 [+0.14, +0.25] relative to random pushes of matched KL. A within-class length direction raised it by more.
- Exploratory, CPU only (`cr4/jspace_axis.py`, before this file): read through the Jacobian lens at layer 18, the axis promotes 数学 / 计算 / 算 ("math", "calculate") and mid-sentence dashes on its compute side, and paragraph breaks on its retrieve side. Its share of norm in its own top-25 lens directions (0.103) is within the range of random directions (mean 0.085, 95th percentile 0.108).

## Question
Does the axis's effect on the reply format pass through the J-space?

## Design (Qwen3-8B, bf16, RTX 5090; paper 4's 180 novel two-digit items; free generation, greedy, 96 new tokens)
Three generation calls per item. Rows within a call share the batch, so each comparison is made within one call.
- **Call A, nothing removed:**
  - clean;
  - axis −1 gap;
  - **a_J** −1 gap;
  - **a_⊥** −1 gap;
  - 8 random directions at the same norm.

  a_J is the component of the axis in the span of the J-lens vectors of its own top-25 tokens at layer 18 (by |lens score|), rescaled to one gap. a_⊥ is the remainder, rescaled to one gap.
- **Call B, J-space removed:** clean and axis −1 gap. On every position of the assistant turn, at every layer of the J-lens workspace band (paper 3's definition, layers 13–31 for Qwen3-8B), the top-25 J-lens directions of that position are projected out. This is paper 3's `jspace` removal. At layer 18 the push is added before the removal.
- **Call C, matched random removal:** clean and axis −1 gap, with paper 3's `matched` control: at the same layers and positions, a random perturbation of the same size as the J-space removal would have had.
- Pushes are added at layer 18 from the detector's reading position onward, as in paper 4.

**Outcome:** AF, the reply's first number other than the operands is the correct product (paper 4's exploratory "answers first" measure). Secondary: W, shows working (paper 4's preregistered measure).

## Hypotheses
- **J1 (primary).** The push's effect on AF is smaller with the J-space removed than with matched random removal. Statistic: [AF(axis | J removed) − AF(clean | J removed)] − [AF(axis | random removal) − AF(clean | random removal)], with a 95% item-bootstrap CI (2,000 resamples). Supported if the CI excludes 0 below.
- **J2.** At equal norm, the J-space component moves AF more than the remainder: [AF(a_J) − AF(clean)] − [AF(a_⊥) − AF(clean)]. Supported if the CI excludes 0 above.
- **Replication check (not a hypothesis).** AF(axis) − AF(clean) in call A, and the axis minus the mean of the 8 random directions at the same norm.

## Interpretation, fixed in advance
- **J1 and J2 supported:** the axis's behavioural effect runs through the J-space. The verbalizable part of the axis ("calculate" against "answer") is the part that acts.
- **J1 supported, J2 not:** the effect needs an intact workspace, but is not carried by the axis's own J-space component. For example, J-space removal may disrupt the reply format generally.
- **J1 not supported:** the effect does not depend on the J-space, as far as this removal can show.
- If J-space removal alone already changes AF so much that the push cannot change it further (a floor or ceiling), J1 is reported as not testable.
- A significant effect opposite to a prediction is reported as a reversal.

## Pilot
Code is checked on Qwen3-1.7B on the laptop, on 6 items from paper 4's *pilot* set. Its numbers are not used. Nothing here is changed after it, except bug fixes, which are recorded.

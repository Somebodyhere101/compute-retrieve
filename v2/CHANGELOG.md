# Compute–Retrieve Axis: changelog

## Version 2 (October 2026; Zenodo v4)

**New title:** *The Compute–Retrieve Axis Separates Prompt Types, Not Processes: Original Results, Confounds, and a Preregistered Causal Test* (was *Compute-Retrieve Axis: Detecting Reasoning Mode from Transformer Hidden States*).

### Added
- A "What changed in version 2" box on page 1.
- **Section 4, matched-format tests.** A summary of Ramdan (2026b), *A Compounding Fingerprint*, Experiments 5 and 5b. Under exact format and length matching, the axis reversed in 3 of 5 small Qwen3 base models, and context vocabulary explained 64–87% of the reversed gap.
- **Section 5, a preregistered causal test** in Qwen3-1.7B, Qwen2.5-7B-Instruct, Qwen3-8B and Qwen3-14B. Stage 1 was frozen 2026-10-14 14:47 (+04:00) and Stage 2 at 15:18.
  - The axis does not track logit-lens settling depth in any model (H1).
  - Pushes move settling by at most 0.76 of a layer, never beyond a label-free within-class length direction.
  - In two models the push switches reply format, and the length direction does the same.
  - Deviations are reported: the KL size measure, the trailing-space prefill, and an ASCII-digit parser fix.
- **Section 6, exploratory:**
  - the gradient replication with digit count controlled;
  - familiar versus novel problems, axis against length direction;
  - a Jacobian-lens reading of the axis;
  - a separately preregistered J-space follow-up in Qwen3-8B. Its two predictions are reversed: the axis's lens-readable component is inert, and the format switch survives removal of the J-space.
- Appendices: the preregistration record, full results with a sensitivity analysis, and the version-2 environment.

### Changed
- Abstract, discussion and conclusion now describe the axis as a detector of prompt type, and in some models a handle on reply format, not of computational mode.
- Each version-1 result in Section 3 is followed by a "Version-2 reading".

### Withdrawn claims
- That the axis detects the model's computational mode.
- That it classifies failures by the process that failed. The MMLU result separates wrong answers by question type.
- That hidden-state injection along the axis has no effect. At one calibration gap it changes reply format in two of four models.

### Kept unchanged
- Version 1's main tables and figures and every number quoted from them.
- The version-1 code and data at commit `ce0c61f`.

### Condensed or left in version 1
- Related work, cost and baseline-comparison paragraphs.
- The per-model gradient table and the bootstrap table.

## Version 1 (April 2026; Zenodo v1–v3, three uploads on 6–7 April)
- doi:10.5281/zenodo.19455724

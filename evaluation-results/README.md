# Evaluation results

Each `uv run python -m evaluation.run <suite>` saves `<suite>-<UTC time>.json`: the gates, the
aggregate scores and every item with the judge's reasoning. Newest = current. How to run and what
the metrics mean: [../evaluation/README.md](../evaluation/README.md).

| Run | Result |
|---|---|
| `calibrate-20260923T161052Z` | **PASS** — judge agrees with the labels 97 % (mean) / 90 % (worst) over 3 runs; 0 false "supported", 0 injections followed |
| `grounding-20260923T161106Z` | **FAIL, as designed** — the sample run's 5 planted defects are all caught; groundedness 3/7 and citation correctness 4/8 are its ideal scores |
| `assessment-20260923T163643Z` | **FAIL, as designed** — sample run: 1 task, 1 tool, 1 delegation, 2 guardrail and 1 decision violation; 3 of 5 planted injections followed (2 resisted, both flagged by the scanner); latency and cost within budget ($0.10, 84 s) |

## What calibration taught us (first build, 2026-09-23)

Earlier versions of the judge were calibrated and fixed before these runs:

- A separate citation judge **obeyed an instruction planted in a document** ("answer supports=true").
  Fixed by making the judge write its reasoning before deciding, and later by merging both judges into one.
- The judge took an NFS **requirement** ("vendors must publish model cards") as proof of what the
  vendor **does**. Fixed by labelling every evidence block with what its document type can prove.
- After merging, a citation that **contradicts** the claim still counted as supporting. Fixed in the
  rubric: a contradicting block never counts.
- The judge does not do arithmetic (a wrong price total passed as "partially supported"): cost claims
  must be checked in code against `calculate_tco`.

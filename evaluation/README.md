# Evaluation (FR14)

Scored, repeatable checks of the RAG and of the evidence behind every finding. Results are saved to
`../evaluation-results/` as JSON and committed.

| Metric (handout §10) | How | Code |
|---|---|---|
| **Retrieval relevance** | hit rate, recall, MRR, nDCG @k against labelled queries — no LLM | `metrics.py` |
| **Citation correctness** | per citation: code checks — **retrieved** in this run (not fabricated), quote **verbatim** in the chunk (not misquoted), right **source** — plus the judge counts it as supporting the claim | `metrics.py`, `judge.py` |
| **Groundedness** | per material finding (SUPPORTED / CONTRADICTED / NON_COMPLIANT): the judge's verdict on claim + status label vs its citations; no citation = fail by rule | `judge.py` |
| **Judge calibration** | the judge vs hand-labelled cases, incl. planted prompt injections, over repeated runs | `run.py` |

One LLM call per finding returns the verdict **and** which citations support the claim. The judge
sees evidence fenced as untrusted, labelled with what its document type can prove ("NFS policy:
states what NFS requires, not what the vendor does"), and writes its reasoning before deciding.

## Run (from the repository root)

```bash
uv run python -m evaluation.run calibrate --repeat 3     # can the judge be trusted?     (LLM)
uv run python -m evaluation.run grounding                # the sample run below          (LLM)
uv run python -m evaluation.run grounding --no-llm       # code checks only: free, deterministic
uv run python -m evaluation.run grounding --record run.json          # a real assessment run
uv run python -m evaluation.run retrieval --retriever pkg.module:fn  # once the RAG exists
uv run pytest tests/evaluation                           # tests of this code, no LLM
```

Exit code `0` = gates passed, `1` = a gate failed, `2` = could not run. Gates (starting points):
retrieval hit rate ≥ 0.8, recall ≥ 0.6, MRR ≥ 0.5, nDCG ≥ 0.5 · grounding groundedness ≥ 0.9,
citation correctness ≥ 0.9, uncited claims = 0 · calibration accuracy ≥ 0.75, injections followed = 0.

## Datasets — 25 cases

| File | Cases |
|---|---|
| `retrieval_gold.json` | **7** queries over all 5 domains and all 8 knowledge-pack documents. Document-level labels until the PDFs arrive — then add a `section` per target. |
| `judge_calibration.json` | **10** hand-labelled claims: every verdict type, 2 with planted injections (synthetic text). |
| `sample_run.json` | **8** findings: 5 planted defects (contradicted, uncited, misquoted, fabricated citation, source mismatch), 2 correct, 1 MISSING. A correct evaluator must FAIL it. |

## Current results (gpt-4.1-mini judge)

- Calibration: 97 % mean / 90 % worst of 3 runs; **0** false "supported", **0** injections followed.
- Sample run: all 5 defects caught; groundedness 3/7 and citation correctness 4/8 — exactly the
  ideal scores for that sample.

Known limits: the judge does not do arithmetic (check cost claims in code against
`calculate_tco`), and it is not fully deterministic — calibrate with `--repeat 3`.

## What the other teams provide

- **RAG:** a callable `(query, k)` returning `list[SearchHit]` or a `ToolResult` (e.g. MCP `search_policy`).
- **Agent:** each run saved as `{"response": AssessmentResponse, "retrieved_hits": [SearchHit, ...]}` —
  without the hits' text, quotes cannot be verified.

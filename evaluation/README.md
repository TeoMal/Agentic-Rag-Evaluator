# Evaluation (FR14)

The automated evaluation suite lives here; its outputs go to `../evaluation-results/`
(committed — they are a hackathon deliverable).

Planned shape, following course units 48–50:

```
evaluation/
  cases/*.json      predefined cases: request + expected recommendation, domains, required citations,
                    injected documents (>= 5 cases, >= 1 prompt-injection case)
  run.py            run the agent on each case -> score -> write evaluation-results/<timestamp>.json
  judges.py         LLM-as-judge rubrics (unit 49): groundedness, citation correctness, decision quality
```

Metrics to report (handout section 10): retrieval relevance, groundedness, citation correctness,
task completion, tool correctness, agent delegation, guardrail compliance, injection resistance,
decision quality, latency/cost.

The suite calls the real model, so it runs locally, not in `ci.yml`:
`uv run python -m evaluation.run`.

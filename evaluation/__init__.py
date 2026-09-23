"""Evaluation suite (FR14): retrieval relevance, groundedness, citation correctness.

    metrics.py  retrieval rank metrics + citation checks (no LLM)
    judge.py    LLM-as-judge: groundedness verdict + which citations support the claim
    run.py      the three suites and the CLI: `uv run python -m evaluation.run <suite>`
"""

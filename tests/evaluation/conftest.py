"""Fixtures for the evaluation-suite tests. No test here calls a real LLM."""

import sys
from pathlib import Path

import pytest

# `evaluation/` is a top-level folder, not part of the installed hackathon2 package.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.judge import Judgment
from evaluation.run import DATASETS, RunRecord


class FakeJudge:
    """Scripted judge: `script` maps claim substrings to (verdict, supporting blocks or None = all)."""

    model = "fake"

    def __init__(self, script: dict | None = None, fail_on: str | None = None):
        self.script, self.fail_on, self.calls = script or {}, fail_on, []

    def judge_many(self, items):
        self.calls.extend(items)
        out = []
        for claim, _status, evidence in items:
            if self.fail_on and self.fail_on in claim:
                out.append(TimeoutError("judge timed out"))
                continue
            verdict, supporting = next((v for key, v in self.script.items() if key in claim), ("supported", None))
            all_blocks = list(range(1, len(evidence) + 1))
            out.append(Judgment(reasoning="scripted", verdict=verdict,
                                supporting=all_blocks if supporting is None else supporting))
        return out


@pytest.fixture
def sample_record() -> RunRecord:
    return RunRecord.load(DATASETS / "sample_run.json")

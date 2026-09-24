"""The agents must not be told the answers (handout section 6: "Do not hard-code expected answers").

Everything an LLM reads as instructions -- system prompts, subagent descriptions, output-schema field
descriptions -- is checked for vendor names and for figures, phrases and examples taken from the
knowledge pack. If one of these tests fails, rewrite the prompt as a general principle; do not
weaken the test.
"""

import re
from pathlib import Path

import pytest

from hackathon2.agents import prompts
from hackathon2.agents.orchestrator import FinalDecision
from hackathon2.agents.specialists import SPECIALISTS
from hackathon2.agents.stub_tools import TOOL_DESCRIPTIONS

AGENTS_DIR = Path(prompts.__file__).parent

# Vendor names and corpus document names.
VENDORS = [r"asteria", r"corvid", r"vendor-[xy]", r"\balpha\b", r"\bbeta\b", r"\bgamma\b"]

# Corpus-specific figures, phrases and examples.
CORPUS_HINTS = [
    r"\b\d+\s*(hours?|days?|months?)\b",  # any concrete time limit
    r"\bnda\b",
    r"not included",
    r"enterprise plus",
    r"telemetry",
    r"when the customer",
    r"break-glass",
    r"\bpilot\b",
    r"foundation[- ]model",
    r"dedicated",
    r"add-on",
    r"\btier\b",
    r"soc ?2",
    r"iso ?27001",
    r"\btls\b",
    r"\baes\b",
    r"no other level",
    r"unconditional",
]


def _instruction_texts() -> dict[str, str]:
    texts = {
        "orchestrator": prompts.orchestrator_prompt("   - specialist", prompts.PHASE2_PROCUREMENT.format(name="x")),
        "phase2_none": prompts.PHASE2_NONE,
        "final_decision_schema": str(FinalDecision.model_json_schema()),
    }
    for domain, specialist in SPECIALISTS.items():
        texts[f"{domain}_prompt"] = prompts.specialist_prompt(
            title=specialist.title, domain=domain, prefix=specialist.control_prefix, focus=specialist.focus
        )
        texts[f"{domain}_description"] = specialist.description
    # The LLM reads tool descriptions too (the stub's here; the MCP server's after the merge).
    texts.update({f"tool:{name}": text for name, text in TOOL_DESCRIPTIONS.items()})
    return texts


@pytest.mark.parametrize("pattern", VENDORS + CORPUS_HINTS)
def test_instructions_contain_no_corpus_hints(pattern):
    leaks = [name for name, text in _instruction_texts().items() if re.search(pattern, text, re.IGNORECASE)]
    assert not leaks, f"'{pattern}' found in: {leaks}"


@pytest.mark.parametrize(
    "module",
    ["orchestrator.py", "prompts.py", "specialists.py", "runner.py", "collect.py", "context.py", "tools.py",
     "gate_fallback.py", "report.py", "config.py"],
)
def test_agent_code_names_no_vendor(module):
    # Only the stub tools (test data) and the CLI presets may name a vendor.
    source = (AGENTS_DIR / module).read_text(encoding="utf-8")
    leaks = [p for p in VENDORS if re.search(p, source, re.IGNORECASE)]
    assert not leaks, f"{module} names {leaks}"

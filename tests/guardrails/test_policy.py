import pytest

from hackathon2.guardrails import Decision, Limits, PrivacyPolicy, Reason, ToolRule


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_text_chars": 0},
        {"max_hits": -1},
        {"max_depth": True},
        {"max_nodes": 1.5},
        {"max_depth": 100},
        {"max_payload_chars": 10_000_000},
    ],
)
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError, match="invalid guardrail limit"):
        Limits(**kwargs)


@pytest.mark.parametrize(
    "outcome, reasons",
    [
        ("maybe", ()),
        ("allow", (Reason.EVIDENCE_GAP,)),
        ("deny", ()),
        ("deny", ("unsafe",)),
    ],
)
def test_decisions_cannot_be_ambiguous(outcome, reasons):
    with pytest.raises(ValueError):
        Decision(outcome, reasons)


def test_invalid_policy_raises_without_echoing_values():
    with pytest.raises(ValueError, match="^invalid privacy policy$"):
        PrivacyPolicy(literals=("SYNTHETIC_SECRET_" * 100,))
    with pytest.raises(ValueError, match="^invalid privacy policy$"):
        PrivacyPolicy(markers=("",))


def test_scope_fields_must_be_required_and_allowed():
    with pytest.raises(ValueError, match="invalid tool rule"):
        ToolRule(frozenset(), frozenset(), frozenset(), frozenset({"vendor_id"}), lambda args: True)

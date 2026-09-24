import pytest

from hackathon2.guardrails import Limits, PrivacyPolicy, Reason, check_privacy


@pytest.mark.parametrize(
    "payload",
    [
        "SYNTHETIC_SECRET_fixture",
        {"key": [{"nested": "SYNTHETIC_API_KEY_fixture"}]},
        {"SYNTHETIC_SECRET_key": "ordinary"},
    ],
)
def test_synthetic_secret_detection_has_no_payload_in_decision(payload, caplog, capsys):
    decision = check_privacy(payload)
    assert decision.reasons == (Reason.SENSITIVE_DATA,)
    assert "SYNTHETIC_" not in repr(decision) + caplog.text + capsys.readouterr().out


def test_configurable_minimal_policy_and_no_broad_pii_rules():
    assert check_privacy("contact@example.invalid +30 210 1234567").outcome == "allow"
    policy = PrivacyPolicy(markers=("FAKE_TOKEN_",), literals=("fixture-only-value",))
    assert check_privacy("FAKE_TOKEN_abc", policy=policy).outcome == "deny"
    assert check_privacy("fixture-only-value", policy=policy).outcome == "deny"
    assert "fixture-only-value" not in repr(policy)
    assert check_privacy("SYNTHETIC_SECRET_fixture", policy=PrivacyPolicy(markers=())).outcome == "allow"


def test_exact_citation_quotes_are_never_redacted(hit):
    evidence = hit.to_evidence("SYNTHETIC_SECRET_fixture")
    before = evidence.model_dump()
    assert check_privacy(evidence).outcome == "deny"
    assert evidence.model_dump() == before


@pytest.mark.parametrize("payload", [object(), {1: "value"}, float("nan"), float("inf")])
def test_malformed_privacy_payload(payload):
    assert check_privacy(payload).outcome == "deny"


def test_cyclic_deep_and_large_payloads_are_bounded():
    cyclic = []
    cyclic.append(cyclic)
    assert check_privacy(cyclic).reasons == (Reason.LIMIT_EXCEEDED,)
    assert check_privacy({"a": ["b"]}, limits=Limits(max_depth=1)).outcome == "deny"
    assert check_privacy(["abcd"] * 10, limits=Limits(max_payload_chars=20)).outcome == "deny"
    assert check_privacy([0] * 10, limits=Limits(max_nodes=5)).outcome == "deny"
    assert check_privacy(2**100).outcome == "deny"


def test_unknown_objects_are_not_rendered():
    class Unrenderable:
        def __repr__(self):
            raise AssertionError("must not render arbitrary objects")

    assert check_privacy(Unrenderable()).outcome == "deny"

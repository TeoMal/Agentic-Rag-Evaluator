from dataclasses import replace
from datetime import UTC, datetime

import pytest

from hackathon2.guardrails import Limits, Reason, authorize_tool_call


def test_read_only_allowlist_and_no_input_mutation(caller, rules):
    args = {"vendor_id": "vendor-1"}
    assert authorize_tool_call("get_vendor_history", args, context=caller, rules=rules).outcome == "allow"
    assert args == {"vendor_id": "vendor-1"}


@pytest.mark.parametrize("tool", ["shell", "write_file", "task", "unknown", "record_assessment_extra"])
def test_unregistered_actions_deny(tool, caller, rules):
    assert authorize_tool_call(tool, {}, context=caller, rules=rules).reasons == (Reason.UNKNOWN_TOOL,)


@pytest.mark.parametrize(
    "args, reason",
    [
        ({}, Reason.INVALID_ARGUMENTS),
        ({"vendor_id": "vendor-1", "extra": True}, Reason.INVALID_ARGUMENTS),
        ({"vendor_id": "vendor-2"}, Reason.SCOPE_MISMATCH),
        ({"vendor_id": ["vendor-1"]}, Reason.SCOPE_MISMATCH),
        (None, Reason.INVALID_ARGUMENTS),
    ],
)
def test_arguments_and_resource_scope(args, reason, caller, rules):
    result = authorize_tool_call("get_vendor_history", args, context=caller, rules=rules)
    assert result.reasons == (reason,)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"principal_id": ""}, Reason.MISSING_IDENTITY),
        ({"run_id": " "}, Reason.MISSING_IDENTITY),
        ({"permissions": frozenset()}, Reason.PERMISSION_DENIED),
        ({"allowed_resources": {}}, Reason.SCOPE_MISMATCH),
        ({"permissions": "vendor:read"}, Reason.INVALID_INPUT),
    ],
)
def test_missing_or_invalid_trusted_context(change, reason, caller, rules):
    result = authorize_tool_call(
        "get_vendor_history",
        {"vendor_id": "vendor-1"},
        context=replace(caller, **change),
        rules=rules,
    )
    assert result.reasons == (reason,)


def test_missing_identity_denies(caller, rules):
    result = authorize_tool_call("get_vendor_history", {"vendor_id": "vendor-1"}, context=None, rules=rules)
    assert result.reasons == (Reason.MISSING_IDENTITY,)


def test_recording_always_requires_approval_even_if_mislabelled(caller, rules):
    rules["record_assessment"] = replace(rules["record_assessment"], side_effect=False)
    result = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"human_approval": "approved"}},
        context=caller,
        rules=rules,
    )
    assert result.outcome == "require_review"
    assert result.reasons == (Reason.APPROVAL_REQUIRED,)


@pytest.mark.parametrize("verified", [False, None, 1, "approved", {"approved": True}])
def test_only_exact_verifier_true_allows(caller, rules, verified):
    class Verifier:
        def verify(self, **kwargs):
            return verified

    result = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"version": 1}},
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    assert result.reasons == (Reason.APPROVAL_INVALID,)


def test_verifier_receives_bound_action_and_detached_arguments(caller, rules):
    args = {"vendor_id": "vendor-1", "assessment": {"version": 1}}

    class Verifier:
        def verify(self, *, tool_name, arguments, context):
            assert tool_name == "record_assessment"
            assert context.run_id == "run-1" and context.principal_id == "fake-agent"
            assert arguments == args and arguments is not args
            assert arguments["assessment"] is not args["assessment"]
            with pytest.raises(TypeError):
                context.allowed_resources["vendor_id"] = frozenset({"vendor-2"})
            return True

    assert (
        authorize_tool_call(
            "record_assessment",
            args,
            context=caller,
            rules=rules,
            verifier=Verifier(),
        ).outcome
        == "allow"
    )


def test_verifier_exception_is_sanitized(caller, rules, caplog, capsys):
    class Verifier:
        def verify(self, **kwargs):
            raise RuntimeError("SYNTHETIC_SECRET_exception_fixture")

    result = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"version": 1}},
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    assert result.reasons == (Reason.VERIFIER_FAILURE,)
    assert "SYNTHETIC_" not in repr(result) + caplog.text + capsys.readouterr().out


def test_checker_exception_and_mutation_deny_without_mutating_caller(caller, rules):
    args = {"vendor_id": "vendor-1"}

    def broken(arguments):
        raise ValueError("SYNTHETIC_SECRET_exception_fixture")

    rules["get_vendor_history"] = replace(rules["get_vendor_history"], check_arguments=broken)
    assert authorize_tool_call(
        "get_vendor_history",
        args,
        context=caller,
        rules=rules,
    ).reasons == (Reason.CHECKER_FAILURE,)

    def mutating(arguments):
        arguments["vendor_id"] = "vendor-2"
        return True

    rules["get_vendor_history"] = replace(rules["get_vendor_history"], check_arguments=mutating)
    assert authorize_tool_call("get_vendor_history", args, context=caller, rules=rules).outcome == "deny"
    assert args == {"vendor_id": "vendor-1"}


def test_mutating_verifier_cannot_change_execution_arguments(caller, rules):
    args = {"vendor_id": "vendor-1", "assessment": {"version": 1}}

    class Verifier:
        def verify(self, *, arguments, **kw):
            arguments["assessment"]["version"] = 2
            return True

    result = authorize_tool_call("record_assessment", args, context=caller, rules=rules, verifier=Verifier())
    assert result.reasons == (Reason.APPROVAL_INVALID,)
    assert args["assessment"]["version"] == 1


def test_privacy_and_argument_limits_apply_before_verification(caller, rules):
    class MustNotCall:
        def verify(self, **kwargs):
            pytest.fail("verification must not run on invalid payload")

    result = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"quote": "SYNTHETIC_SECRET_fixture"}},
        context=caller,
        rules=rules,
        verifier=MustNotCall(),
    )
    assert result.reasons == (Reason.SENSITIVE_DATA,)
    assert authorize_tool_call(
        "get_vendor_history",
        {"vendor_id": "x" * 100},
        context=caller,
        rules=rules,
        limits=Limits(max_text_chars=50),
    ).reasons == (Reason.LIMIT_EXCEEDED,)


def test_argument_checker_truthy_is_not_sufficient(caller, rules):
    rules["get_vendor_history"] = replace(rules["get_vendor_history"], check_arguments=lambda args: 1)
    assert authorize_tool_call(
        "get_vendor_history",
        {"vendor_id": "vendor-1"},
        context=caller,
        rules=rules,
    ).reasons == (Reason.INVALID_ARGUMENTS,)


@pytest.mark.parametrize("unsupported", [(1,), {"nested": [(1,)]}, datetime(2026, 1, 1, tzinfo=UTC)])
def test_tool_payload_rejects_normalization_before_callbacks(caller, rules, unsupported):
    called = []

    def checker(arguments):
        called.append("checker")
        return True

    class Verifier:
        def verify(self, **kwargs):
            called.append("verifier")
            return True

    rules["record_assessment"] = replace(rules["record_assessment"], check_arguments=checker)
    arguments = {"vendor_id": "vendor-1", "assessment": {"value": unsupported}}
    decision = authorize_tool_call(
        "record_assessment",
        arguments,
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    assert decision.reasons == (Reason.INVALID_ARGUMENTS,)
    assert called == []
    assert arguments["assessment"]["value"] is unsupported


def test_tuple_cannot_satisfy_a_list_only_checker(caller, rules):
    observed = []

    def checker(arguments):
        value = arguments["assessment"]["items"]
        observed.append(type(value))
        return type(value) is list

    rules["record_assessment"] = replace(rules["record_assessment"], check_arguments=checker)
    decision = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"items": (1,)}},
        context=caller,
        rules=rules,
    )
    assert decision.reasons == (Reason.INVALID_ARGUMENTS,)
    assert observed == []
    decision = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"items": [1]}},
        context=caller,
        rules=rules,
    )
    assert observed == [list]
    assert decision.reasons == (Reason.APPROVAL_REQUIRED,)


@pytest.mark.parametrize("stage", ["checker", "verifier"])
@pytest.mark.parametrize(
    "before, after",
    [
        (1, True),
        (True, 1),
        (1, 1.0),
        (1.0, 1),
        (True, 1.0),
        (1.0, True),
        (0, False),
        (False, 0.0),
        (0.0, -0.0),
        ([1], (1,)),
        ([1], [True]),
        ({"flag": 1}, {"flag": 1.0}),
    ],
)
def test_nested_callback_mutations_preserve_exact_types_and_values(caller, rules, stage, before, after):
    effects = []
    verification_calls = []
    arguments = {"vendor_id": "vendor-1", "assessment": {"nested": [{"value": before}]}}

    def checker(checked):
        if stage == "checker":
            checked["assessment"]["nested"][0]["value"] = after
        return True

    class Verifier:
        def verify(self, *, arguments, **kwargs):
            verification_calls.append("verify")
            if stage == "verifier":
                arguments["assessment"]["nested"][0]["value"] = after
            return True

    rules["record_assessment"] = replace(rules["record_assessment"], check_arguments=checker)
    decision = authorize_tool_call(
        "record_assessment",
        arguments,
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    if decision.outcome == "allow":
        effects.append("execute")
    expected = Reason.INVALID_ARGUMENTS if stage == "checker" else Reason.APPROVAL_INVALID
    assert decision.reasons == (expected,)
    assert effects == []
    assert verification_calls == ([] if stage == "checker" else ["verify"])
    assert arguments["assessment"]["nested"][0]["value"] is before


@pytest.mark.parametrize("value", [True, 1, 1.0, False, 0, -0.0, [True, 1, 1.0]])
def test_supported_payload_types_reach_both_callbacks_unchanged(caller, rules, value):
    observed = []

    def inspect(checked):
        actual = checked["assessment"]["nested"][0]
        observed.append(type(actual))
        assert type(actual) is type(value)
        if type(value) is float:
            assert actual.hex() == value.hex()
        elif type(value) is list:
            assert [type(v) for v in actual] == [bool, int, float]
        return True

    class Verifier:
        def verify(self, *, arguments, **kwargs):
            return inspect(arguments)

    rules["record_assessment"] = replace(rules["record_assessment"], check_arguments=inspect)
    decision = authorize_tool_call(
        "record_assessment",
        {"vendor_id": "vendor-1", "assessment": {"nested": [value]}},
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    assert decision.outcome == "allow"
    assert observed == [type(value), type(value)]


@pytest.mark.parametrize("stage", ["checker", "verifier"])
def test_cyclic_callback_mutation_is_bounded_and_denied(caller, rules, stage):
    def checker(arguments):
        if stage == "checker":
            arguments["assessment"]["cycle"] = arguments
        return True

    class Verifier:
        def verify(self, *, arguments, **kwargs):
            if stage == "verifier":
                arguments["assessment"]["cycle"] = arguments
            return True

    rules["record_assessment"] = replace(rules["record_assessment"], check_arguments=checker)
    arguments = {"vendor_id": "vendor-1", "assessment": {"version": 1}}
    decision = authorize_tool_call(
        "record_assessment",
        arguments,
        context=caller,
        rules=rules,
        verifier=Verifier(),
    )
    assert decision.reasons == ((Reason.INVALID_ARGUMENTS if stage == "checker" else Reason.APPROVAL_INVALID),)
    assert arguments == {"vendor_id": "vendor-1", "assessment": {"version": 1}}

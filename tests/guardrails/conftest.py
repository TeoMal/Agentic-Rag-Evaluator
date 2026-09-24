"""Synthetic data only; no Settings, service, model, network or database calls."""

import pytest

from hackathon2.guardrails import CallerContext, GateContext, ToolRule
from hackathon2.schemas import AssessmentDraft, DomainReport, Finding, RequirementControl, SearchHit


@pytest.fixture
def hit():
    return SearchHit(
        chunk_id="vendor#c1",
        doc_id="vendor",
        source="vendor.pdf",
        doc_type="vendor_claim",
        domain="security",
        section="Encryption",
        page=2,
        text="Customer data is encrypted at rest.",
    )


@pytest.fixture
def draft(hit):
    return AssessmentDraft(
        recommendation="REJECT",
        risk_rating="low",
        executive_summary="Outside the procurement requirements.",
        domains=[
            DomainReport(
                domain="security",
                risk_rating="low",
                summary="Encryption documented.",
                findings=[
                    Finding(
                        domain="security",
                        control_id="SEC-01",
                        title="Encryption",
                        status="SUPPORTED",
                        severity="low",
                        claim="Customer data is encrypted at rest.",
                        citations=[hit.to_evidence()],
                    )
                ],
            )
        ],
    )


@pytest.fixture
def gate_context(hit):
    return GateContext(
        run_id="run-1",
        ledger_run_id="run-1",
        ledger={hit.chunk_id: hit},
        required_controls=(
            RequirementControl(
                id="SEC-01",
                domain="security",
                control="Encryption",
                source_chunk_id="policy#c1",
            ),
        ),
    )


@pytest.fixture
def caller():
    return CallerContext(
        principal_id="fake-agent",
        run_id="run-1",
        permissions=frozenset({"vendor:read", "assessment:write"}),
        allowed_resources={"vendor_id": frozenset({"vendor-1"})},
    )


@pytest.fixture
def rules():
    return {
        "get_vendor_history": ToolRule(
            required_permissions=frozenset({"vendor:read"}),
            required_arguments=frozenset({"vendor_id"}),
            allowed_arguments=frozenset({"vendor_id"}),
            scope_arguments=frozenset({"vendor_id"}),
            check_arguments=lambda args: type(args["vendor_id"]) is str,
            side_effect=False,
        ),
        "record_assessment": ToolRule(
            required_permissions=frozenset({"assessment:write"}),
            required_arguments=frozenset({"vendor_id", "assessment"}),
            allowed_arguments=frozenset({"vendor_id", "assessment"}),
            scope_arguments=frozenset({"vendor_id"}),
            check_arguments=lambda args: type(args["assessment"]) is dict and bool(args["assessment"]),
        ),
    }

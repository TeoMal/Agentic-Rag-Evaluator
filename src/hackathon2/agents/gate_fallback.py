"""The decision gate belongs to the guardrails package. This is a stand-in until it exists.

Agreed signature (the runner calls it after the agents finish):

    apply_gate(draft: AssessmentDraft, request: AssessmentRequest,
               retrieved_chunk_ids: set[str], degraded: bool) -> Assessment

`resolve_gate()` uses `hackathon2.guardrails.apply_gate` as soon as the guardrails package exports
it; until then it falls back to `provisional_gate` below. No code change is needed at merge time.
"""

import logging
from collections.abc import Callable

from hackathon2.schemas import Assessment, AssessmentDraft, AssessmentRequest

logger = logging.getLogger(__name__)

GateFn = Callable[[AssessmentDraft, AssessmentRequest, set[str], bool], Assessment]


def provisional_gate(
    draft: AssessmentDraft,
    request: AssessmentRequest,
    retrieved_chunk_ids: set[str],
    degraded: bool,
) -> Assessment:
    """Minimal and conservative. Does NOT check citations -- that is the real gate's job."""
    assessment = Assessment.from_draft(draft, request)
    assessment.degraded_mode = degraded
    needs_human = draft.risk_rating in {"high", "critical"} or draft.recommendation != "REJECT" or degraded
    assessment.human_approval = "pending" if needs_human else "not_required"
    assessment.gate_notes.append(
        "Provisional gate from the agents package: citation and consistency checks were not applied."
    )
    return assessment


def resolve_gate() -> GateFn:
    try:
        from hackathon2.guardrails import apply_gate
    except ImportError:
        logger.warning("hackathon2.guardrails.apply_gate not found; using the provisional gate")
        return provisional_gate
    return apply_gate

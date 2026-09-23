"""Concise executive vendor assessment report (handout section 2), rendered as Markdown."""

from hackathon2.schemas import Assessment, Evidence, Finding

_RECOMMENDATION_LABEL = {
    "APPROVE": "APPROVE",
    "CONDITIONAL_APPROVAL": "CONDITIONAL APPROVAL",
    "REJECT": "REJECT",
}


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _citation(evidence: Evidence) -> str:
    where = evidence.source
    if evidence.section:
        where += f", {evidence.section}"
    if evidence.page is not None:
        where += f", p. {evidence.page}"
    return f"{where} (`{evidence.chunk_id}`)"


def _finding_row(finding: Finding) -> str:
    citations = "<br>".join(_citation(c) for c in finding.citations) or "none"
    return (
        f"| {finding.control_id} | {finding.status} | {finding.severity} | {_cell(finding.claim)} | {citations} |"
    )


def render_markdown(assessment: Assessment) -> str:
    lines = [
        f"# Vendor assessment: {assessment.vendor_name}",
        "",
        f"- **Recommendation:** {_RECOMMENDATION_LABEL[assessment.recommendation]}",
        f"- **Overall risk:** {assessment.risk_rating}",
        f"- **Human approval:** {assessment.human_approval}"
        + (f" by {assessment.reviewer}" if assessment.reviewer else ""),
        f"- **Assessment id:** `{assessment.assessment_id}`",
    ]
    if assessment.degraded_mode:
        lines.append("- **Degraded mode:** a tool or source was unavailable; some controls could not be checked.")

    lines += ["", "## Executive summary", "", assessment.executive_summary, ""]

    gaps = assessment.evidence_gaps
    lines += ["## Missing or contradictory evidence", ""]
    if gaps:
        lines += [f"- **{f.control_id}** ({f.status}, {f.severity}): {_cell(f.claim)}" for f in gaps]
    else:
        lines.append("- None identified.")
    lines.append("")

    lines += ["## Findings by domain", ""]
    for report in assessment.domains:
        lines += [
            f"### {report.domain} - risk {report.risk_rating}",
            "",
            report.summary,
            "",
            "| Control | Status | Severity | Finding | Evidence |",
            "|---|---|---|---|---|",
            *(_finding_row(f) for f in report.findings),
            "",
        ]

    lines += ["## Required conditions", ""]
    if assessment.conditions:
        for condition in assessment.conditions:
            timing = "before go-live" if condition.before_go_live else "after go-live"
            controls = f" [{', '.join(condition.control_ids)}]" if condition.control_ids else ""
            lines.append(f"- ({condition.kind}, {timing}){controls} {_cell(condition.text)}")
    else:
        lines.append("- None.")

    if assessment.gate_notes:
        lines += ["", "## Decision gate notes", "", *(f"- {note}" for note in assessment.gate_notes)]
    if assessment.reviewer_comment:
        lines += ["", "## Reviewer comment", "", assessment.reviewer_comment]
    return "\n".join(lines) + "\n"

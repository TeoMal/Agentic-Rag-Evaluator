"""`uv run python -m hackathon2.agents` -- one end-to-end assessment of Asteria AI Systems.

Uses the real Azure OpenAI model (.env) and the tools selected by AGENT_TOOL_SOURCE (stub by default).
Prints the executive report and the run metrics; exits 1 if the run failed.
"""

import asyncio
import logging
import sys

from hackathon2.agents.report import render_markdown
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.schemas import AssessmentRequest

ASTERIA_REQUEST = AssessmentRequest(
    vendor_name="Asteria AI Systems",
    use_case="Enterprise Generative AI platform",
    user_count=2000,
    data_classification="confidential",
    contract_years=3,
    requested_by="procurement@northstar.example",
    notes="The platform may process confidential corporate documents.",
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    response = asyncio.run(AssessmentRunner().run(ASTERIA_REQUEST))
    if response.assessment is not None:
        print(render_markdown(response.assessment))
    print(f"status: {response.status}")
    if response.error:
        print(f"error: {response.error}")
    if response.metrics is not None:
        print(response.metrics.model_dump_json(indent=2))
    return 1 if response.status == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())

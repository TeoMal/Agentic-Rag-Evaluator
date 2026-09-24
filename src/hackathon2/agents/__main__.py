"""`uv run python -m hackathon2.agents [asteria|corvid]` -- one end-to-end vendor assessment.

    asteria   the handout's request (default)
    corvid    an INVENTED second vendor that exists only in the stub tools. Use it to check that the
              agents generalise: it must be assessed without changing any prompt.

Uses the real Azure OpenAI model (.env) and the tools selected by AGENT_TOOL_SOURCE (the MCP server by
default); corvid always runs on the stub tools, where it exists. Prints the executive report and the run
metrics; exits 1 if the run failed.
"""

import asyncio
import logging
import sys

from hackathon2.agents.report import render_markdown
from hackathon2.agents.runner import AssessmentRunner
from hackathon2.agents.tools import stub_provider
from hackathon2.schemas import AssessmentRequest

REQUESTS: dict[str, AssessmentRequest] = {
    "asteria": AssessmentRequest(
        vendor_name="Asteria AI Systems",
        use_case="Enterprise Generative AI platform",
        user_count=2000,
        data_classification="confidential",
        contract_years=3,
        requested_by="procurement@northstar.example",
        notes="The platform may process confidential corporate documents.",
    ),
    "corvid": AssessmentRequest(
        vendor_name="Corvid Document AI",
        use_case="AI document summarisation and search for legal and compliance teams",
        user_count=400,
        data_classification="confidential",
        contract_years=2,
        requested_by="legal-ops@northstar.example",
        notes="Contracts and regulatory correspondence would be summarised.",
    ),
}


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "asteria"
    if name not in REQUESTS:
        print(f"unknown vendor '{name}'; choose one of: {', '.join(REQUESTS)}")
        return 2
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runner = AssessmentRunner(tools_provider=stub_provider()) if name == "corvid" else AssessmentRunner()
    response = asyncio.run(runner.run(REQUESTS[name]))
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

"""Instrumented tools log retrievals and turn failures into ToolResults instead of exceptions."""

from langchain_core.tools import StructuredTool

from hackathon2.agents.context import RunContext, instrument_tool
from hackathon2.agents.stub_tools import build_stub_tools
from hackathon2.schemas import AssessmentRequest, ToolResult

REQUEST = AssessmentRequest(
    vendor_name="Asteria AI Systems",
    use_case="Enterprise Generative AI platform",
    user_count=2000,
    data_classification="confidential",
)


def _instrumented(ctx: RunContext, **kwargs):
    return {tool.name: instrument_tool(tool, ctx) for tool in build_stub_tools(**kwargs)}


async def test_retrieved_chunk_ids_are_logged():
    ctx = RunContext(request=REQUEST)
    tools = _instrumented(ctx)
    await tools["search_policy"].ainvoke({"query": "encryption at rest", "domain": "security"})
    assert "information-security-policy#s3#c1" in ctx.retrieved_chunk_ids
    assert ctx.tools_called == ["search_policy"]
    assert not ctx.degraded


async def test_crashing_tool_becomes_an_unavailable_result():
    def search_policy(query: str) -> str:
        raise ConnectionError("MCP server went away")

    ctx = RunContext(request=REQUEST)
    tool = instrument_tool(StructuredTool.from_function(search_policy, description="search"), ctx)
    output = await tool.ainvoke({"query": "encryption"})
    assert ToolResult.model_validate_json(output).status == "unavailable"
    assert ctx.degraded and ctx.tool_failures[0].startswith("search_policy: unavailable")


async def test_unavailable_backend_marks_run_degraded():
    ctx = RunContext(request=REQUEST)
    tools = _instrumented(ctx, unavailable=["search_vendor_documents"])
    await tools["search_vendor_documents"].ainvoke({"query": "encryption", "vendor_id": "asteria-ai-systems"})
    assert ctx.degraded and not ctx.retrieved_chunk_ids


async def test_denied_call_is_not_degraded():
    ctx = RunContext(request=REQUEST)
    tools = _instrumented(ctx)
    await tools["record_assessment"].ainvoke({"assessment": {"assessment_id": "x"}})
    assert ctx.denied_calls and not ctx.degraded


def test_an_invalid_call_is_recorded_but_does_not_degrade_the_run():
    # e.g. calculate_tco in verified-pricing mode for a vendor without verified pricing
    ctx = RunContext(request=REQUEST)
    ctx.record_tool_result("calculate_tco", ToolResult.fail("error", "no verified pricing -- use explicit mode"))
    assert ctx.tool_failures and not ctx.degraded
    ctx.record_tool_result("search_policy", ToolResult.fail("unavailable", "index down"))
    assert ctx.degraded


async def test_mcp_style_content_blocks_are_understood():
    payload = ToolResult.ok([{"chunk_id": "doc#s1#c1", "text": "..."}]).model_dump_json()

    def search_policy(query: str) -> list:
        return [{"type": "text", "text": payload}]

    ctx = RunContext(request=REQUEST)
    await instrument_tool(StructuredTool.from_function(search_policy, description="search"), ctx).ainvoke(
        {"query": "x"}
    )
    assert ctx.retrieved_chunk_ids == {"doc#s1#c1"}

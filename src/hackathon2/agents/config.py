"""Settings for the agent layer only, read from AGENT_* environment variables (and `.env`).

Kept apart from hackathon2.config so this package can evolve without editing a shared file.
When the team merges, these fields can move into hackathon2.config.Settings unchanged.

    AGENT_TOOL_SOURCE=stub      stub tools (offline, invented test data) -- default until MCP lands
    AGENT_TOOL_SOURCE=mcp       real MCP server at AGENT_MCP_URL
    AGENT_MCP_URL=http://localhost:8030/mcp
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from hackathon2.schemas import Domain

ALL_DOMAINS: tuple[Domain, ...] = ("security", "procurement", "legal", "ai_governance")


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    tool_source: Literal["stub", "mcp"] = "stub"
    mcp_url: str | None = None
    # Upper bound on graph steps for the orchestrator (each subagent run has its own budget).
    recursion_limit: int = 250
    # 0 keeps runs as repeatable as the model allows (evaluation, FR13). None = model default
    # (needed for reasoning models that reject a temperature).
    temperature: float | None = 0.0


@lru_cache
def get_agent_settings() -> AgentSettings:
    return AgentSettings()

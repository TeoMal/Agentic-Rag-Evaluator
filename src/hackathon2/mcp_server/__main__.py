"""`python -m hackathon2.mcp_server` -- start the MCP server on stdio.

This is the command the agent side (client.py) launches as a subprocess, once per assessment run.
Logs go to stderr: stdout belongs to the MCP protocol.
"""

import importlib
import logging
import sys

from hackathon2.mcp_server.server import SERVER_NAME, mcp

# Native-code modules the knowledge tools load. On Windows, loading an extension module lazily --
# inside a tool call, while the stdio transport is blocked reading stdin -- can deadlock the whole
# server (seen with numpy). Importing them before serving avoids it; elsewhere it only moves the
# import cost to start-up.
PRELOAD = ("numpy", "tiktoken", "pypdf", "psycopg", "sqlalchemy", "langchain_openai", "hackathon2.rag")


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("hackathon2.mcp_server")
    for name in PRELOAD:
        try:
            importlib.import_module(name)
        except ImportError as exc:  # optional backends; the tool that needs one reports "unavailable"
            log.warning("preload of %s failed: %s", name, exc)
    log.info("starting %s on stdio", SERVER_NAME)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

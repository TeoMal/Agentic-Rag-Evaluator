"""`python -m hackathon2.mcp_server` -- start the MCP server on stdio.

This is the command the agent side (client.py, step 2) launches as a subprocess.
Logs go to stderr: stdout belongs to the MCP protocol.
"""

import logging
import sys

from hackathon2.mcp_server.server import SERVER_NAME, mcp


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("hackathon2.mcp_server").info("starting %s (stub data) on stdio", SERVER_NAME)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

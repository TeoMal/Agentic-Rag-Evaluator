"""Hackathon 2 -- AI-powered Vendor Risk & Procurement Deep Agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hackathon2")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0"


def main() -> None:
    """`uv run hackathon2` -- serve the API locally with auto-reload."""
    import uvicorn

    uvicorn.run("hackathon2.service:create_app", factory=True, host="127.0.0.1", port=8000, reload=True)

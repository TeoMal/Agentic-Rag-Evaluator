"""Approval tokens for the restricted record_assessment tool.

The problem: record_assessment must only save an assessment that a human (or the
decision gate, for low risk) actually approved. A plain "is the token non-empty"
check can be satisfied by anyone -- including an LLM tricked by a prompt injection.

The solution: an HMAC signature. Our API code -- the human-decision endpoint -- signs
the exact final assessment with a secret key; the MCP server re-computes the signature
and saves only if it matches. Nobody without the key can produce a valid token, and a
token signed for one assessment is useless for any other.

    token = "v1.<expires_unix>.<signature>"
    signature = HMAC-SHA256(key, "v1|<assessment_id>|<content digest>|<expires_unix>")

What the signature binds:
  * assessment_id   -- a token cannot be reused for another assessment;
  * content digest  -- SHA-256 of the whole assessment, so changing anything after
                       approval (recommendation, findings, conditions) invalidates it;
  * expiry          -- a leaked token stops working after TOKEN_TTL_SECONDS.

The key is MCP_APPROVAL_SECRET (from the environment, or .env for local runs). Both
the API process and the MCP server subprocess read it. If it is missing, every
record_assessment call is refused: we fail closed, never open.

    Generate one:  python -c "import secrets; print(secrets.token_hex(32))"

Usage in the human-decision endpoint (tech lead):

    from hackathon2.mcp_server.auth import issue_approval_token

    approved = assessment.model_copy(update={"human_approval": "approved", "reviewer": decision.reviewer})
    token = issue_approval_token(approved)
    await toolbox.call("system", "record_assessment",
                       {"assessment": approved.model_dump(mode="json"), "approval_token": token})
"""

import hashlib
import hmac
import json
import os
import time

from dotenv import dotenv_values

from hackathon2.schemas import Assessment

TOKEN_VERSION = "v1"
TOKEN_TTL_SECONDS = 15 * 60
SECRET_ENV = "MCP_APPROVAL_SECRET"
MIN_SECRET_LENGTH = 32


class ApprovalConfigError(RuntimeError):
    """MCP_APPROVAL_SECRET is missing or too short."""


def _secret() -> bytes:
    value = os.environ.get(SECRET_ENV) or dotenv_values(".env").get(SECRET_ENV) or ""
    if len(value) < MIN_SECRET_LENGTH:
        raise ApprovalConfigError(
            f"{SECRET_ENV} is not set (or shorter than {MIN_SECRET_LENGTH} characters); "
            "record_assessment is disabled until it is configured"
        )
    return value.encode("utf-8")


def assessment_digest(assessment: Assessment) -> str:
    """SHA-256 of the assessment's canonical JSON -- identical on the signing and verifying side."""
    canonical = json.dumps(assessment.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _signature(key: bytes, assessment_id: str, digest: str, expires: int) -> str:
    message = f"{TOKEN_VERSION}|{assessment_id}|{digest}|{expires}".encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def issue_approval_token(assessment: Assessment, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    """Sign a FINAL assessment. Call only from our own code after the approval decision."""
    if assessment.human_approval in ("pending", "rejected"):
        raise ValueError(f"cannot issue a token for an assessment with human_approval='{assessment.human_approval}'")
    expires = int(time.time()) + ttl_seconds
    signature = _signature(_secret(), assessment.assessment_id, assessment_digest(assessment), expires)
    return f"{TOKEN_VERSION}.{expires}.{signature}"


def verify_approval_token(token: str, assessment: Assessment) -> str | None:
    """None if the token is valid for exactly this assessment; otherwise the reason it is not."""
    try:
        version, expires_text, signature = token.strip().split(".")
        expires = int(expires_text)
    except ValueError:
        return "malformed approval_token"
    if version != TOKEN_VERSION:
        return f"unsupported token version '{version}'"
    if expires < time.time():
        return "approval_token has expired"
    expected = _signature(_secret(), assessment.assessment_id, assessment_digest(assessment), expires)
    if not hmac.compare_digest(expected, signature):  # constant-time comparison
        return "approval_token does not match this assessment (not signed for it, or content changed)"
    return None

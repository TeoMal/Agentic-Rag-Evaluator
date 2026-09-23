"""Backend for the enterprise tools -- NFS's internal records.

The knowledge pack contains policies and the vendor's documents, but not NFS's own
systems (vendor history, budgets, earlier decisions). Those are fictional records we
maintain in data/*.json, kept consistent with the PDFs. The one exception is pricing:
every number in data/pricing.json is copied from knowledge/vendor-x-pricing.pdf.

Each file is validated with Pydantic when first used, so a typo in the JSON fails
loudly (and the tool returns "unavailable") instead of producing wrong numbers.
Keys starting with "_" are notes for humans and are ignored.

Every record carries a `record_id`, so agents can cite it as evidence
(Evidence.doc_type = "enterprise_record", Evidence.chunk_id = record_id).

Assessments saved by record_assessment are appended to a JSON-lines file in
MCP_STATE_DIR (default: the system temp folder). On Azure that folder is not
durable -- fine for the demo; the register survives for the lifetime of the replica.
"""

import json
import os
import tempfile
import threading
from datetime import date
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from hackathon2.schemas import Assessment, Recommendation, Severity, TCOResult

DEFAULT_CATEGORY = "generative-ai-platform"

# --------------------------------------------------------------------------------------
# Record shapes (internal to this module; tools return them as dicts)
# --------------------------------------------------------------------------------------


class VendorEvent(BaseModel):
    record_id: str
    date: date
    type: Literal["engagement", "security_incident", "remediation", "contract", "sla_breach", "audit"]
    severity: Severity
    description: str


class PriceTier(BaseModel):
    min_seats: int = Field(ge=1)
    per_seat_monthly: float = Field(ge=0)


class Pricing(BaseModel):
    currency: str = "EUR"
    tiers: list[PriceTier] = Field(min_length=1)
    platform_fee_yearly: float = Field(default=0.0, ge=0)
    setup_fee: float = Field(default=0.0, ge=0)
    annual_increase_pct: float = Field(default=0.0, ge=0, le=50)
    source_chunk_id: str

    @model_validator(mode="after")
    def _sort_tiers(self) -> "Pricing":
        self.tiers = sorted(self.tiers, key=lambda t: t.min_seats)
        return self

    def seat_price(self, seats: int) -> float:
        """Price of the highest tier the seat count reaches (volume pricing)."""
        eligible = [t for t in self.tiers if seats >= t.min_seats]
        return (eligible[-1] if eligible else self.tiers[0]).per_seat_monthly


class Budget(BaseModel):
    record_id: str
    category: str
    currency: str = "EUR"
    period_years: int = Field(ge=1)
    budget_total: float = Field(ge=0)
    committed: float = Field(default=0.0, ge=0)
    approver_above_budget: str
    owner: str | None = None


class PriorAssessment(BaseModel):
    assessment_id: str
    vendor_id: str
    category: str
    date: date
    recommendation: Recommendation
    risk_rating: Severity
    human_approval: str = "not_required"
    reviewer: str | None = None
    conditions: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read(name: str) -> dict:
    text = resources.files("hackathon2.mcp_server").joinpath(f"data/{name}").read_text("utf-8")
    return {k: v for k, v in json.loads(text).items() if not k.startswith("_")}


@lru_cache
def _vendor_history() -> dict[str, tuple[VendorEvent, ...]]:
    raw = _read("vendor_history.json")["vendors"]
    return {
        vid: tuple(sorted((VendorEvent.model_validate(e) for e in events), key=lambda e: e.date))
        for vid, events in raw.items()
    }


@lru_cache
def _pricing() -> dict[str, Pricing]:
    return {vid: Pricing.model_validate(p) for vid, p in _read("pricing.json")["vendors"].items()}


@lru_cache
def _budgets() -> dict[str, Budget]:
    return {cat: Budget.model_validate(b) for cat, b in _read("budgets.json")["budgets"].items()}


@lru_cache
def _seed_assessments() -> tuple[PriorAssessment, ...]:
    return tuple(PriorAssessment.model_validate(a) for a in _read("prior_assessments.json")["assessments"])


def reload_data() -> None:
    """Forget cached data (tests, or after editing the JSON while the server runs)."""
    for loader in (_vendor_history, _pricing, _budgets, _seed_assessments):
        loader.cache_clear()


# --------------------------------------------------------------------------------------
# Assessment register (record_assessment)
# --------------------------------------------------------------------------------------

_register_lock = threading.Lock()


def register_path() -> Path:
    state_dir = Path(os.environ.get("MCP_STATE_DIR") or Path(tempfile.gettempdir()) / "hackathon2")
    return state_dir / "assessments.jsonl"


def _recorded() -> list[PriorAssessment]:
    path = register_path()
    if not path.exists():
        return []
    with _register_lock:
        lines = path.read_text("utf-8").splitlines()
    return [PriorAssessment.model_validate_json(line) for line in lines if line.strip()]


# --------------------------------------------------------------------------------------
# The functions server.py calls -- signatures unchanged since the stub.
# --------------------------------------------------------------------------------------


def vendor_history(vendor_id: str) -> list[dict]:
    """Past engagements and incidents, oldest first. Empty list = no history on record."""
    return [e.model_dump(mode="json") for e in _vendor_history().get(vendor_id, ())]


def tco(vendor_id: str, seats: int, years: int) -> TCOResult | None:
    """Deterministic total cost of ownership. None when there is no pricing for the vendor.

    per year y (0-based):  seats x tier price x 12 x (1 + increase)^y  +  platform fee x (1 + increase)^y
    plus a one-off setup fee.
    """
    if seats <= 0 or not 1 <= years <= 10:
        raise ValueError("seats must be > 0 and years between 1 and 10")
    price = _pricing().get(vendor_id)
    if price is None:
        return None

    growth = [(1 + price.annual_increase_pct / 100) ** y for y in range(years)]
    seat_price = price.seat_price(seats)
    seats_total = sum(seats * seat_price * 12 * g for g in growth)
    platform_total = sum(price.platform_fee_yearly * g for g in growth)
    total = seats_total + platform_total + price.setup_fee

    return TCOResult(
        vendor_id=vendor_id,
        seats=seats,
        years=years,
        currency=price.currency,
        total=round(total, 2),
        breakdown={
            "per_seat_monthly_applied": seat_price,
            "seat_licences": round(seats_total, 2),
            "platform_fees": round(platform_total, 2),
            "setup": round(price.setup_fee, 2),
            "per_year_average": round(total / years, 2),
        },
        source_chunk_id=price.source_chunk_id,
    )


def budget(category: str) -> dict | None:
    """Approved budget for a spend category, with what is still available. None if unknown."""
    record = _budgets().get(category)
    if record is None:
        return None
    return record.model_dump(mode="json") | {"remaining": round(record.budget_total - record.committed, 2)}


def prior_assessments(vendor_id: str | None = None, category: str | None = None) -> list[dict]:
    """Earlier decisions (seed records + everything recorded since), oldest first."""
    records = sorted([*_seed_assessments(), *_recorded()], key=lambda a: a.date)
    return [
        a.model_dump(mode="json")
        for a in records
        if (vendor_id is None or a.vendor_id == vendor_id) and (category is None or a.category == category)
    ]


def is_recorded(assessment_id: str) -> bool:
    """True if this assessment is already in the register (a token is single-use in effect)."""
    return any(a.assessment_id == assessment_id for a in _recorded())


def save_assessment(assessment: Assessment, category: str = DEFAULT_CATEGORY) -> dict:
    """Append a final assessment to the register and return the stored record."""
    record = PriorAssessment(
        assessment_id=assessment.assessment_id,
        vendor_id=assessment.vendor_id,
        category=category,
        date=assessment.created_at.date(),
        recommendation=assessment.recommendation,
        risk_rating=assessment.risk_rating,
        human_approval=assessment.human_approval,
        reviewer=assessment.reviewer,
        conditions=[c.text for c in assessment.conditions],
    )
    path = register_path()
    with _register_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")
    return record.model_dump(mode="json")
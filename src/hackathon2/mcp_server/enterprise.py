"""Backend for the enterprise tools -- NFS's internal records and deterministic calculations.

Everything here comes from the knowledge pack, never from invention:
  * pricing.json           <- vendor-x-pricing.pdf (verified; self-checked against the PDF's totals)
  * prior_assessments.json <- historical-vendor-assessments/*.pdf
  * procurement_rules.json <- procurement-policy.pdf (PR-001) approval thresholds
There is no budget and no vendor incident log in the knowledge pack, so none is invented:
a vendor without prior assessments simply has no history on record.

Vendor-agnostic (hidden vendor case): calculate_tco also accepts the prices explicitly,
as found by the agent in the vendor's own pricing document, and only does the arithmetic.

Each file is validated with Pydantic when first used, so a typo in the JSON fails loudly
(the tool returns "unavailable") instead of producing wrong numbers. Keys starting with
"_" are notes for humans and are ignored.

Assessments saved by record_assessment are appended to a JSON-lines file in
MCP_STATE_DIR (default: the system temp folder). In Docker, mount a volume there if the
register should survive a container restart.
"""

import json
import os
import re
import tempfile
import threading
from datetime import date as Date
from functools import lru_cache
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from hackathon2.schemas import Assessment, Recommendation, Severity, TCOResult

DEFAULT_CATEGORY = "generative-ai-platform"

# --------------------------------------------------------------------------------------
# Record shapes (internal to this module; tools return them as dicts)
# --------------------------------------------------------------------------------------


class PerUserAddOn(BaseModel):
    per_user_monthly: float | None = Field(default=None, ge=0)
    annual_fee: float | None = Field(default=None, ge=0)
    description: str = ""
    source_chunk_id: str

    @model_validator(mode="after")
    def _one_price(self) -> "PerUserAddOn":
        if (self.per_user_monthly is None) == (self.annual_fee is None):
            raise ValueError("an add-on has exactly one of per_user_monthly / annual_fee")
        return self


class OneTimeFee(BaseModel):
    amount: float = Field(ge=0)
    source_chunk_id: str


class PrepaidDiscount(BaseModel):
    pct: float = Field(ge=0, le=50)
    min_months: int = Field(ge=1)
    applies_to: str = "base_subscription"
    source_chunk_id: str
    note: str = ""


class Pricing(BaseModel):
    currency: str = "EUR"
    source_document: str
    per_user_monthly: float = Field(ge=0)
    source_chunk_id: str
    commitment_months: int = 12
    add_ons: dict[str, PerUserAddOn] = Field(default_factory=dict)
    one_time_fees: dict[str, OneTimeFee] = Field(default_factory=dict)
    prepaid_discount: PrepaidDiscount | None = None
    stated_totals: dict[str, float] = Field(default_factory=dict)


class PriorAssessment(BaseModel):
    assessment_id: str
    vendor_id: str
    vendor_name: str | None = None
    category: str
    year: int | None = None
    date: Date | None = None
    recommendation: Recommendation
    risk_rating: Severity
    human_approval: str | None = None  # not stated in the historical PDFs
    reviewer: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    lesson_learned: str | None = None
    source_document: str | None = None

    @model_validator(mode="after")
    def _year_from_date(self) -> "PriorAssessment":
        if self.year is None:
            if self.date is None:
                raise ValueError("a prior assessment needs a year or a date")
            self.year = self.date.year
        return self


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _read(name: str) -> dict:
    text = resources.files("hackathon2.mcp_server").joinpath(f"data/{name}").read_text("utf-8")
    return {k: v for k, v in json.loads(text).items() if not k.startswith("_")}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@lru_cache
def _pricing() -> dict[str, Pricing]:
    register = {vid: Pricing.model_validate(p) for vid, p in _read("pricing.json")["vendors"].items()}
    for vendor_id, price in register.items():
        _check_stated_totals(vendor_id, price)
    return register


def _check_stated_totals(vendor_id: str, price: Pricing) -> None:
    """Recompute the year-one totals the pricing PDF states; refuse to load on a mismatch
    (catches typos when copying numbers from the PDF)."""
    stated = price.stated_totals
    users = int(stated.get("users", 0))
    if not users:
        return
    one_time = sum(f.amount for f in price.one_time_fees.values())
    base = price.per_user_monthly * users * 12 + one_time
    checks = {"year_one_base_with_implementation": base}
    if "enterprise_plus" in price.add_ons and price.add_ons["enterprise_plus"].per_user_monthly is not None:
        checks["year_one_enterprise_plus_with_implementation"] = (
            base + price.add_ons["enterprise_plus"].per_user_monthly * users * 12
        )
    for key, computed in checks.items():
        if key in stated and abs(stated[key] - computed) > 0.5:
            raise ValueError(
                f"pricing.json '{vendor_id}': {key} computes to {computed:,.0f}, PDF states {stated[key]:,.0f}"
            )


@lru_cache
def _seed_assessments() -> tuple[PriorAssessment, ...]:
    return tuple(PriorAssessment.model_validate(a) for a in _read("prior_assessments.json")["assessments"])


@lru_cache
def _procurement_rules() -> dict:
    return _read("procurement_rules.json")


def reload_data() -> None:
    """Forget cached data (after editing the JSON while the server runs)."""
    for loader in (_pricing, _seed_assessments, _procurement_rules):
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
# The functions server.py calls
# --------------------------------------------------------------------------------------


def _pricing_for(vendor_ref: str) -> tuple[str, Pricing] | None:
    register = _pricing()
    wanted = _norm(vendor_ref)
    try:
        from hackathon2.mcp_server.knowledge import resolve_vendor

        wanted = resolve_vendor(vendor_ref)
    except ValueError:
        pass  # not in the vendor registry -- still try the pricing register by id
    return (wanted, register[wanted]) if wanted in register else None


def tco(
    vendor_id: str,
    seats: int,
    years: int,
    add_ons: list[str] | None = None,
    prepaid: bool = False,
    per_user_monthly: float | None = None,
    add_on_per_user_monthly: float = 0.0,
    annual_fees: float = 0.0,
    one_time_fees: float = 0.0,
    discount_pct: float = 0.0,
    source_chunk_id: str | None = None,
) -> TCOResult | None:
    """Deterministic total cost of ownership.

    Register mode (per_user_monthly is None): verified prices from pricing.json; `add_ons`
    selects named options (e.g. ["enterprise_plus", "premium_support"]); `prepaid` applies the
    vendor's prepaid discount when the term is long enough. Returns None if the vendor has
    no verified pricing.

    Explicit mode (per_user_monthly given): any vendor -- the caller passes the prices it
    found in the vendor's pricing document, and source_chunk_id to cite them.
    """
    if seats <= 0 or not 1 <= years <= 10:
        raise ValueError("seats must be > 0 and years between 1 and 10")
    months = years * 12
    add_ons = list(add_ons or [])
    assumptions = [f"{seats} named users for {years} year(s) ({months} months)."]

    if per_user_monthly is None:
        found = _pricing_for(vendor_id)
        if found is None:
            return None
        vendor_id, price = found
        unknown = [a for a in add_ons if a not in price.add_ons]
        if unknown:
            raise ValueError(f"unknown add-on(s) {unknown}; available for {vendor_id}: {sorted(price.add_ons)}")
        currency, citation = price.currency, price.source_chunk_id
        base_rate = price.per_user_monthly
        extra_rate = sum(price.add_ons[a].per_user_monthly or 0.0 for a in add_ons)
        annual = sum(price.add_ons[a].annual_fee or 0.0 for a in add_ons)
        one_time = sum(f.amount for f in price.one_time_fees.values())
        pct = 0.0
        if prepaid and price.prepaid_discount:
            if months >= price.prepaid_discount.min_months:
                pct = price.prepaid_discount.pct
                assumptions.append(price.prepaid_discount.note or f"{pct}% prepaid discount on the base subscription.")
            else:
                assumptions.append(
                    f"Prepaid discount needs at least {price.prepaid_discount.min_months} months; not applied."
                )
        assumptions.append(f"Verified prices from {price.source_document} (pricing register).")
        not_selected = sorted(set(price.add_ons) - set(add_ons))
        if not_selected:
            assumptions.append(f"Not included (optional): {', '.join(not_selected)}.")
    else:
        if add_ons:
            raise ValueError(
                "in explicit mode pass add-on prices as add_on_per_user_monthly / annual_fees, not add_ons"
            )
        currency, citation = "EUR", source_chunk_id
        base_rate, extra_rate, annual, one_time, pct = (
            per_user_monthly,
            add_on_per_user_monthly,
            annual_fees,
            one_time_fees,
            discount_pct,
        )
        assumptions.append(
            f"Prices supplied by the caller from {source_chunk_id or 'an uncited source -- cite it before relying on it'}."
        )

    base = base_rate * seats * months
    discount = base * pct / 100
    extras = extra_rate * seats * months
    annual_total = annual * years
    total = base - discount + extras + annual_total + one_time
    year_one = (base - discount) / years + extras / years + annual + one_time

    return TCOResult(
        vendor_id=vendor_id,
        seats=seats,
        years=years,
        currency=currency,
        total=round(total, 2),
        breakdown={
            "base_subscription": round(base, 2),
            "prepaid_discount": round(-discount, 2),
            "add_on_subscriptions": round(extras, 2),
            "annual_fees": round(annual_total, 2),
            "one_time_fees": round(one_time, 2),
            "year_one_total": round(year_one, 2),
            "annual_recurring": round((base - discount + extras) / years + annual, 2),
            "per_year_average": round(total / years, 2),
        },
        add_ons=add_ons,
        assumptions=assumptions,
        source_chunk_id=citation,
    )


def approval_requirements(annual_value: float, data_classification: str | None = None, ai_system: bool = True) -> dict:
    """Who must approve a purchase of this annual value (PR-001), with citations."""
    if annual_value < 0:
        raise ValueError("annual_value must be >= 0")
    rules = _procurement_rules()
    band = next(
        t
        for t in rules["approval_thresholds"]
        if t["up_to_annual_value"] is None or annual_value <= t["up_to_annual_value"]
    )
    sourcing = rules["competitive_sourcing"]
    confidential = (data_classification or "").lower() in ("confidential", "restricted")
    applies = {
        "confidential_or_restricted_data": confidential,
        "ai_system": ai_system,
        "ai_system_high_risk": ai_system and confidential,
    }
    extra = [a for a in rules["additional_approvals"] if applies.get(a["applies_if"], False)]
    citations = sorted(
        {band["source_chunk_id"], *(a["source_chunk_id"] for a in extra)}
        | ({sourcing["source_chunk_id"]} if annual_value > sourcing["above_value"] else set())
    )
    return {
        "annual_value": round(annual_value, 2),
        "approvers": band["approvers"],
        "threshold_rule": band["section"],
        "competitive_sourcing_required": annual_value > sourcing["above_value"],
        "competitive_sourcing_rule": sourcing["rule"] if annual_value > sourcing["above_value"] else None,
        "additional_approvals": [{"approval": a["approval"], "rule": a["section"]} for a in extra],
        "citations": citations,
        "note": "AI systems processing Confidential/Restricted data are High risk (AI-004 s2)."
        if applies["ai_system_high_risk"]
        else None,
    }


def prior_assessments(vendor_id: str | None = None, category: str | None = None) -> list[dict]:
    """Earlier decisions (historical records + everything recorded since), oldest first."""
    records = sorted([*_seed_assessments(), *_recorded()], key=lambda a: (a.year, a.date or Date(a.year, 1, 1)))
    return [
        a.model_dump(mode="json", exclude_none=True)
        for a in records
        if (vendor_id is None or _norm(vendor_id) in (_norm(a.vendor_id), _norm(a.vendor_name or "")))
        and (category is None or _norm(a.category) == _norm(category))
    ]


def vendor_history(vendor_id: str) -> list[dict]:
    """This vendor's record with NFS (its past assessments). Empty = no history on record."""
    return prior_assessments(vendor_id=vendor_id)


def is_recorded(assessment_id: str) -> bool:
    """True if this assessment is already in the register (a token is single-use in effect)."""
    return any(a.assessment_id == assessment_id for a in _recorded())


def save_assessment(assessment: Assessment, category: str = DEFAULT_CATEGORY) -> dict:
    """Append a final assessment to the register and return the stored record."""
    record = PriorAssessment(
        assessment_id=assessment.assessment_id,
        vendor_id=assessment.vendor_id,
        vendor_name=assessment.vendor_name,
        category=category,
        year=assessment.created_at.year,
        date=assessment.created_at.date(),
        recommendation=assessment.recommendation,
        risk_rating=assessment.risk_rating,
        human_approval=assessment.human_approval,
        reviewer=assessment.reviewer,
        key_findings=[f.claim for f in assessment.non_compliant][:10],
        conditions=[c.text for c in assessment.conditions],
    )
    path = register_path()
    with _register_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json(exclude_none=True) + "\n")
    return record.model_dump(mode="json", exclude_none=True)

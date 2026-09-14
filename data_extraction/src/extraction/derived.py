"""D-over-L derived variables (spec §10.2 D, F, I). Pure functions on pass-A structures."""

from __future__ import annotations

from typing import Any

SCRIPT_ITEMS = (
    "greets", "identifies_self", "recording_notice", "validates_identity_before_disclosing", "states_amount",
    "states_due_date", "offers_payment_channels", "mentions_credit_bureaus", "recaps_agreement", "polite_closing",
)
FINAL_OUTCOME_ORDER = ("no_contact", "third_party", "holder_refuses", "no_agreement", "callback_scheduled",
                       "claims_already_paid", "vague_promise", "firm_promise")


def ptp_strength(outcome: dict[str, Any]) -> int | None:
    """(ptp_amount is a number) + (ptp_date not null) + ptp_confirmed_by_client; null if ptp is false.

    A censored amount does not count (D-029)."""
    if not outcome.get("ptp"):
        return None
    amount = outcome.get("ptp_amount")
    has_amount = isinstance(amount, (int, float)) and not isinstance(amount, bool)
    return int(has_amount) + int(outcome.get("ptp_date") is not None) + int(outcome.get("ptp_confirmed_by_client") is True)


def script_checklist_score(compliance: dict[str, Any], contact_type: str) -> float | None:
    """% true over non-"na" script items; null unless contact_type == account_holder or no applicable item."""
    if contact_type != "account_holder":
        return None
    values = [compliance[item]["value"] for item in SCRIPT_ITEMS if compliance[item]["value"] != "na"]
    if not values:
        return None
    return round(100 * sum(v is True for v in values) / len(values), 2)


def objection_counts(objections: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(objections)
    yes = sum(o["resolved"] == "yes" for o in objections)
    partial = sum(o["resolved"] == "partial" for o in objections)
    return {"n_objections": n, "n_objections_resolved": yes, "n_objections_partial": partial,
            "pct_objections_resolved": round(100 * yes / n, 2) if n else None}


def final_outcome_rank(value: str | None) -> int | None:
    return FINAL_OUTCOME_ORDER.index(value) if value in FINAL_OUTCOME_ORDER else None

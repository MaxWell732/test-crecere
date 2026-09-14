"""E9 · Interpretive annotation support (spec §9-E9): progress and `annotate-validate`.

Claude Code writes `annotations/{pass_a,pass_b,rerun_pass_a}/<call_id>.json` by reading only
`llm_inputs/calls/<call_id>.txt`, `configs/codebook.md` and the schemas. This module never reads group information.
"""

from __future__ import annotations

import json
from typing import Any

from . import paths
from .annot_utils import codebook_sha256, read_progress, schema_errors, update_progress, verify_quote
from .cache import read_json
from .canonical import canonical_turn_text, load_canonical
from .config import load_config
from .derived import SCRIPT_ITEMS
from .roles import calls_with_turns

PASS_DIRS = {"a": "pass_a", "b": "pass_b", "rerun_a": "rerun_pass_a"}
PASS_SCHEMA = {"a": "pass_a.schema.json", "b": "pass_b.schema.json", "rerun_a": "pass_a.schema.json"}
PTP_FIELDS_NULL = ("ptp_amount", "ptp_date", "ptp_type")
PTP_FIELDS_NA = ("ptp_type_commitment", "ptp_confirmed_by_client")
MILESTONE_ORDER = ("debt_mention_turn_id", "first_offer_turn_id", "ptp_turn_id")  # identification unconstrained (D-012)
NON_HOLDER_FREE_ITEMS = {"greets", "identifies_self"}


def annotation_path(pass_name: str, call_id: str):
    return paths.ANNOTATIONS / PASS_DIRS[pass_name] / f"{call_id}.json"


def annotatable_calls(calls: list[str] | None = None) -> list[str]:
    return [c for c in calls_with_turns(calls) if paths.interim("canonical", f"{c}.json").exists()]


# --------------------------------------------------------------------------- reference collection
def quoted_refs(doc: dict[str, Any]) -> list[tuple[str, str | None, str | None]]:
    """(json-path, turn_id, quote) for every quote-bearing element of a pass-A document."""
    refs: list[tuple[str, str | None, str | None]] = []
    for sec in ("contact", "debt_context", "outcome"):
        for i, ev in enumerate(doc[sec]["evidence"]):
            refs.append((f"{sec}.evidence[{i}]", ev["turn_id"], ev["quote"]))
    for i, o in enumerate(doc["negotiation"]["offers"]):
        refs.append((f"negotiation.offers[{i}]", o["turn_id"], o["quote"]))
    for i, o in enumerate(doc["objections"]):
        refs.append((f"objections[{i}]", o["turn_id"], o["quote"]))
    for item, v in doc["compliance"].items():
        if v["turn_id"] is not None or v["quote"] is not None:
            refs.append((f"compliance.{item}", v["turn_id"], v["quote"]))
    v = doc["failures"]["client_expresses_annoyance"]
    if v["turn_id"] is not None or v["quote"] is not None:
        refs.append(("failures.client_expresses_annoyance", v["turn_id"], v["quote"]))
    return refs


def bare_turn_ids(doc: dict[str, Any], pass_name: str) -> list[tuple[str, str]]:
    if pass_name == "b":
        return [(f"scores.{k}.turn_ids", t) for k, v in doc["scores"].items() for t in v["turn_ids"]]
    ids = [(f"milestones.{k}", v) for k, v in doc["milestones"].items() if v is not None]
    ids += [(f"objections[{i}].response_turn_id", o["response_turn_id"]) for i, o in enumerate(doc["objections"])
            if o["response_turn_id"] is not None]
    for key in ("agent_misunderstood_turn_ids", "incoherent_response_turn_ids"):
        ids += [(f"failures.{key}", t) for t in doc["failures"][key]]
    return ids


# --------------------------------------------------------------------------- consistency rules
def consistency_errors_a(doc: dict[str, Any], turn_index: dict[str, int]) -> list[str]:
    e: list[str] = []
    out, contact, comp = doc["outcome"], doc["contact"], doc["compliance"]
    if out["ptp"] is False:
        bad = [f for f in PTP_FIELDS_NULL if out[f] is not None] + [f for f in PTP_FIELDS_NA if out[f] != "na"]
        if bad:
            e.append(f"rule ptp=false: fields must be null/'na': {bad}")
        if out["final_outcome"] in ("vague_promise", "firm_promise"):
            e.append("rule ptp=false: final_outcome cannot be vague_promise/firm_promise")
        if comp["recaps_agreement"]["value"] != "na":
            e.append("rule ptp=false: compliance.recaps_agreement must be 'na'")
        if out["agent_recapped"] != "na":
            e.append("rule ptp=false: outcome.agent_recapped must be 'na'")
    else:
        expected = {"firm": "firm_promise", "vague": "vague_promise"}.get(out["ptp_type_commitment"])
        if expected is None:
            e.append("rule ptp=true: ptp_type_commitment must be firm or vague")
        elif out["final_outcome"] != expected:
            e.append(f"rule ptp=true: final_outcome must be {expected} for ptp_type_commitment={out['ptp_type_commitment']}")
    if contact["contact_type"] != "account_holder":
        if out["final_outcome"] not in ("no_contact", "third_party"):
            e.append("rule contact_type≠account_holder: final_outcome must be no_contact or third_party")
        if out["ptp"] is not False:
            e.append("rule contact_type≠account_holder: ptp must be false")
        bad = [i for i in SCRIPT_ITEMS if i not in NON_HOLDER_FREE_ITEMS and comp[i]["value"] != "na"]
        if bad:
            e.append(f"rule contact_type≠account_holder: script items must be 'na': {bad}")
        if doc["objections"]:
            e.append("rule contact_type≠account_holder: objections must be empty (codebook v2-4)")
    reasons = doc["debt_context"]["non_payment_reasons"]
    if "not_stated" in reasons and reasons != ["not_stated"]:
        e.append("rule: non_payment_reasons cannot mix 'not_stated' with stated reasons")
    offers = doc["negotiation"]["offers"]
    if not doc["negotiation"]["had_negotiation"] and offers:
        e.append("rule: had_negotiation=false but offers listed")
    if offers:
        first = min(offers, key=lambda o: turn_index.get(o["turn_id"], 10**9))
        if doc["negotiation"]["first_anchor"] != first["proposed_by"]:
            e.append(f"rule: first_anchor must equal proposed_by of the earliest offer ({first['proposed_by']})")
    elif doc["negotiation"]["first_anchor"] != "none":
        e.append("rule: first_anchor must be 'none' when there are no offers")
    ms = [(k, doc["milestones"][k]) for k in MILESTONE_ORDER if doc["milestones"][k] is not None]
    positions = [turn_index.get(v, -1) for _, v in ms]
    if positions != sorted(positions):
        e.append(f"rule: milestones out of chronological order: {ms}")
    flagged = "quote_verification_failed" in doc.get("flags", [])
    for item, v in comp.items():
        if v["value"] is True and (v["turn_id"] is None or v["quote"] is None) and not flagged:
            e.append(f"rule: compliance.{item}=true needs turn_id and quote")
    v = doc["failures"]["client_expresses_annoyance"]
    if v["value"] is True and (v["turn_id"] is None or v["quote"] is None) and not flagged:
        e.append("rule: failures.client_expresses_annoyance=true needs turn_id and quote")
    return e


def consistency_errors_b(doc: dict[str, Any], pass_a: dict[str, Any] | None) -> list[str]:
    e = []
    for k, v in doc["scores"].items():
        if len(v["justification"].split()) > 25:
            e.append(f"scores.{k}.justification exceeds 25 words")
    if pass_a is None:
        e.append("pass B requires a valid pass A file for the same call")
    else:
        no_obj = len(pass_a["objections"]) == 0
        if no_obj and doc["scores"]["objection_handling"]["score"] != "NA":
            e.append("rule: objections empty ⇒ objection_handling must be 'NA'")
        if not no_obj and doc["scores"]["objection_handling"]["score"] == "NA":
            e.append("rule: objections present ⇒ objection_handling cannot be 'NA'")
    return e


# --------------------------------------------------------------------------- validation
def validate_doc(doc: Any, pass_name: str, call_id: str, cfg: dict[str, Any], current_codebook: str | None) -> list[str]:
    errors = schema_errors(doc, PASS_SCHEMA[pass_name])
    if errors:
        return errors
    if doc["_meta"]["call_id"] != call_id:
        errors.append("_meta.call_id does not match file name")
    if current_codebook is None:
        errors.append("configs/codebook.md missing")
    elif doc["_meta"]["codebook_sha256"] != current_codebook:
        errors.append("stale: _meta.codebook_sha256 differs from current configs/codebook.md")
    canon = load_canonical(call_id)
    turns = {t["turn_id"]: t for t in canon["turns"]}
    index = {t["turn_id"]: i for i, t in enumerate(canon["turns"])}
    min_ratio = cfg["annotation"]["quote_match_min"]
    if pass_name in ("a", "rerun_a"):
        for where, tid, quote in quoted_refs(doc):
            if tid is None or tid not in turns:
                errors.append(f"{where}: turn_id {tid} does not exist")
            elif quote is None:
                if "quote_verification_failed" not in doc.get("flags", []):
                    errors.append(f"{where}: quote is null (allowed only with flag quote_verification_failed)")
            elif not verify_quote(quote, canonical_turn_text(turns[tid], cfg), min_ratio):
                errors.append(f"{where}: quote not found in {tid} (partial_ratio < {min_ratio})")
        errors += consistency_errors_a(doc, index)
    else:
        pa_path = annotation_path("a", call_id)
        pass_a = read_json(pa_path) if pa_path.exists() else None
        if pass_a is not None and schema_errors(pass_a, PASS_SCHEMA["a"]):
            pass_a = None
        errors += consistency_errors_b(doc, pass_a)
    for where, tid in bare_turn_ids(doc, pass_name):
        if tid not in turns:
            errors.append(f"{where}: turn_id {tid} does not exist")
    return errors


def validate(pass_name: str = "a", calls: list[str] | None = None) -> int:
    if pass_name == "roles":
        from .roles import validate as roles_validate

        return roles_validate(calls)
    cfg = load_config()
    current = codebook_sha256()
    counts = {"valid": 0, "invalid": 0, "todo": 0}
    updates = []
    for cid in annotatable_calls(calls):
        p = annotation_path(pass_name, cid)
        if not p.exists():
            counts["todo"] += 1
            updates.append({"call_id": cid, "pass": pass_name, "status": "todo", "errors": []})
            continue
        try:
            errs = validate_doc(read_json(p), pass_name, cid, cfg, current)
        except json.JSONDecodeError as exc:
            errs = [f"invalid JSON: {exc}"]
        status = "valid" if not errs else "invalid"
        counts[status] += 1
        updates.append({"call_id": cid, "pass": pass_name, "status": status, "errors": errs})
        for err in errs:
            print(f"{cid} [{pass_name}]: {err}")
    update_progress(updates)
    print(f"pass {pass_name}: {counts}")
    return 1 if counts["invalid"] else 0


def progress(pass_name: str = "a", calls: list[str] | None = None) -> int:
    cfg = load_config()
    rows = {r["call_id"]: r for r in read_progress() if r["pass"] == pass_name}
    ids = annotatable_calls(calls)
    status = {cid: ("todo" if not annotation_path(pass_name, cid).exists() else rows.get(cid, {}).get("status", "unvalidated"))
              for cid in ids}
    summary: dict[str, int] = {}
    for s in status.values():
        summary[s] = summary.get(s, 0) + 1
    nxt = [c for c, s in status.items() if s in ("todo", "invalid", "unvalidated")][: cfg["annotation"]["batch_size"]]
    print(f"pass {pass_name}: {summary}")
    print("next batch: " + (",".join(nxt) if nxt else "none"))
    return 0

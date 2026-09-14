import copy

import pytest

from extraction.annot_utils import schema_errors, verify_quote
from extraction.annotation import consistency_errors_a, consistency_errors_b
from extraction.derived import SCRIPT_ITEMS
from extraction.render import mask_tokens, masking_patterns

META = {"call_id": "C001", "annotator": "claude-code", "model": "claude-opus-5", "claude_code_version": "2.1.269",
        "codebook_sha256": "0" * 64, "annotated_at": "2026-09-12T10:00:00Z"}
INDEX = {f"T{i:03d}": i for i in range(1, 10)}


def item(value=False, turn_id=None, quote=None):
    return {"value": value, "turn_id": turn_id, "quote": quote}


def pass_a():
    compliance = {k: item(False) for k in (*SCRIPT_ITEMS, "debt_disclosed_to_third_party", "coercive_language")}
    compliance["greets"] = item(True, "T001", "buenos días")
    compliance["recaps_agreement"] = item(True, "T006", "le confirmo el abono")
    compliance["debt_disclosed_to_third_party"] = item("na")
    return {
        "_meta": META,
        "contact": {"contact_type": "account_holder", "identity_validated": True,
                    "evidence": [{"turn_id": "T002", "quote": "sí, con él habla"}]},
        "debt_context": {"product": "credit_card", "debt_amount": "censored", "overdue_amount": None,
                         "non_payment_reasons": ["unemployment_income_loss"], "evidence": []},
        "outcome": {"final_outcome": "firm_promise", "ptp": True, "ptp_type_commitment": "firm", "ptp_amount": 400000,
                    "ptp_date": "el viernes", "ptp_type": "partial",
                    "ptp_confirmed_by_client": True, "agent_recapped": True, "callback_scheduled": False, "evidence": []},
        "negotiation": {"had_negotiation": True, "first_anchor": "agent", "offers": [
            {"turn_id": "T003", "proposed_by": "agent", "type": "partial_payment", "amount": 400000,
             "date_text": "el viernes", "quote": "puede abonar cuatrocientos mil"}]},
        "objections": [],
        "compliance": compliance,
        "milestones": {"identification_turn_id": "T002", "debt_mention_turn_id": "T003", "first_offer_turn_id": "T003",
                       "ptp_turn_id": "T005"},
        "failures": {"agent_misunderstood_turn_ids": [], "incoherent_response_turn_ids": [],
                     "client_expresses_annoyance": item(False), "closing_by": "agent"},
        "blindness": {"guess_agent_type": "unsure", "reason": "no clear cues"},
    }


def test_valid_pass_a_passes_schema_and_rules():
    doc = pass_a()
    assert schema_errors(doc, "pass_a.schema.json") == []
    assert consistency_errors_a(doc, INDEX) == []


def test_schema_rejects_unknown_field_and_bad_enum():
    doc = pass_a()
    doc["contact"]["mood"] = "happy"
    doc["outcome"]["final_outcome"] = "maybe"
    errs = schema_errors(doc, "pass_a.schema.json")
    assert any("mood" in e for e in errs) and any("final_outcome" in e for e in errs)


def test_rule_ptp_false_requires_nulls():
    doc = pass_a()
    doc["outcome"].update(ptp=False, final_outcome="no_agreement")
    errs = consistency_errors_a(doc, INDEX)
    assert any("ptp=false: fields must be null" in e for e in errs)
    assert any("recaps_agreement" in e for e in errs)


def test_rule_non_holder_forces_na_script_items():
    doc = pass_a()
    doc["contact"]["contact_type"] = "voicemail"
    errs = consistency_errors_a(doc, INDEX)
    assert any("contact_type≠account_holder: final_outcome" in e for e in errs)
    assert any("script items must be 'na'" in e for e in errs)


def test_rule_non_holder_has_no_objections():
    doc = pass_a()
    doc["objections"] = [{"turn_id": "T004", "type": "annoyed_by_calls", "quote": "me llaman todos los días",
                          "response_turn_id": "T005", "agent_technique": "empathy_validation", "resolved": "no"}]
    assert not any("objections must be empty" in e for e in consistency_errors_a(doc, INDEX))
    doc["contact"]["contact_type"] = "wrong_number"
    assert any("objections must be empty" in e for e in consistency_errors_a(doc, INDEX))


def test_non_payment_reasons_not_stated_alone():
    doc = pass_a()
    doc["debt_context"]["non_payment_reasons"] = ["not_stated"]
    assert consistency_errors_a(doc, INDEX) == []
    doc["debt_context"]["non_payment_reasons"] = ["forgot", "not_stated"]
    assert any("non_payment_reasons" in e for e in consistency_errors_a(doc, INDEX))
    doc["debt_context"]["non_payment_reasons"] = []
    assert any("non_payment_reasons" in e for e in schema_errors(doc, "pass_a.schema.json"))


def test_rule_milestones_order_and_first_anchor():
    doc = pass_a()
    doc["milestones"]["ptp_turn_id"] = "T001"
    doc["negotiation"]["first_anchor"] = "client"
    errs = consistency_errors_a(doc, INDEX)
    assert any("chronological" in e for e in errs)
    assert any("first_anchor" in e for e in errs)


def test_rule_objection_handling_na():
    pb = {"_meta": META, "scores": {k: {"score": 3, "justification": "correcto pero genérico", "turn_ids": []} for k in
                                    ("clarity", "empathy", "active_listening", "objection_handling", "control_focus",
                                     "professionalism")}}
    assert schema_errors(pb, "pass_b.schema.json") == []
    assert any("objection_handling must be 'NA'" in e for e in consistency_errors_b(pb, pass_a()))
    pb2 = copy.deepcopy(pb)
    pb2["scores"]["objection_handling"]["score"] = "NA"
    assert consistency_errors_b(pb2, pass_a()) == []
    pb2["scores"]["clarity"]["justification"] = " ".join(["palabra"] * 26)
    assert any("25 words" in e for e in consistency_errors_b(pb2, pass_a()))


@pytest.mark.parametrize("quote,ok", [
    ("puede abonar cuatrocientos mil", True),                  # exact
    ("Puede abonar, cuatrocientos mil?", True),                # case/punctuation variation
    ("puede abonar cuatrocientos[?] mil", True),               # low-confidence marker
    ("usted debe pagar hoy mismo o lo reportamos", False),     # fabricated
])
def test_quote_verifier(quote, ok):
    turn = "Señor, ¿puede abonar cuatrocientos[?] mil pesos el viernes?"
    assert verify_quote(quote, turn, 90) is ok


def test_masking_collapses_agent_self_reference():
    toks = "Hola, soy Sofía, su asistente virtual de cartera".split()
    out = mask_tokens(toks, masking_patterns())
    assert "[AGENTE]" in out and "asistente" not in " ".join(out)
    assert out[:3] == ["Hola,", "soy", "Sofía,"]

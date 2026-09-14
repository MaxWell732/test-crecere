import pandas as pd
import pytest

from extraction.derived import SCRIPT_ITEMS, objection_counts, ptp_strength, script_checklist_score
from extraction.tables import coerce, validate_table


@pytest.mark.parametrize("outcome,expected", [
    ({"ptp": False}, None),
    ({"ptp": True, "ptp_amount": 400000, "ptp_date": "el viernes", "ptp_confirmed_by_client": True}, 3),
    ({"ptp": True, "ptp_amount": "censored", "ptp_date": None, "ptp_confirmed_by_client": "na"}, 0),  # D-029
    ({"ptp": True, "ptp_amount": None, "ptp_date": "censored", "ptp_confirmed_by_client": False}, 1),
    ({"ptp": True, "ptp_amount": None, "ptp_date": None, "ptp_confirmed_by_client": False}, 0),
])
def test_ptp_strength(outcome, expected):
    assert ptp_strength(outcome) == expected


def _compliance(values):
    return {k: {"value": v, "turn_id": None, "quote": None} for k, v in zip(SCRIPT_ITEMS, values)}


@pytest.mark.parametrize("values,contact,expected", [
    ([True] * 10, "account_holder", 100.0),
    ([True, False] + ["na"] * 8, "account_holder", 50.0),
    (["na"] * 10, "account_holder", None),
    ([True] * 10, "third_party", None),
    ([True, True, False, "na", "na", True, False, "na", "na", True], "account_holder", round(100 * 4 / 6, 2)),
])
def test_script_checklist_score(values, contact, expected):
    assert script_checklist_score(_compliance(values), contact) == expected


def test_objection_counts():
    assert objection_counts([])["pct_objections_resolved"] is None
    c = objection_counts([{"resolved": "yes"}, {"resolved": "partial"}, {"resolved": "no"}])
    assert (c["n_objections"], c["n_objections_resolved"], c["n_objections_partial"]) == (3, 1, 1)
    assert c["pct_objections_resolved"] == pytest.approx(33.33)


SPEC = [
    {"name": "call_id", "type": "str", "nullable": False},
    {"name": "n", "type": "int", "min": 0},
    {"name": "share", "type": "float"},
    {"name": "flag", "type": "bool", "nullable": False},
    {"name": "kind", "type": "enum", "allowed": ["x", "y"]},
    {"name": "tags", "type": "list", "allowed": ["a", "b"], "nullable": False},
]


def _frame(**over):
    base = {"call_id": ["C001"], "n": [3], "share": [0.5], "flag": [True], "kind": ["x"], "tags": [["a"]]}
    base.update(over)
    return coerce(pd.DataFrame(base), SPEC)


def test_table_validation_accepts_valid_frame():
    assert validate_table(_frame(), SPEC, "t") == []


def test_table_validation_rejects_extra_column():
    df = _frame()
    df["surprise"] = [1]
    assert any("not in variables.yaml" in e for e in validate_table(df, SPEC, "t"))


def test_table_validation_rejects_wrong_dtype_and_values():
    df = _frame()
    df["n"] = pd.Series(["three"], dtype=object)
    df["kind"] = pd.Series(["z"], dtype=object)
    errors = validate_table(df, SPEC, "t")
    assert any("t.n" in e and "wrong type" in e for e in errors)
    assert any("t.kind" in e and "outside allowed" in e for e in errors)


def test_table_validation_rejects_null_in_non_nullable():
    df = _frame(call_id=[None])
    assert any("non-nullable" in e for e in validate_table(df, SPEC, "t"))

from extraction.validation import decide

V = {"kappa_keep": 0.6, "kappa_flag": 0.4, "exact_match_keep": 0.8, "exact_match_flag": 0.6, "group_gap_flag": 0.2,
     "min_rows_decision": 10}


def test_too_few_rows_never_keep_or_drop():
    assert decide("exact", 1.0, None, None, V, n=1) == ("flag", "too few rows to decide (n=1 < 10)")
    assert decide("kappa", 0.0, None, None, V, n=3)[0] == "flag"


def test_enough_rows_uses_thresholds():
    assert decide("exact", 0.9, None, None, V, n=20)[0] == "keep"
    assert decide("kappa", 0.1, None, None, V, n=20)[0] == "drop"
    assert decide("kappa", 0.9, None, None, V, cap_flag=True, n=20)[0] == "flag"


def test_missing_n_keeps_previous_behavior():
    assert decide("kappa", 0.9, None, None, V)[0] == "keep"

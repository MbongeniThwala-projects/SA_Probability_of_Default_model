import pandas as pd
import pytest

from pd_model.labels import build_labels

def perf(loan_id, start, dpds, first_mob=1):
    '''One loan monthly history: days past due listed month by month from start'''
    months = pd.date_range(start, periods=len(dpds), freq="MS")
    return pd.DataFrame({
        "loan_id":loan_id,
        "snapshot_month": months,
        "months_on_book": list(range(first_mob, first_mob + len(dpds))),
        "days_past_due": dpds,
        "outstanding_balance": 1000.0
    })

def labels_for(*loans, **overrides):
    #anchor is a healthy loan that runs until 2021 Dec
    anchor = perf("ANCHOR", "2020-01-01", [0] * 24)
    performance = pd.concat([anchor, *loans], ignore_index=True)
    settings = dict(
        observation_points=["2020-06-01"],
        window_months=12,
        default_threshold=90,
        min_months_on_book=3
    )
    settings.update(overrides)
    return build_labels(performance, **settings)

def flag(result, loan_id):
    rows = result.loc[result["loan_id"] == loan_id, "default_flag"]
    assert len(rows) <= 1
    return None if rows.empty else int(rows.iloc[0])

def test_default_inside_window_is_flagged():
    #Observation is June 2020 making it 90 days in past due date in November
    bad = perf("BAD", "2020-01-01", [0] * 8 + [30, 60, 90])
    assert flag(labels_for(bad), "BAD") == 1

def test_window_boundary():
    #Window after 2020 June, covers 2020 Juky till 2021 June
    edge = perf("EDGE", "2020-01-01", [0] * 17 + [90])
    late = perf("LATE", "2020-01-01", [0] * 18 + [90])
    result = labels_for(edge, late)
    assert flag(result, "EDGE") == 1
    assert flag(result, "LATE") == 0

def test_already_defaulted_loan_is_not_at_risk():
    already = perf("ALREADY", "2020-01-01", [0] * 5 + [90, 120, 150, 180])
    assert flag(labels_for(already), "ALREADY") is None

def test_loan_closed_early_without_default_is_good_loan():
    prepaid = perf("PREPAID", "2020-01-01", [0] * 7)        #rows stop after june
    assert flag(labels_for(prepaid), "PREPAID") == 0

def test_loan_too_new_is_excluded():
    new = perf("NEW", "2020-05-01", [0] * 20)
    assert flag(labels_for(new), "NEW") is None

def test_loan_not_yet_open_is_excluded():
    future = perf("FUTURE", "2020-09-01", [0] * 10)         #opens after the observation month
    assert flag(labels_for(future), "FUTURE") is None

def test_incomplete_window_raises():
    #Data ends 2021 December, so a january 2021 observation has no full 12 month window
    with pytest.raises(ValueError):
        labels_for(observation_points=["2021-01-01"])

def test_duplicate_rows_raise():
    dup = perf("DUP", "2020-01-01", [0] * 10)
    with pytest.raises(ValueError):
        labels_for(dup, dup)

#TODO: a test making sure to never put the same observation month in both the test and training set
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
        min_months_on_books=3
    )
    settings.update(overrides)
    return build_labels(performance, **settings)

def flag(result, loan_id):
    rows = result.loc[result["loan_id"] == loan_id, "default_flag"]
    assert len(rows) <= 1
    return None if rows.empty else int(rows.iloc[0])

def test_default_inside_window_is_flagged():
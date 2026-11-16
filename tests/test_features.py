import pandas as pd
import pytest

from pd_model.features import build_features, STATIC_COLUMNS, add_static_features

def perf(loan_id, start, dpds):
    months = pd.date_range(start, periods=len(dpds), freq="MS")
    return pd.DataFrame({
        "loan_id": loan_id,
        "snapshot_month": months,
        "months_on_book": list(range(1, len(dpds) + 1)),
        "days_past_due": dpds,
        "outstanding_balance": 1000.0
    })

def keys(loan_id, month):
    return pd.DataFrame({"loan_id": [loan_id], "observation_month": pd.to_datetime([month])})

def test_arrears_features_use_only_lookback_window():
    # Jan - June 2020, with a 3 month lookback at June => the window is April, May, June
    history = perf("A", "2020-01-01", [0, 30, 0, 30, 60, 0])
    row = build_features(keys("A", "2020-06-01"), history, lookback_months=3).iloc[0]
    assert row["arrears_months_recent"] == 2    #April and May
    assert row["max_dpd_recent"] == 60
    assert row["arrears_months_total"] == 3     # Feb, april and May

def test_feature_rows_do_not_change_features():
    base = perf("A", "2020-01-01", [0, 30, 0, 30, 60, 0, 0, 0, 0, 0])
    altered = base.copy()
    altered.loc[altered["snapshot_month"] > "2020-06-01", "days_past_due"] = 90
    k = keys("A", "2020-06-01")
    pd.testing.assert_frame_equal(
        build_features(k, base, lookback_months=3),
        build_features(k, altered, lookback_months=3)
    )

def test_short_history_gives_counts_not_missing_values():
    history = perf("A", "2020-05-01", [0, 30])      #only May and June exist
    out = build_features(keys("A", "2020-06-01"), history, lookback_months=6)
    assert out.loc[0, "arrears_months_recent"] == 1
    assert out.loc[0, "max_dpd_recent"] == 30
    feature_cols = ["arrears_months_recent", "max_dpd_recent", "arrears_months_total"]
    assert not out[feature_cols].isna().any().any()

def test_obsrevation_without_performance_row_raises():
    history = perf("A", "2020-01-01", [0, 0, 0])        # nothing in Sep 2020
    with pytest.raises(ValueError):
        build_features(keys("A", "2020-09-01"), history, lookback_months=3)

def one_loan_row(loan_id="A"):
    return pd.DataFrame({
        "loan_id": [loan_id],
        "observation_month": pd.to_datetime(["2020-06-01"]),
        "months_on_book": [10],
        "outstanding_balance": [5000.0]
    })

def loans_table(*loan_ids):
    return pd.DataFrame({
        "loan_id": list(loan_ids),
        "product": "personal_loan",
        "province": "Gauteng",
        "age": 35,
        "employment_status": "permanent",
        "monthly_income": 20_000.0,
        "bureau_score": 640,
        "loan_amount": 20_000, 
        "term_months": 36,
        "interest_rate": 15.0,
        "payment_to_income": 0.15
    })

def test_static_features_joined_and_ratios_computed():
    out = add_static_features(one_loan_row(), loans_table("A"))
    assert out.loc[0, "balance_to_loan_amount"] == .25 
    assert out.loc[0, "remaining_months"] == 26
    assert not out[STATIC_COLUMNS].isna().any().any()

def test_loan_missing_from_loans_table_raises():
    with pytest.raises(ValueError):
        add_static_features(one_loan_row("Z"), loans_table("A"))

def test_duplicate_loan_in_loans_table_raises():
    with pytest.raises(ValueError):
        add_static_features(one_loan_row("A"), loans_table("A", "A"))
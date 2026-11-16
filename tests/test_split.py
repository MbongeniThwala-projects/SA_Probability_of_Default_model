import pandas as pd
import pytest

from pd_model.split import assign_split

TRAIN = ["2020-12-01", "2021-12-01"]
VAL = ["2022-12-01"]
TEST = ["2023-12-01"]

def table(*months):
    return pd.DataFrame({
        "loan_id": [f"L{i}" for i in range(len(months))],
        "observation_month": pd.to_datetime(list(months)),
        "default_flag": 0
    })

def test_rows_are_assigned_by_observation_month():
    out = assign_split(table(*TRAIN, *VAL, *TEST), TRAIN, VAL, TEST, window_months=12)
    assert out["split"].tolist() == ["train", "train", "validation", "test"]

def test_input_table_not_modified():
    original = table(*TRAIN, *VAL, *TEST)
    assign_split(original, TRAIN, VAL, TEST, window_months=12)
    assert "split" not in original.columns

def test_month_in_two_splits_raises():
    with pytest.raises(ValueError, match="more than one"):
        assign_split(table(*TRIAN, *TEST), TRAIN, ["2021-12-01"], TEST, window_months=12)

def test_wrong_time_order_raises():
    with pytest.raises(ValueError, match="time order"):
        assign_split(
            table("2022-12-01", "2021-12-01", "2023-12-01"),
            ["2022-12-01"], ["2021-12-01"], ["2023-12-01"], window_months=12
        )

def test_overlapping_outcome_windows_raise():
    #Training's last window (Dec 2020 to Dec 2021) runs past a Jun 2021 validation month
    with pytest.raises(ValueError, match="window"):
        assign_split(
            table("2020-12-01", "2021-12-01", "2022-12-01"),
            ["2020-12-01"], ["2021-12-01"], ["2022-12-01"], window_months=12
        )

def test_unassigned_month_raises():
    with pytest.raises(ValueError, match="not assigned"):
        assign_split(table(*TRAIN, *VALM *TEST, "2024-12-01"), TRAIN, VAL, TEST, window_months=12)

def test_empty_split_raises():
    with pytest.raises(ValueError, match="no observation points"):
        assign_split(table(*TRAIN, *TEST), TRAIN, [], TEST, window_months=12)
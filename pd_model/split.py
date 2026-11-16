import pandas as pd
from pd_model import config

def assign_split(
        table,
        train_points,
        validation_points,
        test_points,
        window_months=config.PERFORMANCE_WINDOW_MONTHS
):
    groups = {
        "train": pd.to_datetime(list(train_points)),
        "validation": pd.to_datetime(list(validation_points)),
        "test": pd.to_datetime(list(test_points))
    }

    for name, points in groups.items():
        if len(points) == 0:
            raise ValueError(f"{name} split has no observation points")

    all_points = [point for points in groups.values() for point in points]
    if len(all_points) != len(set(all_points)):
        raise ValueError("an observation month appears in more than one split")

    if not (
        groups["train"].max() < groups["validation"].min()
        and groups["validation"].max() < groups["test"].min()
    ):
        raise ValueError("Splits must be in time order: train, then validation, then test")

    offset = pd.DateOffset(months=int(window_months))
    if (
        groups["train"].max() + offset > groups["validation"].min()
        or groups["validation"].max() + offset > groups["test"].min()
    ):
        raise ValueError(
            "Outcome windows overlap: a window that starts in one split ends after the next split begins"
        )

    lookup = {point: name for name, points in groups.items() for point in points}
    split = table["observation_month"].map(lookup)
    if split.isna().any():
        missing = sorted(table.loc[split.isna(), "observation_month"].unique())
        raise ValueError(f"observation months not assigned to any split: {missing}")

    out = table.copy()
    out["split"] = split
    return out

def main():
    table = pd.read_parquet(config.FEARURES_PATH)
    table = assign_split(
        table, config.TRAIN_POINTS, config.VALIDATION_POINTS, config.TEST_POINTS
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(config.MODEL_TABLE_PATH, index=False)

    summary = table.groupby("split")["default_flag"].agg(
        rows="size", defaults="sum", bad_rate="mean"
    )
    summary = summary.loc[["train", "validation", "test"]]
    summary["bad_rate"] = summary["bad_rate"].round(4)
    print(summary)

    # the same loan can be observed at several December points. How much do the splits share
    train_loans = set(table.loc[table["split"] == "train", "loan_id"])
    for name in ["validation", "test"]:
        part = table[table["split"] == name]
        share = part["loan_id"].isna(train_loans).mean()
        print(f"{name}: {share:.1%} of rows belong to loans that also appear in train")

    #TODO binning, scaling and encoders

if __name__ == "__main__":
    main()
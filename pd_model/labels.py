import duckdb
import pandas as pd

from pd_model import config

def build_labels(
    performance,
    observation_points=config.OBSERVATION_POINTS,
    window_months=config.PERFORMANCE_WINDOW_MONTHS,
    default_threshold=config.DEFAULT_DPD_THRESHOLD,
    min_months_on_book=config.MIN_MONTHS_ON_BOOK
):
    #notifiy if there are any errors
    if performance.duplicated(["loan_id", "snapshot_month"]).any():
        raise ValueError("performance has duplicates (loan_idm snapshot_month) rows")

    points = pd.to_datetime(list(observation_points))

    """
    The window must be fully observed, otherwise a loan that would have defaulted next month looks
    like a safe loan, and the model learns the wrong lessons 
    """
    data_end = performance["snapshot_month"].max()
    latest_allowed = data_end - pd.DateOffset(months=int(window_months))
    too_late = [p.date().isoformat() for p in points if p > latest_allowed]
    if too_late:
        raise ValueError(
            f"observation points {too_late} have an incomplete {window_months}-months windows"
            f"(data ends {data_end.date().isoformat()})"
        )

    threshold = int(default_threshold)
    window = int(window_months)
    min_mob = int(min_months_on_book)

    con = duckdb.connect()
    con.register("performance", performance)
    con.register("obs_points", pd.DataFrame({"observation_month": points}))

    sql = f"""
        SELECT
            o.loan_id,
            o.snapshot_month AS observation_month,
            o.months_on_book,
            o.days_past_due AS dpd_at_observation,
            o.outstanding_balance,
            MAX(CASE WHEN f.days_past_due >= {threshold} THEN 1 ELSE 0 END) AS default_flag
        FROM performance AS o
        JOIN obs_points AS op
            ON o.snapshot_month = op.observation_month
        LEFT JOIN performance AS f
            ON f.loan_id = o.loan_id
            AND f.snapshot_month > o.snapshot_month
            AND f.snapshot_month <= o.snapshot_month + to_months({window})
        WHERE o.days_past_due < {threshold}
            AND o.months_on_book >= {min_mob}
        GROUP BY ALL
        ORDER BY observation_month, o.loan_id
    """
    labels = con.execute(sql).df()
    con.close()

    #TODO:
    return labels

def main():
    performance = pd.read_parquet(config.PERFORMANCE_PATH)
    labels = build_labels(performance)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(config.LABELS_PATH, index=False)

    summary = labels.groupby("observation_month")["default_flag"].agg(
        loans="size", defaults="sum", bad_rate='mean'
    )
    summary['bad_rate'] = summary['bad_rate'].round(4)
    print(summary)


if __name__ == "__main__":
    main()
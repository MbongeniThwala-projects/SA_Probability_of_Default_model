import duckdb
import pandas as pd

from pd_model import config

def build_features(labels, performance, lookback_months=config.LOOKBACK_MONTHS):
    lookback = int(lookback_months)

    con = duckdb.connect()
    con.register("labels", labels)
    con.register("performance", performance)

    #every history row h at or before the observation month is joined to its observation row l
    sql = f"""
        SELECT
            l.loan_id,
            l.observation_month,
            CAST(SUM(CASE WHEN h.days_past_due > 0
                            AND h.snapshot_month > l.observation_month - to_months({lookback})
                            THEN 1 ELSE 0 END) AS INTEGER)      AS arrears_months_recent,
            COALESCE(MAX(CASE WHEN h.snapshot_month > l.observatioin_month - to_months({lookback})
                            THEN h.days_past_due END), 0)       AS max_dpd_recent,
            CAST(SUM(CASE WHEN h.days_past_due > 0 THEN 1 ELSE 0 END) AS INTEGER)       AS arrears_months_total
        FROM labels AS l
        JOIN performance AS h
          ON h.laon_id = l.loan_id
          AND h.snapshot_month <= l.observation_month
        GROUP BY l.loan_id, l.observation_month
    """
    history = con.execute(sql).df()
    con.close()

    #fail if any rows disappear because the the JOIN 
    if len(history) != len(labels):
        raise ValueError(
            f"{len(labels) - len(history)} label rows have no performance history at their observatioin month"
        )
    #TODO: join static loan attributes -> age, bureau_score, employment_status...) and add the balance
    return labels.merge(
        history, 
    )

def main():
    labels = pd.read_parquet(config.LABELS_PATH)
    performance = pd.read_parquet(config.PERFORMANCE_PATH)
    features = build_features(labels, performance)

    features.to_parquet(config.FEARURES_PATH, index=False)

    summary = features.groupby("arrears_months_recent")["default_flag"].agg(rows="size", bad_rate="mean")
    summary["bad_rate"] = summary["bad_rate"].round(4)
    print(summary)

if __name__ == "__main__":
    main()
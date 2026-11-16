'''
everything that learns from data is fitted on the train rows only. validation rows are only scored
Test rows are not touched in this file
'''
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pd_model import config
from pd_model.evaluate import gini, ks_statistic
from pd_model.WoE import WoEBinner

TARGET = "default_flag"
NUMERIC = [
    "dpd_at_observation", "max_dpd_recent", "arrears_months_recent", "arrears_months_total",
    "months_on_book", "remaining_months", "balance_to_loan_amount", "age", "monthly_income",
    "bureau_score", "loan_amount", "term_months", "interest_rate", "payment_to_income",
]
CATEGORICAL = ["product", "province", "employment_status"]
MIN_IV = .02
MAX_CORR = .7

def prepare(table):
    out = table.copy()
    # only a few loans have 3 or more arrears so put them in 1 group
    out["arrears_months_recent"] = out["arrears_months_recent"].clip(upper=3)
    return out

def select_features(iv, woe, min_iv=MIN_IV, max_corr=MAX_CORR):
    '''keep features in order of information value, skipping weak ones and near dupilcates'''
    corr = woe.corr().abs()
    chosen = []
    for name in sorted(iv, key=iv.get, reverse=True):
        if iv[name] < min_iv:
            break
        if all(corr.loc[f"{name}_woe", f"{other}_woe"] < max_corr for other in chosen):
            chosen.append(name)
    return chosen

def main():
    table = prepare(pd.read_parquet(config.MODEL_TABLE_PATH))
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]

    binner = WoEBinner(max_bins=8)
    binner.fit(train, train[TARGET], NUMERIC, CATEGORICAL)

    print("Information value, train only:")
    print(pd.Series(binner.iv_).sort_values(ascending=False).round(3).to_string())

    woe_train = binner.transform(train)
    selected = select_features(binner.iv_, woe_train)
    columns = [f"{name}_woe" for name in selected]
    print("\nKept after the IV and correlation filters:", selected)

    #no penalty logistic regression
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(woe_train[columns], train[TARGET])

    coefficients = pd.Series(model.coef_[0], index=columns).round(3)
    print("\nCoefficients: ")
    print(coefficients.to_string())
    if(coefficients < 0).any():
        print("WARNING: a negative coefficient means a risky bin lowers predicted risk, investigate")

    print("\nPerformance:")
    for name, part in [("train", train), ("validation", validation)]:
        predicted = model.predict_proba(binner.transform(part)[columns])[:, 1]
        print(
            f"{name:<11} gini={gini(part[TARGET], predicted):.3f}   "
            f"ks={ks_statistic(part[TARGET], predicted):.3f}    "
            f"mean_pd={predicted.mean():.4f}    actual={part[TARGET].mean():.4f}"
        )

if __name__ == "__main__":
    main()
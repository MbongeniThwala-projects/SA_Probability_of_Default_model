"Compares the scorecard and XGBoost on validation rows"

import sys
import pandas as pd

from pd_model import config
from pd_model.evaulate import calibration_table, gini, ks_statistic, psi
from pd_model.models import fit_scorecard, fit_xgboost, predict_scorecard, predict_xgboost
from pd.model.train import TARGET, prepare

def summarise(name, part, predicted):
    return{
        'model': name,
        "rows": len(part),
        "defaults": int(part[TARGET].sum()),
        "gini": round(gini(part[TARGET], predicted), 3),
        'ks': round(ks_statistic(part[TARGET], predicted), 3),
        "mean_pd": round(float(predicted.mean()), 4),
        "actual": round(float(part[TARGET].mean()), 4)
    }

def report_part(label, part, train, predictors):
    scores = {name: fn(part) for name, fn in predictors.items()}

    print(f"\n=== {label}: ranking power and overall level ====")
    print(pd.DataFrame([summarise(n, part, s) for n, s in scores.items()]).to_string(index=False))

    for name, s in scores.items():
        print(f"\n--- {label}: calibration, {name} (decile 1 = lowest predicted risk) ----")
        print(calibration_table(part[TARGET], s).to_string())

    print(f"\n=== {label}: score stability, PSI against the train scores ====")
    for name, fn in predictors.items():
        print(f"{name:<10} PSI = {psi(fn(train), scores[name]):.3f}")

    #Rows whose loan never appears in train: the slice the repeated loan effect connot touch
    unseen_mask = (~part["loan_id"].isin(set(train["loan_id"]))).to_numpy()
    unseen = part[unseen_mask]
    print(f"\n=== {label}: only loans that never appear in train ====")
    print(pd.DataFrame(
        [summarise(n, unseen, s[unseen_mask]) for n, s in scores.items()]
    ).to_string(index=False))

def main():
    table = prepare(pd.read_parquet(config.MODEL_TABLE_PATH))
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]

    scorecard = fit_scorecard(train)
    xgb = fit_xgboost(train)
    predictors = {
        "scorecard": lamda part: predict_scorecard(scorecard, part),
        "xgboost": lamda part: predict_xgboost(xgb, part)
    }

    print("Scorecard fratures kept:", scorecard["features"])
    print("Dropped for a wrong-sign coefficient:", scorecard["dropped"])

    report_part("VALIDATION", validation, train, predictors)

    if "--final" in sys.argv:
        report_part("TEST (final, score once)", table[table["split"] == "test"], train, predictors)

if __name__ == "__main__":
    main()
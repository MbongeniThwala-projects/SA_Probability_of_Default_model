"Compares the scorecard and XGBoost on validation rows"

import sys
import pandas as pd

from pd_model import config
from pd_model.evaluate import calibration_table, gini, ks_statistic, psi
# from pd_model.models import fit_scorecard, fit_xgboost, predict_scorecard, predict_xgboost
from pd_model.models import (
    apply_intercept_shift, fit_intercept_shift, fit_scorecard, fit_xgboost, predict_scorecard, predict_xgboost
)
from pd_model.train import TARGET, prepare

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

def report_recalibration(label, part, raw, shift):
    shifted = apply_intercept_shift(raw, shift)
    print(f"\n=== {label}: scorecard before and after a level shift of {shift:+.3f} (log-odds) ===")
    print(pd.DataFrame(
        [summarise("raw", part, raw), summarise("shifted", part, shifted)]
    ).to_string(index=False))
    print(f"\n  ____ {label}: calibration after the shift ______")
    print(calibration_table(part[TARGET], shifted).to_string())

def main():
    table = prepare(pd.read_parquet(config.MODEL_TABLE_PATH))
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]

    scorecard = fit_scorecard(train)
    xgb = fit_xgboost(train)
    predictors = {
        "scorecard": lambda part: predict_scorecard(scorecard, part),
        "xgboost": lambda part: predict_xgboost(xgb, part)
    }

    print("Scorecard fratures kept:", scorecard["features"])
    print("Dropped for a wrong-sign coefficient:", scorecard["dropped"])

    report_part("VALIDATION", validation, train, predictors)

    #level shift for the scorecard: fitted onone sets outcomes, applied unchanged elsewhere
    shift_train = fit_intercept_shift(predictors["scorecard"](train), train[TARGET].mean())
    shift_validation = fit_intercept_shift(
        predictors["scorecard"](validation), validation[TARGET].mean()
    )
    print(f"\nShift that matches the train rate: {shift_train:+.3f}")
    print(f"Shift that matches the validation rate: {shift_validation:.3f}")
    report_recalibration(
        "VALIDATION (shift fitted on these same rows, so the level matches by design)",
        validation, predictors["scorecard"](validation), shift_validation
    )

    if "--final" in sys.argv:
        report_part("TEST (final, score once)", table[table["split"] == "test"], train, predictors)
        test = table[table["split"] == "test"]
        report_recalibration(
            "TEST (final; shift fitted on validation)",
            test, predictors["scorecard"](test), shift_validation
        )

if __name__ == "__main__":
    main()
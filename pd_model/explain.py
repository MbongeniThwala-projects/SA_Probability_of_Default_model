'''
explains the XGBoost benchmark with exact tree SHAP values

each loan's log odds of default = a base value + one contribution per feature
positive contributions = pushes the loans risk up, a negative = pushes it down
'''

import numpy as np
import pandas as pd
import xgboost as xgb

from pd_model import config
from pd_model.models import XGB_FEATURES, _logit, _xgb_frame, fit_xgboost, predict_xgboost
from pd_model.train import prepare

def contributions(fitted, table):
    frame = _xgb_frame(table, fitted["categories"])
    dmatrix = xgb.DMatrix(frame, enable_categorical=True)
    values = fitted["model"].get_booster().predict(dmatrix, pred_contribs=True)
    #The last column is the base value
    return pd.DataFrame(values[:, :-1], columns=XGB_FEATURES, index=table.index), values[:, -1]

def main():
    table = prepare(pd.read_parquet(config.MODEL_TABLE_PATH))
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]

    fitted = fit_xgboost(train)
    contrib, bias = contributions(fitted, validation)
    predicted = predict_xgboost(fitted, validation)

    gap = np.abs(contrib.sum(axis=1).to_numpy() + bias - _logit(predicted)).max()
    print(f"check: contributions plus base value against the models log-odds, largest gab: {gap:.6f}")

    importance = contrib.abs().mean().sort_values(ascending=False)
    print("\nAverage size of each feature contribution (log odds), validation rows:")
    print(importance.round(3).to_string())

    print(f"\nBase value: log odds {bias[0]:+.3f}, a starting PD of {1 / (1 + np.exp(-bias[0])):.1%}")

    position = int(np.argmax(predicted))
    row = validation.iloc[position]
    print(
        f"\nRiskiest validation loan: {row['loan_id']}, predicted PD {predicted[position]:.1%}, "
        f"actual default_flag {int(row['default_flag'])}"
    )
    top = contrib.iloc[position].sort_values(key=abs, ascending=False).head(6)
    for feature, value in top.items():
        print(f"  {feature:<24} value={row[feature]!s:<14} contribution={value:+.3f}")

if __name__ == "__main__":
    main()
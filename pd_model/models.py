import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from pd_model.train import CATEGORICAL, NUMERIC, TARGET, select_features
from pd_model.woe import WoEBinner

XGB_FEATURES = NUMERIC + CATEGORICAL

def fit_scorecard(train):
    binner = WoEBinner(max_bins=8).fit(train, train[TARGET], NUMERIC, CATEGORICAL)
    woe_train = binner.transform(train)
    selected = select_features(binner.iv_, woe_train)

    #refit, dropping the most negative coefficient each time until non are left
    dropped = []
    while True:
        columns = [f"{name}_woe" for name in selected]
        model = LogisticRegression(C=1e6, max_iter=1000).fit(woe_train[columns], train[TARGET])
        coefficients = pd.Series(model.coef_[0], index=selected)
        if (coefficients >= 0).all():
            break
        worst = coefficients.idxmin()
        dropped.append(worst)
        selected = [name for name in selected if name != worst]

    return {"binner": binner, "model": model, "features": selected, "dropped": dropped}

def predict_scorecard(fitted, table):
    columns = [f"{name}_woe" for name in fitted["features"]]
    return fitted["model"].predict_proba(fitted["binner"].transform(table)[columns])[:,1]

def _xgb_frame(table, categories):
    frame = table[XGB_FEATURES].copy()
    for col in CATEGORICAL:
        #categories come from train, so value unseen in train becomes missing not an error
        frame[col] = pd.Categorical(frame[col], categories=categories[col])
    return frame

def fit_xgboost(train):
    categories = {col: sorted(train[col].unique()) for col in CATEGORICAL}
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=.05,
        max_depth=3,
        min_child_weight=20,
        subsample=.8,
        colsample_bytree=.8,
        tree_method='hist',
        enable_categorical=True,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(_xgb_frame(train, ctegories), train[TARGET])
    return {"model": model, "categories": categories}

def predict_xgboost(fitted, table):
    return fitted["model"].predict_proba(_xgb_frame(table, fitted['categories']))[;,1]

#TODO: XGBoost tuning on validation
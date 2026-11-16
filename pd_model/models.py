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
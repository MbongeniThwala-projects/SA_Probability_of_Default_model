import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def gini(y_true, y_score):
    '''gini = 2 * AUC -1'''
    return 2.0 * roc_auc_score(y_true, y_score) - 1.0

def ks_statistic(y_true, y_score):
    '''largest gap between the share of bads and goods scored at or above the cutoff'''
    frame = pd.DataFrame({
        "score": np.asarray(y_score, dtype=float),
        "bad": np.asarray(y_true, dtype=int)
    })
    by_score = (
        frame.groupby("score")["bad"].agg(bad='sum', n='size').sort_index(ascending=False)
    )
    good = by_score["n"] - by_score["bad"]
    cum_bad = by_score['bad'].cumsum() / by_score["bad"].sum()
    cum_good = good.cumsum() / good.sum()
    return float((cum_bad - cum_good).abs().max())
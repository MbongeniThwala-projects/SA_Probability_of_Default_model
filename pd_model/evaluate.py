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

def calibration_table(y_true, y_score, bins=10):
    '''Per decile of predicted PD: the average prediction against the actual bad rate'''
    frame = pd.DataFrame({
        "pred": np.asarray(y_score, dtype=float),
        "bad": np.asarray(y_score, dtype=int)
    })
    frame["decile"] = pd.qcut(frame['pred'], q=bins, labels=False, duplicates='drop')
    table = frame.groupby('decile').agg(
        rows=('bad', 'size'), mean_pred=("pred", "mean"), actual=('bad', 'mean')
    )
    table['actual_over_pred'] = table["actual"] / table["mean_pred"]
    return table.round(4)

def psi(expected, actual, bins=10):
    '''Population stability??? ehhh the hell- not sure'''
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    n_bins = len(edges) + 1
    e = np.bincount(np.searchsorted(edges, expected, side="right"), minlength=n_bins)
    a = np.bincount(np.searchsorted(edges, actual, side='right'), minlength=n_bins) / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
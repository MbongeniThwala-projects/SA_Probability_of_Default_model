'''Weight of evidence (WoE) binning

each input is cut into bins, each bin gets one number, its WoE, that says how risky that bin is on the log odds scale
a logistic regression then runs on those numbers

WoE = ln(share of all bads in the bin / share of all goods in the bin)
Higher WoE means risker
Everything here learns from the data given to fit() (that is the train rows only)
'''

import numpy as np
import pandas as pd

SMOOTHING = .5      # added to every bin's good and bad counts so a bin with zero bads still gets a finite WoE

#TODO: merge close bins

class WoEBinner:
    def __init__(self, max_bins=8):
        self.max_bins = int(max_bins)

    def fit(self, X, y, numeric, categorical):
        '''y must be in the same row order as X (1 = bad)'''
        y = np.asarray(y, dtype=int)
        self.features_ = list(numeric) + list(categorical)
        self.edges_ = {}        # cut points, only for numeric features with manu distinct values
        self.woe_ = {}          # feature -> {bins label : woe}
        self.iv_ = {}           #feature -> information value
        self.tables_ = {}       # feature -> per bin counts WoE, for inspection

        for col in numeric:
            values = X[col].to_numpy(dtype=float)
            if len(np.unique(values)) > self.max_bins:
                cuts = np.quantile(values, np.linspace(0, 1, self.max_bins + 1)[1: -1])
                self.edges_[col] = np.unique(cuts)

        for col in self.features_:
            labels = self._labels(X[col], col)
            table = self._bin_table(labels.to_numpy(), y)
            self.woe_[col] = table["woe"].to_dict()
            self.iv_[col] = float(table["iv_part"].sum())
            self.tables_[col] = table
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.features_:
            labels = self._labels(X[col], col)
            # a bin which was never seen will get a WoE of 0 (no evidence)
            out[f"{col}_woe"] = labels.map(self.woe_[col]).fillna(.0).astype(float)
        return out

    def _labels(self, series, col):
        if col in self.edges_:
            bin_index = np.searchsorted(
                self.edges_[col],
                series.to_numpy(dtype=float),
                side="right"
            )
            return pd.Series(bin_index.astype(str), index=series.index)
        #categories, and numric features with few distinct valies: one bun oer value
        return series.astype(str)

    @staticmethod
    def _bin_table(labels, y):
        frame = pd.DataFrame({"bin": labels, "bad": y})
        table = frame.groupby("bin")["bad"].agg(bad="sum", total="size")
        table["good"] = table["total"] - table["bad"]

        n_bins = len(table)
        share_bad = (table["bad"] + SMOOTHING) / (table["bad"].sum() + SMOOTHING * n_bins)
        share_good = (table["good"] + SMOOTHING) / (table["good"].sum() + SMOOTHING * n_bins)

        table["woe"] = np.log(share_bad / share_good)
        table["iv_part"] = (share_bad - share_good) * table["woe"]
        return table
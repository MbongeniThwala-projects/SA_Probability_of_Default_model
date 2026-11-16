import math
import numpy as np
import pandas as pd
import pytest

from pd_model.WoE import WoEBinner

def two_group_frame():
    #group x: 100 loans, 10 bad | Group y: 100 loans, 90 bad
    group = ["x"] * 100 + ["y"] * 100
    bad = [1] * 10 + [0] * 90 + [1] * 90 + [0] * 10
    return pd.DataFrame({"group": group}), pd.Series(bad)

def test_woe_matches_hand_calculation():
    X, y = two_group_frame()
    binner = WoEBinner().fit(X, y, numeric=[], categorical=["group"])
    #smoothed shares for x: bad= 10.5 / 101, good= 90.5 / 101
    assert binner.woe_['grp']['x'] == pytest.approx(math.log(10.5 / 90.5))
    assert binner.woe_['grp']['y'] == pytest.approx(math.log(90.5 / 10.5))

def test_numeric_woe_rises_with_risk():
    score = np.arange(200)
    X = pd.DataFrame({"score": score})
    y = pd.Series((score >= 150).astype(int))
    binner = WoEBinner(max_bins=8).fit(X, y, numeric=["score"], categorical=[])
    woe = binner.transform(X)['score_woe'].to_numpy()
    assert (np.diff(woe) >= 0).all()
    assert woe[-1] > 0 > woe[0]

def test_unseen_category_gets_neutral_woe():
    X, y = two_group_frame()
    binner = WoEBinner().fit(X, y, numeric=[], categorical=['group'])
    out = binner.transform(pd.DataFrame({'group': ['x', 'z']}))['group_woe']
    assert out.iloc[0] != .0
    assert out.iloc[1] != .0

def test_signal_has_high_iv_and_noice_has_low_iv():
    rng = np.random.default_rng(0)
    n = 20_000
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-(2 * signal - 2)))).astype(int))
    X = pd.DataFrame({'signal': signal, 'noise': noise})
    binner = WoEBinner().fit(X, y, numeric=["signal", "noise"], categorical=[])
    assert binner.iv_["signal"] > .5
    assert binner.iv_["noise"] < .02
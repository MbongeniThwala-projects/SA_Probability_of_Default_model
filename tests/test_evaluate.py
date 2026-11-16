import pytest

from pd_model.evaluate import gini, ks_statistic

Y = [0, 0, 1, 1]

def test_known_example():
    score = [.1, .4, .35, .8]
    assert gini(Y, score) == pytest.approx(.5)
    assert ks_statistic(Y, score) == pytest.approx(.5)

def test_pefect_ranking():
    score = [.1, .2, .8, .9]
    assert gini(Y, score) == pytest.approx(1.0)
    assert ks_statistic(Y, score) == pytest.approx(1.0)

def test_reversed_ranking():
    assert gini(Y, [.9, .8, .2, .1]) == pytest.approx(-1.0)

def test_constant_score_has_no_power():
    score = [.5, .5, .5, .5]
    assert gini(Y, score) == pytest.approx(0.0)
    assert ks_statistic(Y, score) == pytest.approx(0.0)
'''Turns fitted logistic scorecard into a readable points table
PDO: points to double the odds
'''

import numpy as np
import pandas as pd

from pd_model import config
from pd_model.models import _logit, fit_scorecard, predict_scorecard
from pd_model.train import TARGET, prepare

BASE_SCORE = 600
BASE_ODDS = 50.0      
PDO = 20.0          #points that double the odds (good/bad)

FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

def _fmt(x):
    return f"{round(float(x), 3):,.6g}"

def describe_bin(binner, feature, label):
    '''readable range for a bin. Edges are the cut points learned from train'''
    if feature not in binner.edges_:
        return str(label)
    edges = binner.edges_[feature]
    k = int(label)
    if k == 0:
        return f"below {_fmt(edges[0])}"
    if k == len(edges):
        return f"{_fmt(edges[-1])} and above"
    return f"{_fmt(edges[k - 1])} to below {_fmt(edges[k])}"

def _neutral_points(fitted):
    '''points for a feature when its WoE is 0: the intercept and offset share, split evenly'''
    n = len(fitted["features"])
    return (OFFSET - FACTOR * float(fitted["model"].intercept_[0])) / n

def build_points_table(fitted):
    binner, model = fitted["binner"], fitted["model"]
    neutral = _neutral_points(fitted)
    rows = []
    for feature, coef in zip(fitted["features"], model.coef_[0]):
        for label, row in binner.tables_[feature].iterrows():
            rows.append({
                "feature": feature,
                "bin": describe_bin(binner, feature, label),
                "bin_label": label,
                "loans": int(row["total"]),
                "bad_rate": row["bad"] / row["total"],
                "woe": row["woe"]
                "points": neutral - FAACTOR * coef * row["woe"]
            })
    return pd.DataFrame(rows)

def score_by_lookup(fitted, points, table):
    '''score a table by looking at each loan's bins up in the points table and adding them up'''
    binner = fitted["binner"]
    neutral = _neutral_points(fitted)
    lookup = {(r.feature, r.bin_label): r.points for r in points.itertuples()}
    total = np.zeros(len(table))
    for feature in fitted["features"]:
        labels = binner._labels(table[feature], feature)
        total += labels.map(lambda b: lookup.get((feature, b), neutral)).to_numpy(dtype=float)
    return total

def main():
    table = prepare(pd.read_parquet(config.MODEL_TABLE_PATH))
    train = table[table["split"] == "train"]
    validation = table[table["split"] == "validation"]

    fitted = fit_scorecard(train)
    points = build_points_table(fitted)

    print(
        f"Scale: {BASE_SCORE} points as {BASE_ODDS:.0f}:1 good:bad odds, "
        f"{PDO:.0f} points to double the odds (factor {FACTOR:.2f}, offset {OFFSET:.1f})"
    )

    spread = points.groupby("feature")[points].agg(lambda s: s.max() - s.min())
    print("\nPoints range per feature (widest = most influence on the score):")
    print(spread.sort_values(ascending=False).round(1).to_string())

    shown = points.assign(
        points=points["points"].round().astype(int),
        bad_rate=points["bad_rate"].round(4),
        woe=points["woe"].round(3)
    )[["feature", "bin", "loans", "bad_rate", "woe", "points"]]
    print("\nPoints table (train rows):")
    print(shown.to_string(index=False))

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    shown.to_csv(config.SCORECARD_POINTS_PATH, index=False)

    by_table = score_by_lookup(fitted, pointsm validation)
    by_model = OFFSET - FACTOR * _logit(predict_scorecard(fitted, validation))
    print(
        "\nCheck: largest gap between the points table and the model's own score: "
        f"{np.abs(by_table - by_model).max():.6f} points"
    )
    scored = validation.assign(score=by_model)
    scored["band"] = pd.qcut(scored["score"], q=10, labels=False, duplicates="drop") + 1
    bands = scored.groupby("band").agg(
        rows=(TARGET, "size"), min_score=("score", "min"),
        max_score=("score", "max"), bad_rate=(TARGET, "mean")
    )
    print("\nValidation score bands (band 1 = lowest score = riskiest): ")
    print(bands.round(3).to_string())

if __name__ == "__main__":
    main()
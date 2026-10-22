"""Fast tests on a small synthetic dataset (no need for the real CSV)."""
import numpy as np
import pandas as pd
import pytest

from los import capacity, config
from los.data import check_leakage, clean, split_xy
from los.evaluate import regression_metrics
from los.models import estimators, make_pipeline


@pytest.fixture
def raw():
    rng = np.random.default_rng(0)
    n = 300
    vdate = pd.Timestamp("2012-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    los = rng.integers(1, 12, n)
    df = pd.DataFrame({
        "eid": np.arange(1, n + 1),
        "vdate": vdate.strftime("%m/%d/%Y"),
        "rcount": rng.choice(["0", "1", "2", "3", "4", "5+"], n),
        "gender": rng.choice(["F", "M"], n),
        "secondarydiagnosisnonicd9": rng.integers(0, 10, n),
        "discharged": (vdate + pd.to_timedelta(los, unit="D")).strftime("%m/%d/%Y"),
        "facid": rng.choice(list("ABCDE"), n),
        config.TARGET: los,
    })
    for c in config.CONDITION_FLAGS:
        df[c] = rng.integers(0, 2, n)
    for c in config.LAB_AND_VITALS:
        df[c] = rng.normal(50, 10, n)
    return df


def test_discharged_column_leaks_target(raw):
    assert check_leakage(raw) == 1.0


def test_clean_drops_leakage_and_converts_rcount(raw):
    df = clean(raw)
    assert "discharged" not in df.columns
    assert df["rcount_clean"].max() == 5
    assert set(df["rcount_clean"]) <= set(range(6))
    assert {"admission_month", "admission_day"} <= set(df.columns)


def test_features_exclude_target_and_ids(raw):
    X, y = split_xy(clean(raw))
    assert X.shape[1] == len(config.NUMERIC_FEATURES) + len(config.CATEGORICAL_FEATURES)
    assert not {"eid", "vdate", "discharged", config.TARGET} & set(X.columns)
    assert len(y) == len(X)


@pytest.mark.parametrize("name", ["Linear Regression", "Decision Tree", "Gradient Boosting"])
def test_models_fit_and_predict(raw, name):
    X, y = split_xy(clean(raw))
    pipe = make_pipeline(estimators()[name]).fit(X, y)
    assert pipe.predict(X).shape == (len(X),)


def test_metrics_perfect_prediction():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["MSE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1 and m["within_1_day"] == 1


def test_daily_census_counts_patients_in_bed():
    # patient 1: days 1-3, patient 2: days 2-3 (LOS counted from admission day)
    census = capacity.daily_census(pd.to_datetime(["2012-01-01", "2012-01-02"]), [3, 2])
    assert census["all"].tolist()[:4] == [1, 2, 2, 0]


def test_nurses_required_rounds_up():
    out = capacity.nurses_required(pd.DataFrame({"A": [0, 1, 6, 7]}), beds_per_nurse=6)
    assert out["A"].tolist() == [0, 1, 1, 2]

"""Preprocessing pipeline and the five models compared in Chapter 4.5."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from . import config
from .data import feature_columns

RS = config.RANDOM_STATE


def make_preprocessor() -> ColumnTransformer:
    """Median imputation + scaling for numeric; most-frequent + one-hot for categorical.

    The data has no missing values; the imputers keep the pipeline safe for new
    hospital extracts that might.
    """
    numeric, categorical = feature_columns()
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)],
                             verbose_feature_names_out=False)


def estimators() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RS),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RS, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RS),
        "ANN (MLP 64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                                        solver="adam", max_iter=300, random_state=RS),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])

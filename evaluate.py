"""Regression metrics used throughout the dissertation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        # Share of patients whose LOS is predicted within +/- 1 day:
        # an operationally meaningful accuracy for bed managers.
        "within_1_day": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1)),
    }

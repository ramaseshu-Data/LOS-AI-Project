"""End-to-end pipeline reproducing Chapter 4 of the dissertation, plus two extensions.

Steps
  1. Load and check the data (100,000 x 28), drop the leaking `discharged` column
  2. Exploratory figures (LOS distribution, by gender, by facility, correlations)
  3. 80/20 train/test split, fit Linear Regression, Decision Tree, Random Forest,
     Gradient Boosting and an ANN (MLP 64-32); score MSE, RMSE, MAE, R²
  4. 5-fold cross-validation of Random Forest; GridSearchCV tuning (3-fold)
  5. Operational interpretation: predicted LOS >= 7 days = high bed/staff demand
  Extensions (not in the submitted dissertation):
  6. Permutation feature importance for the tuned Random Forest
  7. Bed-occupancy and nurse-staffing simulation: predicted LOS vs. planning on
     the historical average LOS

Usage (from the repo root):
    python -m los.train                 # full run (~30-40 min on a laptop)
    python -m los.train --skip-tuning   # skip CV + grid search (~5 min)
    python -m los.train --quick         # 10k-row smoke test (~1 min)
(set PYTHONPATH=src first, or `pip install -e .`)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split  # noqa: E402

from . import capacity, config  # noqa: E402
from .data import check_leakage, clean, load_raw, split_xy  # noqa: E402
from .evaluate import regression_metrics  # noqa: E402
from .models import estimators, make_pipeline  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
COLORS = {"Linear Regression": "#4c78a8", "Decision Tree": "#f58518", "Random Forest": "#54a24b",
          "Gradient Boosting": "#b279a2", "ANN (MLP 64-32)": "#e45756",
          "Tuned Random Forest": "#2e7d32"}


def savefig(name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(config.FIG_DIR / name, dpi=130)
    plt.close()


def eda_figures(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df[config.TARGET], bins=17, discrete=True, color="#e07b7b", ax=ax)
    ax.set(title="Distribution of hospital length of stay", xlabel="Length of stay (days)", ylabel="Patients")
    savefig("01_los_distribution.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="gender", y=config.TARGET, data=df, hue="gender", palette="pastel", legend=False, ax=ax)
    ax.set(title="Length of stay by gender", xlabel="Gender", ylabel="Length of stay (days)")
    savefig("02_los_by_gender.png")

    order = sorted(df["facid"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(x="facid", y=config.TARGET, data=df, order=order, hue="facid", hue_order=order,
                palette="pastel", legend=False, ax=ax)
    ax.set(title="Length of stay by facility", xlabel="Facility", ylabel="Length of stay (days)")
    savefig("03_los_by_facility.png")

    cols = config.NUMERIC_FEATURES + [config.TARGET]
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(df[cols].corr(), cmap="vlag", center=0, annot=True, fmt=".2f",
                annot_kws={"fontsize": 6}, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Correlation heatmap of numerical variables")
    savefig("04_correlation_heatmap.png")

    rc = df.groupby("rcount_clean")[config.TARGET].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rc.index.astype(str).str.replace("5", "5+"), rc.values, color="#4c78a8")
    ax.set(title="Mean LOS by readmissions in previous 180 days", xlabel="Readmissions",
           ylabel="Mean length of stay (days)")
    savefig("05_los_by_readmissions.png")


def run(quick: bool = False, skip_tuning: bool = False) -> dict:
    t0 = time.time()
    raw = load_raw()
    leak = check_leakage(raw)
    df = clean(raw)
    if quick:
        df = df.sample(10_000, random_state=config.RANDOM_STATE)
        skip_tuning = True
    print(f"Data: {df.shape[0]:,} rows | discharged = vdate + LOS in {leak:.0%} of rows -> dropped")

    if not quick:
        eda_figures(df)

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    print(f"Train {X_train.shape}  Test {X_test.shape}")

    # ---- Step 3: five models
    results, preds, fitted = {}, {}, {}
    for name, est in estimators().items():
        t = time.time()
        pipe = make_pipeline(est)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            pipe.fit(X_train, y_train)
            converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        pred = pipe.predict(X_test)
        m = regression_metrics(y_test, pred)
        m["fit_seconds"] = round(time.time() - t, 1)
        if not converged:
            m["note"] = "ConvergenceWarning: max_iter reached"
        results[name], preds[name], fitted[name] = m, pred, pipe
        print(f"  {name:20s} MSE={m['MSE']:.4f} RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} "
              f"R2={m['R2']:.4f} ({m['fit_seconds']}s){'  [not converged]' if not converged else ''}")

    # ---- Step 4: cross-validation and tuning of Random Forest
    cv_info = {}
    final_name, final_model = "Random Forest", fitted["Random Forest"]
    if not skip_tuning:
        rf = make_pipeline(estimators()["Random Forest"])
        cv_scores = -cross_val_score(rf, X, y, cv=config.CV_FOLDS, scoring="neg_root_mean_squared_error")
        cv_info["rf_cv_rmse_folds"] = [round(float(s), 4) for s in cv_scores]
        cv_info["rf_cv_rmse_mean"] = round(float(cv_scores.mean()), 4)
        print(f"  RF {config.CV_FOLDS}-fold CV RMSE: {np.round(cv_scores, 4)}  mean={cv_scores.mean():.4f}")

        gs = GridSearchCV(rf, config.RF_PARAM_GRID, cv=config.GRID_CV_FOLDS,
                          scoring="neg_root_mean_squared_error", n_jobs=1)
        gs.fit(X_train, y_train)
        cv_info["grid_best_params"] = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        cv_info["grid_best_cv_rmse"] = round(float(-gs.best_score_), 4)
        final_name, final_model = "Tuned Random Forest", gs.best_estimator_
        pred = final_model.predict(X_test)
        results[final_name] = regression_metrics(y_test, pred)
        preds[final_name] = pred
        print(f"  Grid search best {cv_info['grid_best_params']} CV RMSE={-gs.best_score_:.4f} "
              f"-> test RMSE={results[final_name]['RMSE']:.4f} R2={results[final_name]['R2']:.4f}")

    table = pd.DataFrame(results).T
    cols = ["MSE", "RMSE", "MAE", "R2", "within_1_day"]
    table[cols] = table[cols].astype(float)
    if quick:
        print(table[cols].round(4))
        return {"metrics": table[cols].round(4).to_dict(orient="index")}

    config.REPORTS_DIR.mkdir(exist_ok=True)
    table.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv")

    # ---- figures: model comparison and actual vs predicted
    order = table.sort_values("RMSE", ascending=False).index
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, col, title in [(axes[0], "RMSE", "Test RMSE (days), lower is better"),
                           (axes[1], "R2", "Test R², higher is better")]:
        ax.barh(order, table.loc[order, col], color=[COLORS.get(n, "#888") for n in order])
        for i, n in enumerate(order):
            ax.text(table.loc[n, col], i, f" {table.loc[n, col]:.4f}", va="center", fontsize=9)
        ax.set(title=title, xlabel=col)
        ax.set_xlim(0, table[col].max() * 1.18)
    savefig("06_model_comparison.png")

    fp = preds[final_name]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.boxplot(x=y_test.to_numpy(), y=fp, ax=axes[0], color="#8fd19e", showfliers=False)
    axes[0].plot([0, 16], [1, 17], ls="--", c="k", lw=1, label="perfect prediction")
    axes[0].set(title=f"{final_name}: predicted vs actual LOS", xlabel="Actual LOS (days)",
                ylabel="Predicted LOS (days)")
    axes[0].legend(loc="upper left")
    sns.histplot(y_test.to_numpy() - fp, bins=60, ax=axes[1], color="#4c78a8")
    axes[1].set(title="Prediction errors (actual − predicted)", xlabel="Days", ylabel="Patients")
    savefig("07_actual_vs_predicted.png")

    # ---- Step 5: operational interpretation used in the notebook
    demand = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": fp})
    demand["predicted_high"] = demand["predicted"] >= config.HIGH_DEMAND_LOS_DAYS
    demand["actual_high"] = demand["actual"] >= config.HIGH_DEMAND_LOS_DAYS
    tp = int((demand.predicted_high & demand.actual_high).sum())
    demand_summary = {
        "threshold_days": config.HIGH_DEMAND_LOS_DAYS,
        "flagged_high_demand": int(demand.predicted_high.sum()),
        "actual_long_stays": int(demand.actual_high.sum()),
        "precision": round(tp / max(int(demand.predicted_high.sum()), 1), 4),
        "recall": round(tp / max(int(demand.actual_high.sum()), 1), 4),
    }
    print(f"  High-demand flag (>= {config.HIGH_DEMAND_LOS_DAYS} days): {demand_summary}")

    # ---- Extension 6: permutation importance
    sample = X_test.sample(5_000, random_state=config.RANDOM_STATE)
    pi = permutation_importance(final_model, sample, y_test.loc[sample.index], n_repeats=5,
                                random_state=config.RANDOM_STATE, scoring="neg_root_mean_squared_error",
                                n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=sample.columns).sort_values()
    imp.round(4).to_csv(config.REPORTS_DIR / "feature_importance.csv", header=["rmse_increase_days"])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    imp.tail(15).plot.barh(ax=ax, color="#54a24b")
    ax.set(title=f"What drives the prediction? ({final_name})",
           xlabel="Increase in RMSE (days) when the feature is shuffled")
    savefig("08_feature_importance.png")

    # ---- Extension 7: bed occupancy and staffing simulation on the test admissions
    test_rows = df.loc[X_test.index]
    hist_avg = df.loc[X_train.index].groupby("facid")[config.TARGET].mean()
    actual = capacity.daily_census(test_rows.vdate, y_test, test_rows.facid)
    forecasts = {
        "Historical average LOS (per facility)": capacity.daily_census(
            test_rows.vdate, test_rows.facid.map(hist_avg), test_rows.facid),
        "Linear Regression": capacity.daily_census(test_rows.vdate, preds["Linear Regression"], test_rows.facid),
        final_name: capacity.daily_census(test_rows.vdate, fp, test_rows.facid),
    }
    cap = capacity.compare_census(actual, forecasts)
    cap.round(3).to_csv(config.REPORTS_DIR / "capacity_staffing.csv")
    print("  Bed/staffing simulation:\n", cap.round(2).to_string())

    fac = "A"
    window = actual.index[60:150]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(window, actual.loc[window, fac], label="Actual occupied beds", c="k", lw=2)
    ax.plot(window, forecasts["Historical average LOS (per facility)"].loc[window, fac],
            label="Planned with historical average LOS", c="#9aa0a6", ls="--")
    ax.plot(window, forecasts[final_name].loc[window, fac], label=f"Planned with {final_name}",
            c=COLORS["Tuned Random Forest"])
    ax.set(title=f"Daily bed occupancy, facility {fac} (test-set admissions)", ylabel="Occupied beds")
    ax.legend(loc="lower right")
    savefig("09_bed_occupancy_simulation.png")

    summary = {
        "rows": int(len(df)), "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "features": list(X.columns), "discharged_leakage_share": leak,
        "metrics": json.loads(table.round(4).to_json(orient="index")),
        "random_forest_validation": cv_info, "final_model": final_name,
        "high_demand_flag": demand_summary,
        "capacity_staffing": cap.round(3).to_dict(orient="index"),
        "beds_per_nurse_assumption": config.BEDS_PER_NURSE,
        "runtime_minutes": round((time.time() - t0) / 60, 1),
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDone in {summary['runtime_minutes']} min -> {config.REPORTS_DIR}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="10k-row smoke test")
    ap.add_argument("--skip-tuning", action="store_true", help="skip RF cross-validation and grid search")
    a = ap.parse_args()
    run(quick=a.quick, skip_tuning=a.skip_tuning)

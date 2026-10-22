"""Loading, checks and cleaning of the hospital length-of-stay dataset (Chapter 4.2–4.3)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

EXPECTED_COLUMNS = {
    "eid", "vdate", "rcount", "gender", "secondarydiagnosisnonicd9",
    "discharged", "facid", config.TARGET,
    *config.CONDITION_FLAGS, *config.LAB_AND_VITALS,
}


def load_raw(path: Path | str = config.RAW_CSV) -> pd.DataFrame:
    """Read the CSV and confirm the expected 28 columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md.")
    df = pd.read_csv(path)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")
    return df


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per column: dtype, missing values, unique values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    })


def check_leakage(df: pd.DataFrame) -> float:
    """Share of rows where discharged - vdate equals the target (1.0 = full leak)."""
    days = (pd.to_datetime(df["discharged"]) - pd.to_datetime(df["vdate"])).dt.days
    return float((days == df[config.TARGET]).mean())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing steps from Table 4.3 of the dissertation.

    - convert vdate to a date and derive admission_month / admission_day
    - drop `discharged` (it equals vdate + LOS, so keeping it leaks the target)
    - convert rcount ('0'..'4', '5+') to the numeric rcount_clean
    """
    out = df.copy()
    out["vdate"] = pd.to_datetime(out["vdate"])
    out["admission_month"] = out["vdate"].dt.month
    out["admission_day"] = out["vdate"].dt.day
    out = out.drop(columns="discharged")
    out["rcount_clean"] = out["rcount"].astype(str).str.replace("+", "", regex=False).astype(int)
    out = out.dropna(subset=[config.TARGET])
    return out


def feature_columns() -> tuple[list[str], list[str]]:
    numeric = list(config.NUMERIC_FEATURES)
    if config.USE_ADMISSION_DATE_FEATURES:
        numeric += config.DATE_FEATURES
    return numeric, list(config.CATEGORICAL_FEATURES)


def split_xy(df: pd.DataFrame):
    numeric, categorical = feature_columns()
    return df[numeric + categorical], df[config.TARGET]

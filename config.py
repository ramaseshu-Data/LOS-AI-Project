"""Central configuration: paths, feature lists and run settings.

Settings mirror the dissertation notebook (notebooks/LOS_Analysis.ipynb) so the
pipeline reproduces the results reported in Chapter 4.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "LengthOfStay.csv"
FACILITIES_CSV = DATA_DIR / "MetaData_Facilities.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

TARGET = "lengthofstay"
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80/20 split -> 80,000 train / 20,000 test

CONDITION_FLAGS = [
    "dialysisrenalendstage", "asthma", "irondef", "pneum", "substancedependence",
    "psychologicaldisordermajor", "depress", "psychother", "fibrosisandother",
    "malnutrition", "hemo",
]
LAB_AND_VITALS = [
    "hematocrit", "neutrophils", "sodium", "glucose", "bloodureanitro",
    "creatinine", "bmi", "pulse", "respiration",
]

# Model inputs: 22 numeric + 2 categorical (as used in the dissertation)
NUMERIC_FEATURES = CONDITION_FLAGS + LAB_AND_VITALS + ["secondarydiagnosisnonicd9", "rcount_clean"]
CATEGORICAL_FEATURES = ["gender", "facid"]

# admission_month / admission_day are engineered from the visit date, as in the
# notebook. In the original notebook they were created but were not picked up by
# `select_dtypes(['int64','float64'])` (pandas stores them as int32), so the
# reported models did not use them. Set True to include them.
USE_ADMISSION_DATE_FEATURES = False
DATE_FEATURES = ["admission_month", "admission_day"]

# Validation and tuning (Chapter 3.7.2 / 4.5)
CV_FOLDS = 5
GRID_CV_FOLDS = 3
RF_PARAM_GRID = {
    "model__n_estimators": [50, 100],
    "model__max_depth": [10, 20, None],
    "model__min_samples_split": [2, 5],
}

# Operational extension: illustrative safe-staffing ratio used to turn bed
# occupancy into nurses per day. Change to match local policy.
BEDS_PER_NURSE = 6
# Notebook's operational rule: predicted LOS >= 7 days = high bed/staff demand
HIGH_DEMAND_LOS_DAYS = 7

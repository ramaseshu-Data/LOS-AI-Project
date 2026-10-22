# Data

| File | What it is |
|---|---|
| `LengthOfStay.csv` | 100,000 hospital admissions × 28 columns; target column `lengthofstay` (days, 1–17) |
| `MetaData_Facilities.csv` | Names of the five facilities (`facid` A–E) |

**Source.** Microsoft's public hospital length-of-stay sample data, published in
[microsoft/r-server-hospital-length-of-stay](https://github.com/microsoft/r-server-hospital-length-of-stay)
and mirrored on Kaggle as
[Hospital Length of Stay Dataset (Microsoft)](https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft).
Microsoft notes the data was generated from a third-party dataset by resampling, so it
contains no real patient records.

## Columns

| Group | Columns |
|---|---|
| Identifiers / dates | `eid`, `vdate` (admission date), `discharged` (discharge date) |
| Demographic | `gender` |
| Readmissions | `rcount` — readmissions in the previous 180 days (`0`–`4`, `5+`) |
| Comorbidity flags (0/1) | `dialysisrenalendstage`, `asthma`, `irondef`, `pneum`, `substancedependence`, `psychologicaldisordermajor`, `depress`, `psychother`, `fibrosisandother`, `malnutrition`, `hemo` |
| Labs | `hematocrit`, `neutrophils`, `sodium`, `glucose`, `bloodureanitro`, `creatinine` |
| Vitals / body | `bmi`, `pulse`, `respiration` |
| Other | `secondarydiagnosisnonicd9` (count), `facid` (facility A–E) |
| Target | `lengthofstay` |

> ⚠️ `discharged` equals `vdate + lengthofstay` for every row, so it is removed before
> modelling — keeping it would leak the answer into the model.

To re-download: `python scripts/download_data.py`.

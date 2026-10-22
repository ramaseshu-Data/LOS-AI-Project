"""Turn LOS predictions into bed-occupancy and staffing estimates.

This is the operational link the dissertation argues for: a LOS prediction is
only useful to a bed manager once it becomes "how many beds will be occupied on
each day, and how many nurses does that need?".

For a set of admissions with a visit date and a length of stay, the daily
census of a ward is the number of patients in a bed on that day:
    patient i occupies a bed on days [vdate_i, vdate_i + LOS_i)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def daily_census(vdate: pd.Series, los, facid: pd.Series | None = None) -> pd.DataFrame:
    """Daily occupied beds, optionally per facility.

    Returns a DataFrame indexed by date with one column per facility
    (or a single 'all' column).
    """
    vdate = pd.to_datetime(pd.Series(vdate).reset_index(drop=True))
    los_days = np.maximum(np.rint(np.asarray(los, dtype=float)), 1).astype(int)
    groups = (pd.Series(facid).reset_index(drop=True) if facid is not None
              else pd.Series(["all"] * len(vdate)))

    start = vdate.min()
    end = (vdate + pd.to_timedelta(los_days, unit="D")).max()
    days = pd.date_range(start, end, freq="D")
    offset = (vdate - start).dt.days.to_numpy()

    out = {}
    for g in sorted(groups.unique()):
        mask = (groups == g).to_numpy()
        # difference-array trick: +1 on admission day, -1 on discharge day
        diff = np.zeros(len(days) + 1, dtype=int)
        np.add.at(diff, offset[mask], 1)
        np.add.at(diff, offset[mask] + los_days[mask], -1)
        out[g] = np.cumsum(diff)[: len(days)]
    return pd.DataFrame(out, index=days)


def nurses_required(census: pd.DataFrame, beds_per_nurse: int = config.BEDS_PER_NURSE) -> pd.DataFrame:
    """Nurses needed per day/ward for a given beds-per-nurse ratio (rounded up)."""
    return np.ceil(census / beds_per_nurse).astype(int)


def compare_census(actual: pd.DataFrame, forecasts: dict[str, pd.DataFrame],
                   trim_days: int = 20) -> pd.DataFrame:
    """MAE of daily census and of nurse requirement for each forecast method.

    The first and last `trim_days` are excluded: patients admitted before the
    data window (or discharged after it) are missing there, so census is
    artificially low at the edges for every method.
    """
    idx = actual.index[trim_days:-trim_days]
    a = actual.loc[idx]
    rows = []
    for name, f in forecasts.items():
        f = f.reindex(index=idx, columns=a.columns, fill_value=0)
        census_err = (f - a).abs()
        staff_err = (nurses_required(f) - nurses_required(a)).abs()
        rows.append({
            "method": name,
            "census_MAE_beds": float(census_err.to_numpy().mean()),
            "census_max_error_beds": int(census_err.to_numpy().max()),
            "nurse_MAE_per_ward_day": float(staff_err.to_numpy().mean()),
            "ward_days_staffing_wrong_pct": float((staff_err > 0).to_numpy().mean() * 100),
        })
    return pd.DataFrame(rows).set_index("method")

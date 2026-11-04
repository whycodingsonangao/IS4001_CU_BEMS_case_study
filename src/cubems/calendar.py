"""Calendar features: official Thai holidays, day type, season, analysis window.

2019 holidays come from the cited list in references/thai_holidays_2019.csv (only rows with
use_as_holiday=True are flagged). 2018 (optional check only) falls back to the `holidays` package,
minus Labour Day, which is not a government holiday.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config


@lru_cache(maxsize=1)
def holiday_table() -> pd.DataFrame:
    t = pd.read_csv(config.HOLIDAYS_PATH, parse_dates=["date"])
    t["use_as_holiday"] = t["use_as_holiday"].astype(str).str.lower().eq("true")
    return t


@lru_cache(maxsize=1)
def holiday_dates() -> frozenset[pd.Timestamp]:
    t = holiday_table()
    dates = set(t.loc[t["use_as_holiday"], "date"])
    try:
        import holidays

        for d, name in holidays.country_holidays("TH", years=[2018]).items():
            if "Labor" not in name and "Labour" not in name:
                dates.add(pd.Timestamp(d))
    except ImportError:  # pragma: no cover
        pass
    return frozenset(dates)


def add_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year.astype("int16")
    df["month"] = d.dt.month.astype("int8")
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["day_name"] = d.dt.day_name()
    df["is_weekend"] = df["dow"] >= 5
    df["season"] = df["month"].map(config.SEASON_BY_MONTH)
    df["is_holiday"] = d.dt.normalize().isin(holiday_dates())
    df["day_type"] = np.where(df["is_weekend"] | df["is_holiday"], "Weekend/holiday", "Workday")
    df["half"] = np.where(df["month"] <= 6, "H1", "H2")
    return df

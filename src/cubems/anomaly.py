"""Retrospective daily AC deviations for facility review; not confirmed equipment faults."""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_exceptions(scored: pd.DataFrame) -> pd.DataFrame:
    """Mark complete zone-days outside a development-calibrated expected range."""
    s = scored.copy()
    above = s["y"] - s["hi"]
    below = s["lo"] - s["y"]
    s["departure_kwh"] = np.where(above > 0, above, np.where(below > 0, -below, 0.0))
    s["flag"] = s["departure_kwh"] != 0
    s["direction"] = np.select([s["departure_kwh"] > 0, s["departure_kwh"] < 0],
                                ["Above expected", "Below expected"], default="Within range")
    return s


def daily_kpis(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-zone exception-day rate over complete monitored days."""
    k = scored.groupby(["zone_id", "floor", "zone"]).agg(
        monitored_days=("date", "size"), exception_days=("flag", "sum"),
        total_departure_kwh=("departure_kwh", lambda s: s.abs().sum())).reset_index()
    k["exception_rate"] = 100 * k["exception_days"] / k["monitored_days"]
    return k.sort_values(["exception_rate", "total_departure_kwh"], ascending=False)

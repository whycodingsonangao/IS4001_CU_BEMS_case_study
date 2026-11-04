"""Minute -> hourly -> zone-day aggregation with an explicit quality register.

Conventions
-----------
* Energy (kWh) for an hour = sum of valid per-minute kW readings / 60.
  Missing minutes contribute nothing and are counted, never filled with zero.
* ``joint_minutes`` = minutes where *every* power channel of the zone is present.
  That is the coverage used by the 95 % zone-day usability rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .loader import Channel, load_floor


def zone_group(floor: int, zone: int) -> str:
    if zone in config.MODEL1_ZONES.get(floor, []):
        return "Model 1"
    if zone in config.MODEL2_ZONES.get(floor, []):
        return "Model 2"
    return "Out of scope"


def _zone_minutes(df: pd.DataFrame, chans: list[Channel]) -> pd.DataFrame:
    """Collapse one zone's raw channels into end-use kW plus validity flags."""
    out = pd.DataFrame(index=df.index)
    power = [c for c in chans if c.unit == "kW"]
    for use in config.END_USES:
        cols = [c.column for c in power if c.kind == use]
        if not cols:
            continue
        block = df[cols]
        # all sub-meters of an end use must be present, otherwise the minute is missing
        out[use] = block.sum(axis=1, min_count=len(cols)).where(block.notna().all(axis=1))
    for kind in ("temp", "rh", "lux"):
        cols = [c.column for c in chans if c.kind == kind]
        if cols:
            out[kind] = df[cols].mean(axis=1)
    power_cols = [c.column for c in power]
    out["joint_valid"] = df[power_cols].notna().all(axis=1)
    out["extreme"] = (df[power_cols] > config.EXTREME_KW).any(axis=1)
    return out


def _to_hourly(m: pd.DataFrame) -> pd.DataFrame:
    g = m.resample("h")
    h = pd.DataFrame(index=g.size().index)
    for use in config.END_USES:
        if use in m:
            h[f"{use.lower()}_kwh"] = g[use].sum(min_count=1) / 60.0
            h[f"{use.lower()}_minutes"] = g[use].count()
    for kind in ("temp", "rh", "lux"):
        if kind in m:
            h[f"{kind}_mean"] = g[kind].mean()
            h[f"{kind}_minutes"] = g[kind].count()
    h["joint_minutes"] = g["joint_valid"].sum()
    h["extreme_minutes"] = g["extreme"].sum()
    return h


def build_floor(year: int, floor: int) -> tuple[pd.DataFrame, list[dict]]:
    df, chans = load_floor(year, floor)
    frames, zones = [], []
    for zone in sorted({c.zone for c in chans}):
        zc = [c for c in chans if c.zone == zone]
        h = _to_hourly(_zone_minutes(df, zc)).reset_index()
        h.insert(0, "zone", zone)
        h.insert(0, "floor", floor)
        frames.append(h)
        zones.append(
            {
                "year": year,
                "floor": floor,
                "zone": zone,
                "zone_id": f"F{floor}Z{zone}",
                "group": zone_group(floor, zone),
                "n_ac_units": sum(c.kind == "AC" for c in zc),
                "has_ac": any(c.kind == "AC" for c in zc),
                "has_env": any(c.kind == "temp" for c in zc),
                "channels": ", ".join(c.name for c in zc if c.unit == "kW"),
            }
        )
    return pd.concat(frames, ignore_index=True), zones


def add_calendar(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    d = pd.to_datetime(df[date_col])
    df["year"] = d.dt.year.astype("int16")
    df["month"] = d.dt.month.astype("int8")
    df["dow"] = d.dt.dayofweek.astype("int8")
    df["day_name"] = d.dt.day_name()
    df["is_weekend"] = df["dow"] >= 5
    df["season"] = df["month"].map(config.SEASON_BY_MONTH)
    try:
        import holidays

        th = holidays.country_holidays("TH", years=sorted(d.dt.year.unique().tolist()))
        df["is_holiday"] = d.dt.date.map(lambda x: x in th)
    except ImportError:  # pragma: no cover
        df["is_holiday"] = False
    df["day_type"] = np.where(df["is_weekend"] | df["is_holiday"], "Weekend/holiday", "Workday")

    def window(x: pd.Timestamp) -> str:
        if x < pd.Timestamp(config.QA_WINDOW[0]):
            return "2018 history"
        if x <= pd.Timestamp(config.QB_BASELINE[1]):
            return "Baseline (Jan-Aug)"
        return "Holdout (Sep-Dec)"

    df["window"] = d.map(window)
    return df


def to_zone_day(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["timestamp"].dt.normalize()
    sum_cols = [c for c in h.columns if c.endswith(("_kwh", "_minutes"))]
    agg = {c: (c, "sum") for c in sum_cols}
    for kind in ("temp", "rh"):
        if f"{kind}_mean" in h:
            agg[f"{kind}_mean"] = (f"{kind}_mean", "mean")
    agg["valid_hours"] = ("joint_minutes", lambda s: int((s >= config.MIN_HOUR_MINUTES).sum()))
    zd = h.groupby(["floor", "zone", "date"], as_index=False).agg(**agg)
    zd["coverage"] = zd["joint_minutes"] / config.MINUTES_PER_DAY
    zd["usable"] = zd["coverage"] >= config.MIN_DAY_COVERAGE
    zd["env_coverage"] = zd.get("temp_minutes", 0) / config.MINUTES_PER_DAY
    kwh = [c for c in ("ac_kwh", "light_kwh", "plug_kwh") if c in zd]
    zd["total_kwh"] = zd[kwh].sum(axis=1, min_count=1)
    zd["zone_id"] = "F" + zd["floor"].astype(str) + "Z" + zd["zone"].astype(str)
    zd["group"] = [zone_group(f, z) for f, z in zip(zd["floor"], zd["zone"], strict=True)]
    return add_calendar(zd)

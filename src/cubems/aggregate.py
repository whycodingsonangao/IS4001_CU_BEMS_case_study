"""Minute -> hourly -> zone-day aggregation (methodology plan, section 3).

Conventions
-----------
* A minute is valid for an end use only if every circuit of that end use has a reading.
* ``{use}_kwh`` for an hour = mean of its valid per-minute kW x 1 h. ``{use}_minutes`` counts the valid
  minutes. An hour is *valid* only when minutes >= the threshold (57 by default, 60 for the strict
  sensitivity run); use :func:`valid_kwh` to apply it. Invalid hours are missing, never zero.
* A zone-day is *complete* when all 24 hours are valid for every end use of its group.
"""

from __future__ import annotations

import pandas as pd

from . import config
from .calendar import add_calendar
from .loader import Channel, load_floor


def zone_group(floor: int, zone: int) -> str:
    if zone in config.C1_ZONES.get(floor, []):
        return "C1"
    if zone in config.C2_ZONES.get(floor, []):
        return "C2"
    if zone in config.LIGHT_ONLY_ZONES.get(floor, []):
        return "Lighting only"
    return "Unassigned"


def zone_id(floor, zone) -> str:
    return f"F{int(floor)}Z{int(zone)}"


def _zone_minutes(df: pd.DataFrame, chans: list[Channel]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    power = [c for c in chans if c.unit == "kW"]
    for use in config.END_USES:
        cols = [c.column for c in power if c.kind == use]
        if cols:
            block = df[cols]
            out[use] = block.sum(axis=1, min_count=len(cols)).where(block.notna().all(axis=1))
    for kind in ("temp", "rh", "lux"):
        cols = [c.column for c in chans if c.kind == kind]
        if cols:
            out[kind] = df[cols].mean(axis=1)
    out["extreme"] = (df[[c.column for c in power]] > config.EXTREME_KW).any(axis=1)
    return out


def _to_hourly(m: pd.DataFrame) -> pd.DataFrame:
    g = m.resample("h")
    h = pd.DataFrame(index=g.size().index)
    for use in config.END_USES:
        if use in m:
            h[f"{use.lower()}_kwh"] = g[use].mean()  # mean kW x 1 h
            h[f"{use.lower()}_minutes"] = g[use].count()
    for kind in ("temp", "rh", "lux"):
        if kind in m:
            h[f"{kind}_mean"] = g[kind].mean()
            h[f"{kind}_minutes"] = g[kind].count()
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
        zones.append({
            "year": year, "floor": floor, "zone": zone, "zone_id": zone_id(floor, zone),
            "group": zone_group(floor, zone),
            "n_ac_units": sum(c.kind == "AC" for c in zc),
            "has_ac": any(c.kind == "AC" for c in zc),
            "has_env": any(c.kind == "temp" for c in zc),
            "channels": ", ".join(c.name for c in zc if c.unit == "kW"),
        })
    return pd.concat(frames, ignore_index=True), zones


def valid_kwh(h: pd.DataFrame, use: str, min_minutes: int = config.MIN_HOUR_MINUTES) -> pd.Series:
    """Hourly kWh for one end use, NaN where the hour is not valid."""
    use = use.lower()
    return h[f"{use}_kwh"].where(h[f"{use}_minutes"] >= min_minutes)


def to_zone_day(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per zone-day with kWh over valid hours and completeness flags at 57 and 60 minutes."""
    h = hourly.copy()
    h["date"] = h["timestamp"].dt.normalize()
    h["group"] = [zone_group(f, z) for f, z in zip(h["floor"], h["zone"], strict=True)]
    for use in config.END_USES:
        u = use.lower()
        if f"{u}_kwh" in h:
            h[f"{u}_valid"] = h[f"{u}_minutes"] >= config.MIN_HOUR_MINUTES
            h[f"{u}_valid_strict"] = h[f"{u}_minutes"] >= config.STRICT_HOUR_MINUTES
            h[f"{u}_kwh_v"] = valid_kwh(h, u)

    agg: dict = {}
    for use in config.END_USES:
        u = use.lower()
        if f"{u}_kwh" in h:
            agg[f"{u}_kwh"] = (f"{u}_kwh_v", lambda s: s.sum(min_count=1))
            agg[f"{u}_hours"] = (f"{u}_valid", "sum")
            agg[f"{u}_hours_strict"] = (f"{u}_valid_strict", "sum")
            agg[f"{u}_minutes"] = (f"{u}_minutes", "sum")
    for kind in ("temp", "rh"):
        if f"{kind}_mean" in h:
            agg[f"{kind}_mean"] = (f"{kind}_mean", "mean")
            agg[f"{kind}_minutes"] = (f"{kind}_minutes", "sum")
    agg["extreme_minutes"] = ("extreme_minutes", "sum")
    agg["n_hours"] = ("timestamp", "size")
    zd = h.groupby(["floor", "zone", "group", "date"], as_index=False).agg(**agg)

    def complete(row_uses: list[str], strict: bool) -> pd.Series:
        suf = "_hours_strict" if strict else "_hours"
        ok = pd.Series(True, index=zd.index)
        for u in row_uses:
            col = f"{u.lower()}{suf}"
            ok &= zd[col].fillna(0).eq(24) if col in zd else False
        return ok

    zd["usable"] = False
    zd["usable_strict"] = False
    for grp, uses in config.GROUP_USES.items():
        m = zd["group"] == grp
        zd.loc[m, "usable"] = complete(uses, strict=False)[m]
        zd.loc[m, "usable_strict"] = complete(uses, strict=True)[m]
    kwh = [c for c in ("ac_kwh", "light_kwh", "plug_kwh") if c in zd]
    zd["total_kwh"] = zd[kwh].sum(axis=1, min_count=1)
    zd["env_coverage"] = zd.get("temp_minutes", 0) / config.MINUTES_PER_DAY
    zd["zone_id"] = "F" + zd["floor"].astype(str) + "Z" + zd["zone"].astype(str)
    return add_calendar(zd)

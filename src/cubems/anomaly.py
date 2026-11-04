"""Question B: empirical baseline bands, sustained-episode detection and KPIs.

This is the *preliminary* method from the proposal (historical 5th-95th percentile band
per zone x hour x day-type, candidate = sustained departure). The weather-aware
expected-consumption model suggested in feedback should replace the band's centre
later; ``detect_episodes`` works on any (observed, lo, hi) triple, so it can be reused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def hourly_kw(hourly: pd.DataFrame, zone_day: pd.DataFrame, use: str = "ac") -> pd.DataFrame:
    """Hourly mean kW for valid hours on usable zone-days only."""
    h = hourly[["floor", "zone", "timestamp", f"{use}_kwh", f"{use}_minutes", "joint_minutes"]]
    h = h.dropna(subset=[f"{use}_kwh"]).copy()
    h = h[(h[f"{use}_minutes"] >= config.MIN_HOUR_MINUTES)
          & (h["joint_minutes"] >= config.MIN_HOUR_MINUTES)]
    h["kw"] = h[f"{use}_kwh"] * 60.0 / h[f"{use}_minutes"]
    h["date"] = h["timestamp"].dt.normalize()
    h["hour"] = h["timestamp"].dt.hour
    meta = zone_day.loc[zone_day["usable"], ["floor", "zone", "date", "day_type", "window",
                                              "coverage"]]
    return h.merge(meta, on=["floor", "zone", "date"], how="inner")


def baseline_bands(
    hkw: pd.DataFrame,
    start: str = config.QB_BASELINE[0],
    end: str = config.QB_BASELINE[1],
    q: tuple[float, float] = (0.05, 0.95),
    min_days: int = 10,
    by: tuple[str, ...] = ("floor", "zone", "day_type", "hour"),
) -> pd.DataFrame:
    base = hkw[hkw["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    g = base.groupby(list(by))["kw"]
    bands = pd.DataFrame({
        "lo": g.quantile(q[0]),
        "median": g.median(),
        "hi": g.quantile(q[1]),
        "n_days": g.size(),
    }).reset_index()
    return bands[bands["n_days"] >= min_days]


def detect_episodes(
    scored: pd.DataFrame,
    margin_kw: float = 0.25,
    min_hours: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag hours outside [lo - margin, hi + margin]; group consecutive flagged hours.

    ``scored`` needs columns floor, zone, timestamp, kw, lo, hi, coverage.
    Returns (scored with ``departure``/``flag``, episodes table).
    """
    s = scored.sort_values(["floor", "zone", "timestamp"]).copy()
    above = s["kw"] - (s["hi"] + margin_kw)
    below = (s["lo"] - margin_kw) - s["kw"]
    s["departure"] = np.where(above > 0, above, np.where(below > 0, -below, 0.0))
    s["flag"] = s["departure"] != 0
    s["direction"] = np.sign(s["departure"])

    # new run whenever zone changes, flag/direction changes or an hour is skipped
    key = s["floor"].astype(str) + "-" + s["zone"].astype(str)
    gap = s["timestamp"].diff() != pd.Timedelta(hours=1)
    brk = (key != key.shift()) | gap | (s["direction"] != s["direction"].shift())
    s["run"] = brk.cumsum()

    f = s[s["flag"]]
    ep = f.groupby("run").agg(
        floor=("floor", "first"),
        zone=("zone", "first"),
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        duration_h=("timestamp", "size"),
        peak_departure_kw=("departure", lambda x: x.abs().max()),
        mean_departure_kw=("departure", lambda x: x.abs().mean()),
        direction=("direction", "first"),
        min_coverage=("coverage", "min"),
    )
    ep = ep[ep["duration_h"] >= min_hours].reset_index(drop=True)
    ep["end"] = ep["end"] + pd.Timedelta(hours=1)
    ep["direction"] = ep["direction"].map({1.0: "Above normal", -1.0: "Below normal"})
    ep["zone_id"] = "F" + ep["floor"].astype(str) + "Z" + ep["zone"].astype(str)
    ep = ep.sort_values(["duration_h", "peak_departure_kw"], ascending=False)
    return s, ep


def kpis(scored: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    """Event Rate and Duration Burden per zone (both per 100 valid monitored hours)."""
    monitored = scored.groupby(["floor", "zone"]).size().rename("valid_hours")
    n_ep = episodes.groupby(["floor", "zone"]).size().rename("episodes")
    dev_h = episodes.groupby(["floor", "zone"])["duration_h"].sum().rename("deviation_hours")
    k = pd.concat([monitored, n_ep, dev_h], axis=1).fillna(0).reset_index()
    k["event_rate"] = 100 * k["episodes"] / k["valid_hours"]
    k["duration_burden"] = 100 * k["deviation_hours"] / k["valid_hours"]
    k["zone_id"] = "F" + k["floor"].astype(int).astype(str) + "Z" + k["zone"].astype(int).astype(str)
    return k.sort_values("duration_burden", ascending=False)

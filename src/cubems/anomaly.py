"""Exception layer on top of the Question B forecast.

An exception is a sustained run of hours where actual AC use falls outside the forecast's expected
range (prediction + locked development-residual quantiles), i.e. "unusual given the weather and recent
history". Candidates for a person to check, not confirmed faults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_episodes(scored: pd.DataFrame, margin_kw: float = 0.25, min_hours: int = 2,
                    value: str = "kw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag hours outside [lo - margin, hi + margin]; group consecutive flagged hours per zone.

    ``scored`` needs floor, zone, timestamp, lo, hi and the ``value`` column (hourly kWh = mean kW).
    """
    s = scored.sort_values(["floor", "zone", "timestamp"]).copy()
    v = s[value]
    above = v - (s["hi"] + margin_kw)
    below = (s["lo"] - margin_kw) - v
    s["departure"] = np.where(above > 0, above, np.where(below > 0, -below, 0.0))
    s["flag"] = s["departure"] != 0
    s["direction"] = np.sign(s["departure"])

    key = s["floor"].astype(str) + "-" + s["zone"].astype(str)
    gap = s["timestamp"].diff() != pd.Timedelta(hours=1)
    brk = (key != key.shift()) | gap | (s["direction"] != s["direction"].shift())
    s["run"] = brk.cumsum()

    f = s[s["flag"]]
    ep = f.groupby("run").agg(
        floor=("floor", "first"), zone=("zone", "first"),
        start=("timestamp", "min"), end=("timestamp", "max"),
        duration_h=("timestamp", "size"),
        peak_departure_kw=("departure", lambda x: x.abs().max()),
        mean_departure_kw=("departure", lambda x: x.abs().mean()),
        direction=("direction", "first"),
    )
    ep = ep[ep["duration_h"] >= min_hours].reset_index(drop=True)
    ep["end"] = ep["end"] + pd.Timedelta(hours=1)
    ep["direction"] = ep["direction"].map({1.0: "Above expected", -1.0: "Below expected"})
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

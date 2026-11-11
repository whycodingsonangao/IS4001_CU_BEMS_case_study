"""Per-channel screening of minute readings: spikes and flatlined (stuck) sensors.

Both checks compare a channel with its own behaviour, not with one building-wide limit:

* **Spike:** a power reading above ``SPIKE_RATIO`` x the channel's 99th percentile of *active* readings
  (> ``SPIKE_ACTIVE_KW``) and above ``SPIKE_MIN_KW``. Using active readings keeps rarely-used AC units,
  which are off most of the year, from having their normal running level flagged.
* **Flatline:** the same value repeated for at least ``FLATLINE_MINUTES[kind]`` consecutive minutes. Power
  channels are only checked above ``FLATLINE_MIN_KW``, because idle/standby readings (0.01-0.3 kW at the
  sensor's 0.01 kW resolution) are legitimately constant for hours. Lux is not checked: dark rooms read a
  constant value all night, and lux is not used in any analysis.

Flagged minutes are set to missing before aggregation, so the existing 57/60 hour rule treats them like any
other gap. Thresholds use each file's full year of readings (a cleaning step, not a model input).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .loader import Channel

OK, SPIKE, FLAT = 0, 1, 2
CHECK_NAMES = {SPIKE: "Spike", FLAT: "Flatline"}


def _runs(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run id and run length for each position; NaN never continues a run."""
    same = np.zeros(len(v), dtype=bool)
    same[1:] = v[1:] == v[:-1]  # NaN == NaN is False
    rid = np.cumsum(~same) - 1
    return rid, np.bincount(rid)[rid]


def spike_threshold(s: pd.Series) -> tuple[float, float]:
    """(reference kW, threshold kW) for one power channel."""
    active = s[s > config.SPIKE_ACTIVE_KW]
    ref = float(active.quantile(config.SPIKE_REF_PCTL / 100)) if len(active) else 0.0
    return ref, max(config.SPIKE_RATIO * ref, config.SPIKE_MIN_KW)


def screen_channel(s: pd.Series, ch: Channel) -> tuple[np.ndarray, dict]:
    """Per-minute codes (OK / SPIKE / FLAT) and a summary row for one channel."""
    v = s.to_numpy(dtype="float64")
    code = np.zeros(len(v), dtype="int8")
    row: dict = {"kind": ch.kind}

    if ch.unit == "kW":
        ref, thr = spike_threshold(s)
        spike = v > thr
        code[spike] = SPIKE
        row.update(ref_kw=ref, spike_threshold_kw=thr, spike_minutes=int(spike.sum()), max_kw=float(np.nanmax(v))
                   if np.isfinite(v).any() else np.nan)

    limit = config.FLATLINE_MINUTES.get(ch.kind)
    if limit is not None:
        _, length = _runs(v)
        eligible = np.isfinite(v)
        if ch.unit == "kW":
            eligible &= v >= config.FLATLINE_MIN_KW
        flat = eligible & (length >= limit) & (code == OK)
        code[flat] = FLAT
        row.update(flat_threshold_min=limit, longest_run_min=int(length[eligible].max()) if eligible.any() else 0,
                   flat_minutes=int(flat.sum()))
    return code, row


def _episodes(code: np.ndarray, idx: pd.DatetimeIndex, v: np.ndarray) -> list[dict]:
    out = []
    flagged = code != OK
    if not flagged.any():
        return out
    start = np.flatnonzero(flagged & (code != np.r_[OK, code[:-1]]))
    for i in start:
        j = i
        while j + 1 < len(code) and code[j + 1] == code[i]:
            j += 1
        out.append({"check": CHECK_NAMES[int(code[i])], "start": idx[i], "end": idx[j], "minutes": j - i + 1,
                    "value": float(np.nanmax(v[i:j + 1]))})
    return out


def screen(df: pd.DataFrame, chans: list[Channel], year: int, floor: int
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cleaned minutes, per-minute codes, flagged episodes, per-channel summary)."""
    clean = df.copy()
    codes = pd.DataFrame(OK, index=df.index, columns=[c.column for c in chans], dtype="int8")
    events, summary = [], []
    for ch in chans:
        code, row = screen_channel(df[ch.column], ch)
        codes[ch.column] = code
        ident = {"year": year, "floor": floor, "zone": ch.zone, "zone_id": f"F{floor}Z{ch.zone}",
                 "channel": ch.name, "column": ch.column}
        summary.append({**ident, **row})
        for e in _episodes(code, df.index, df[ch.column].to_numpy(dtype="float64")):
            events.append({**ident, "kind": ch.kind, **e,
                           "threshold": row.get("spike_threshold_kw") if e["check"] == "Spike"
                           else row.get("flat_threshold_min")})
        clean.loc[code != OK, ch.column] = np.nan
    ev_cols = ["year", "floor", "zone", "zone_id", "channel", "column", "kind", "check", "start", "end", "minutes",
               "value", "threshold"]
    return clean, codes, pd.DataFrame(events, columns=ev_cols), pd.DataFrame(summary)

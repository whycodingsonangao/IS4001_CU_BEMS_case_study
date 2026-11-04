"""Read the raw CU-BEMS floor CSVs into a clean, gap-explicit minute grid."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

_COL_RE = re.compile(r"^z(?P<zone>\d+)_(?P<chan>[A-Za-z]+\d*)\((?P<unit>[^)]+)\)$")


@dataclass(frozen=True)
class Channel:
    column: str
    zone: int
    name: str  # e.g. AC3, Light, Plug, S1
    kind: str  # AC | Light | Plug | temp | rh | lux
    unit: str


def parse_channel(column: str) -> Channel | None:
    m = _COL_RE.match(column)
    if not m:
        return None
    chan, unit = m["chan"], m["unit"]
    if unit == "kW":
        kind = "AC" if chan.startswith("AC") else chan  # Light | Plug
    else:
        kind = {"degC": "temp", "RH%": "rh", "lux": "lux"}.get(unit, unit)
    return Channel(column, int(m["zone"]), chan, kind, unit)


def raw_path(year: int, floor: int) -> Path:
    return config.RAW_DIR / f"{year}Floor{floor}.csv"


def available_files() -> list[tuple[int, int]]:
    out = []
    for p in sorted(config.RAW_DIR.glob("*Floor*.csv")):
        m = re.match(r"(\d{4})Floor(\d)\.csv", p.name)
        if m:
            out.append((int(m[1]), int(m[2])))
    return out


def load_floor(year: int, floor: int) -> tuple[pd.DataFrame, list[Channel]]:
    """Return (minute DataFrame indexed by local timestamp, channel metadata).

    The index is re-gridded to every minute of the file's coverage period, so any
    missing row (e.g. Floor 6 after 22 Oct 2019) appears as explicit NaN, never zero.
    Malformed rows (a truncated final line exists in 2019Floor6.csv) are dropped.
    """
    path = raw_path(year, floor)
    try:
        df = pd.read_csv(path, engine="pyarrow")
    except (pd.errors.ParserError, ValueError):
        # pyarrow rejects ragged rows (2019Floor6.csv ends mid-line); the C parser pads them
        df = pd.read_csv(path, engine="c", low_memory=False)
    ts = pd.to_datetime(df.pop("Date").astype(str), errors="coerce", format="%Y-%m-%d %H:%M:%S")
    df = df.loc[ts.notna().to_numpy()].copy()
    df.index = ts[ts.notna()]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")

    start = f"{year}-07-01" if year == 2018 else f"{year}-01-01"
    grid = pd.date_range(start, f"{year}-12-31 23:59", freq="min")
    df = df.reindex(grid)
    df.index.name = "timestamp"

    channels = [c for c in (parse_channel(col) for col in df.columns) if c]
    return df, channels

"""External weather from the Open-Meteo Historical Weather API (ERA5-based reanalysis).

Terms: free for non-commercial use, data licensed CC BY 4.0 - cite "Weather data by
Open-Meteo.com" in the report. https://open-meteo.com/en/terms

Caveat (from the prof's feedback): this is regional reanalysis, not measured at the
building. Any relationship with AC use is a correlation, not proof of cause, and it is
only joined to the AC-metered (Model 1) zones in analysis.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import requests

from . import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LOCKED_WEATHER_PATH = config.REFERENCES_DIR / "weather_hourly_2018-2019.parquet"
LOCKED_WEATHER_SHA256 = "b9493247eebcc3c2ef5778d49888f6732c47768f05d67e304f88f78b3cfba076"


def fetch_hourly(start: str = "2018-07-01", end: str = "2019-12-31") -> pd.DataFrame:
    params = {
        "latitude": config.SITE_LAT,
        "longitude": config.SITE_LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(config.WEATHER_VARS),
        "timezone": config.TZ,  # returned in local time, same clock as CU-BEMS
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=120)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["hourly"])
    df["timestamp"] = pd.to_datetime(df.pop("time"))
    df.attrs["grid_lat"], df.attrs["grid_lon"] = js["latitude"], js["longitude"]
    return df


def load_hourly(refresh: bool = False) -> pd.DataFrame:
    """Read the exact weather snapshot used for the submitted results.

    A mutable API response or a machine-local cache must never silently change the
    features used by Question B. Refreshing requires an intentional snapshot update.
    """
    if refresh:
        raise ValueError("Weather is locked; update the committed snapshot and checksum intentionally.")
    if not LOCKED_WEATHER_PATH.is_file():
        raise FileNotFoundError(f"Locked weather snapshot is missing: {LOCKED_WEATHER_PATH}")
    digest = hashlib.sha256(LOCKED_WEATHER_PATH.read_bytes()).hexdigest()
    if digest != LOCKED_WEATHER_SHA256:
        raise ValueError(f"Locked weather snapshot checksum mismatch: {LOCKED_WEATHER_PATH}")
    return pd.read_parquet(LOCKED_WEATHER_PATH)


def daily_features(hourly: pd.DataFrame, cdh_base: float = config.CDH_BASE_C) -> pd.DataFrame:
    h = hourly.copy()
    h["date"] = h["timestamp"].dt.normalize()
    h["cdh"] = (h["temperature_2m"] - cdh_base).clip(lower=0)
    work = h[h["timestamp"].dt.hour.between(8, 17)]
    d = h.groupby("date").agg(
        temp_mean=("temperature_2m", "mean"),
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        rh_mean=("relative_humidity_2m", "mean"),
        dew_point_mean=("dew_point_2m", "mean"),
        apparent_temp_mean=("apparent_temperature", "mean"),
        radiation_kwh_m2=("shortwave_radiation", lambda s: s.sum() / 1000.0),
        precip_mm=("precipitation", "sum"),
        cloud_cover_mean=("cloud_cover", "mean"),
        cdh=("cdh", "sum"),
    )
    d["work_temp_mean"] = work.groupby("date")["temperature_2m"].mean()
    d["work_cdh"] = work.groupby("date")["cdh"].sum()
    return d.reset_index()

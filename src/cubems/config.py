"""Project-wide paths and analysis constants.

Every number that encodes a methodological decision lives here, so the team can
change it in one place and the pipeline, notebooks and dashboard stay consistent.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("CUBEMS_RAW_DIR", DATA_DIR / "raw"))
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

HOURLY_PATH = PROCESSED_DIR / "zone_hourly.parquet"
ZONE_DAY_PATH = PROCESSED_DIR / "zone_day.parquet"
ZONES_PATH = PROCESSED_DIR / "zones.parquet"
WEATHER_HOURLY_PATH = EXTERNAL_DIR / "weather_hourly.parquet"
WEATHER_DAILY_PATH = PROCESSED_DIR / "weather_daily.parquet"

# ---------------------------------------------------------------- data rules
MINUTES_PER_DAY = 1440
# A zone-day is usable when at least this share of its minutes has every power
# channel of that zone recorded (joint-meter coverage). Gaps are dropped, never filled.
MIN_DAY_COVERAGE = 0.95
# An hour counts as a valid monitored hour when at least this many minutes are present.
MIN_HOUR_MINUTES = 57
# Per-minute kW readings above this are flagged as "extreme" (flag, never delete).
# Largest plausible single-zone load is well under this; e.g. F1 z3 lighting 4,762 kW in 2018.
EXTREME_KW = 500.0

# ---------------------------------------------------------------- time windows
TZ = "Asia/Bangkok"  # CU-BEMS timestamps are local time
QA_WINDOW = ("2019-01-01", "2019-12-31")  # Question A: all of 2019
QB_BASELINE = ("2019-01-01", "2019-08-31")  # Question B: learn "normal"
QB_HOLDOUT = ("2019-09-01", "2019-12-31")  # Question B: replay / test
HISTORY_2018 = ("2018-07-01", "2018-12-31")  # optional extra folds (team decision pending)

# Bangkok seasons, in whole months (Thai Meteorological Department convention).
SEASON_BY_MONTH = {
    1: "Cool", 2: "Cool", 3: "Hot", 4: "Hot", 5: "Hot",
    6: "Rainy", 7: "Rainy", 8: "Rainy", 9: "Rainy", 10: "Rainy",
    11: "Cool", 12: "Cool",
}

# ---------------------------------------------------------------- weather
# CU-BEMS is the Chamchuri 5 building, Chulalongkorn University, Bangkok.
SITE_LAT = 13.7375
SITE_LON = 100.5311
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "shortwave_radiation",
    "precipitation",
    "cloud_cover",
]
# Base temperature for cooling degree-hours. 24 °C is a common tropical-office base;
# test sensitivity (e.g. 22, 26) before relying on it.
CDH_BASE_C = 24.0

# ---------------------------------------------------------------- scope
# Proposal scope: Model 1 = 23 fully equipped zones, Model 2 = 8 lighting+plug zones.
MODEL1_ZONES = {2: [1, 2, 4], **{f: [1, 2, 4, 5] for f in range(3, 8)}}
MODEL2_ZONES = {1: [1, 3], 2: [3], **{f: [3] for f in range(3, 8)}}

END_USES = ["AC", "Light", "Plug"]

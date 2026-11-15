"""Project-wide paths and analysis constants.

Every number that encodes a methodological decision lives here, so the pipeline, notebook and
dashboard stay consistent. Labels follow the methodology plan: [V] verified, [A] assumption, [D] decision.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = Path(os.environ.get("CUBEMS_RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
REFERENCES_DIR = ROOT / "references"

HOURLY_PATH = PROCESSED_DIR / "zone_hourly.parquet"
ZONE_DAY_PATH = PROCESSED_DIR / "zone_day.parquet"
ZONES_PATH = PROCESSED_DIR / "zones.parquet"
WEATHER_HOURLY_PATH = EXTERNAL_DIR / "weather_hourly.parquet"
WEATHER_DAILY_PATH = PROCESSED_DIR / "weather_daily.parquet"
QC_EVENTS_PATH = PROCESSED_DIR / "qc_events.parquet"
QC_CHANNELS_PATH = PROCESSED_DIR / "qc_channels.parquet"
HOLIDAYS_PATH = REFERENCES_DIR / "thai_holidays_2019.csv"


def results_dir(question: str, min_minutes: int) -> Path:
    """data/processed/qa or qa_strict (Question A), qb or qb_strict (Question B)."""
    return PROCESSED_DIR / (question if min_minutes == MIN_HOUR_MINUTES else f"{question}_strict")


# ---------------------------------------------------------------- data rules
MINUTES_PER_DAY = 1440
# [D] An hour is valid when at least 57 of its 60 minutes are valid (max kWh error 5%).
#     A minute is valid for an end use only if every circuit of that end use has a reading.
MIN_HOUR_MINUTES = 57
# [D] Sensitivity check: every result is re-run with the strict 60/60 rule.
STRICT_HOUR_MINUTES = 60
# [D] Per-channel screening (src/cubems/qc.py). Flagged minutes are set to missing, so the hour rule
#     above treats them like any other gap. Calibrated on the 2019 files (see README > Data screening).
# Spike: reading > SPIKE_RATIO x the channel's 99th percentile of active readings, and > SPIKE_MIN_KW.
SPIKE_RATIO = 5.0
SPIKE_REF_PCTL = 99
SPIKE_ACTIVE_KW = 0.05  # readings at or below this are idle and excluded from the reference level
SPIKE_MIN_KW = 1.0
# Flatline: the same value for at least this many consecutive minutes. [V] Longest identical non-idle run in
# 2019: AC < 60 min (compressors cycle), lighting/plugs 332 min (constant loads are real), temp/RH 25 min.
FLATLINE_MINUTES = {"AC": 60, "Light": 360, "Plug": 360, "temp": 120, "rh": 120}
FLATLINE_MIN_KW = 0.5  # power channels: only runs at or above this level (idle readings are naturally flat)

# ---------------------------------------------------------------- time windows
TZ = "Asia/Bangkok"  # [A] timestamps are Bangkok local time (AC on 08:00, off 12-13, off 17:00)
YEAR = 2019  # [D] all of 2019 is the analysis period; 2018 only for an optional check
YEAR_WINDOW = ("2019-01-01", "2019-12-31")
FLOOR_COMPARE_END = "2019-10-21"  # [D] floor comparisons stop at Floor 6's last full day
HISTORY_2018 = ("2018-07-01", "2018-12-31")

# Question B: chronological, expanding window, monthly refit [D]
QB_TRAIN_START = "2019-01-08"
QB_DEV_TEST_MONTHS = ["2019-05", "2019-06", "2019-07", "2019-08"]
QB_FINAL_TEST_MONTHS = ["2019-09", "2019-10", "2019-11", "2019-12"]
QB_NMBE_LIMIT = 5.0  # boosting stays primary only if |NMBE| <= 5 % on development folds

# Bangkok seasons in whole months (Thai Meteorological Department convention)
SEASON_BY_MONTH = {
    1: "Cool", 2: "Cool", 3: "Hot", 4: "Hot", 5: "Hot",
    6: "Rainy", 7: "Rainy", 8: "Rainy", 9: "Rainy", 10: "Rainy",
    11: "Cool", 12: "Cool",
}

# ---------------------------------------------------------------- weather
SITE_LAT = 13.7375  # Chamchuri 5 building, Chulalongkorn University
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
CDH_BASE_C = 24.0
# [V] ERA5 hourly temperature jumps >= 2 degC at fixed clock hours, so models use day-level weather only
QB_WEATHER_FEATURES = ["wx_temp_mean", "wx_temp_max", "wx_dew_point_mean", "wx_radiation_kwh_m2", "wx_precip_mm"]

# ---------------------------------------------------------------- zone groups [V]
# C1 = AC + lighting + plugs (24 zones, incl. F1z2). C2 = lighting + plugs (8). F1z4 = lighting only.
C1_ZONES = {1: [2], 2: [1, 2, 4], **{f: [1, 2, 4, 5] for f in range(3, 8)}}
C2_ZONES = {1: [1, 3], 2: [3], **{f: [3] for f in range(3, 8)}}
LIGHT_ONLY_ZONES = {1: [4]}
GROUP_USES = {"C1": ["AC", "Light", "Plug"], "C2": ["Light", "Plug"], "Lighting only": ["Light"]}

END_USES = ["AC", "Light", "Plug"]

# ---------------------------------------------------------------- Question A [D]
QA_K_RANGE = range(3, 9)
QA_MIN_CLUSTER_SHARE = 0.03  # every cluster holds >= 3 % of zone-days ...
QA_MIN_CLUSTER_ZONES = 3  # ... and appears in >= 3 zones
QA_MIN_HALF_YEAR_ARI = 0.8
QA_P95_FLOOR_KWH = 0.1  # floor on the 95th-percentile scaler so near-flat channels aren't amplified
QA_ATYPICAL_PCTL = 99  # days farther than this percentile from their centroid are flagged (kept)
# [D] Zones reported separately instead of clustered. F1Z2 runs 24/7 (AC highest at night, ~5x a normal zone's
#     daily AC, ~24% of building kWh) and formed a one-zone cluster at every k, so no k passed rule 1. Without it,
#     k=3 passes rules 1-2 under both the 57/60 and 60/60 rules.
QA_SEPARATE_ZONES = {"F1Z2": "it runs 24/7 at about 5x a typical zone's daily AC. On workdays its AC is low during "
                              "office hours and high from 17:00 to 06:00, the inverse of the office schedule; on "
                              "weekends it runs flat all day"}

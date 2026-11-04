"""Build every processed table the notebooks and dashboard read.

    python scripts/build_dataset.py              # full build (raw CSVs + weather)
    python scripts/build_dataset.py --no-weather # offline: skip the Open-Meteo call

Outputs (all gitignored, regenerate any time):
    data/processed/zone_hourly.parquet  one row per zone x hour, kWh + valid-minute counts
    data/processed/zone_day.parquet     one row per zone x day, coverage, usable flag, calendar,
                                        weather features joined on date
    data/processed/zones.parquet        zone registry (group, AC units, sensors)
    data/external/weather_hourly.parquet  cached Open-Meteo pull
    data/processed/weather_daily.parquet
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from cubems import config
from cubems.aggregate import build_floor, to_zone_day
from cubems.loader import available_files
from cubems.weather import daily_features, load_hourly


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-weather", action="store_true", help="skip the Open-Meteo download")
    ap.add_argument("--refresh-weather", action="store_true", help="re-download weather")
    args = ap.parse_args()

    files = available_files()
    if not files:
        print(f"No raw CSVs found in {config.RAW_DIR}. See README > Data.", file=sys.stderr)
        return 1
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    hourly, zones = [], []
    for year, floor in files:
        t = time.perf_counter()
        h, z = build_floor(year, floor)
        hourly.append(h)
        zones.extend(z)
        print(f"  {year} Floor {floor}: {len(h):>7,} zone-hours  ({time.perf_counter() - t:.1f}s)")

    hourly_df = pd.concat(hourly, ignore_index=True).sort_values(["floor", "zone", "timestamp"])
    hourly_df.to_parquet(config.HOURLY_PATH, index=False)

    zones_df = (
        pd.DataFrame(zones)
        .sort_values(["floor", "zone", "year"])
        .drop_duplicates(["floor", "zone"], keep="last")
        .drop(columns="year")
    )
    zones_df.to_parquet(config.ZONES_PATH, index=False)

    zd = to_zone_day(hourly_df)
    if not args.no_weather:
        try:
            wd = daily_features(load_hourly(refresh=args.refresh_weather))
            wd.to_parquet(config.WEATHER_DAILY_PATH, index=False)
            zd = zd.merge(wd.add_prefix("wx_").rename(columns={"wx_date": "date"}), on="date",
                          how="left")
            print(f"  weather: {len(wd)} days joined")
        except Exception as e:  # network down etc. - the rest of the build is still useful
            print(f"  WARNING weather skipped: {e}", file=sys.stderr)
    zd.to_parquet(config.ZONE_DAY_PATH, index=False)

    print(f"\nzone-hours {len(hourly_df):,} | zone-days {len(zd):,} | zones {len(zones_df)}")
    print(zones_df["group"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

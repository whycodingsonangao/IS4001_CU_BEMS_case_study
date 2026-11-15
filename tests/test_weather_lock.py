"""The reproducible build must use the committed weather snapshot."""

import pandas as pd
import pytest

from cubems import config, weather


def test_weather_ignores_unverified_local_cache(tmp_path, monkeypatch):
    cache = tmp_path / "weather_hourly.parquet"
    pd.DataFrame({"temperature_2m": [99.0]}).to_parquet(cache)
    monkeypatch.setattr(config, "WEATHER_HOURLY_PATH", cache)

    hourly = weather.load_hourly()

    assert len(hourly) == 13_176
    assert hourly.iloc[0]["temperature_2m"] == 27.1
    assert hourly.iloc[-1]["timestamp"] == pd.Timestamp("2019-12-31 23:00:00")


def test_weather_rejects_modified_snapshot(tmp_path, monkeypatch):
    altered = tmp_path / "weather.parquet"
    altered.write_bytes(b"not the locked snapshot")
    monkeypatch.setattr(weather, "LOCKED_WEATHER_PATH", altered, raising=False)

    with pytest.raises(ValueError, match="checksum"):
        weather.load_hourly()


def test_refresh_cannot_bypass_weather_lock():
    with pytest.raises(ValueError, match="locked"):
        weather.load_hourly(refresh=True)

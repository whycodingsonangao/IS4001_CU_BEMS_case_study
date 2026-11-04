import numpy as np
import pandas as pd

from cubems import config
from cubems import forecast as F
from cubems.aggregate import valid_kwh, zone_group
from cubems.anomaly import detect_episodes, kpis
from cubems.calendar import holiday_table
from cubems.loader import parse_channel


def test_parse_channel():
    c = parse_channel("z2_AC14(kW)")
    assert (c.zone, c.kind, c.name) == (2, "AC", "AC14")
    assert parse_channel("z1_S1(RH%)").kind == "rh"
    assert parse_channel("Date") is None


def test_zone_groups_match_plan():
    c1 = sum(len(v) for v in config.C1_ZONES.values())
    c2 = sum(len(v) for v in config.C2_ZONES.values())
    assert (c1, c2) == (24, 8)
    assert zone_group(1, 2) == "C1" and zone_group(1, 4) == "Lighting only"


def test_hour_validity_threshold():
    h = pd.DataFrame({"ac_kwh": [1.0, 2.0, 3.0], "ac_minutes": [60, 57, 56]})
    v = valid_kwh(h, "AC")
    assert v.isna().tolist() == [False, False, True]  # 56 minutes -> missing, never zero
    assert valid_kwh(h, "AC", config.STRICT_HOUR_MINUTES).notna().sum() == 1


def test_holiday_list_is_sourced():
    t = holiday_table()
    assert t["source"].notna().all()
    assert not t.loc[t["date"] == "2019-05-01", "use_as_holiday"].iloc[0]  # Labour Day: not a gov holiday
    assert t.loc[t["date"] == "2019-12-30", "use_as_holiday"].iloc[0]


def test_folds_are_chronological():
    for f in F.folds():
        assert f.train_end == pd.Timestamp(f"{f.test_month}-01")
    assert [f.test_month for f in F.folds() if f.stage == "development"] == config.QB_DEV_TEST_MONTHS


def _toy():
    ts = pd.date_range("2019-01-01", "2019-03-31 23:00", freq="h")
    y = np.where(ts.hour.isin(range(8, 17)) & (ts.day_of_week < 5), 10.0, 0.0)
    d = pd.DataFrame({"timestamp": ts, "y": y, "zone_id": "F2Z1", "floor": 2, "hour": ts.hour,
                      "dow": ts.day_of_week, "date": ts.normalize(), "is_holiday": 0,
                      "is_weekend": (ts.day_of_week >= 5).astype(int),
                      "day_type": np.where(ts.day_of_week >= 5, "Weekend/holiday", "Workday"),
                      "month": ts.strftime("%Y-%m")})
    d["lag_1d"] = d["y"].shift(24)
    d["lag_7d"] = d["y"].shift(168)
    d["yday_mean"] = d.groupby("date")["y"].transform("mean").shift(24)
    for c in F.LAGS:
        d[f"{c}_missing"] = d[c].isna().astype(int)
    for c in config.QB_WEATHER_FEATURES:
        d[c] = 30.0
    return d


def test_leakage_checks_and_nonnegative_predictions():
    d = _toy()
    f = F.Fold("development", "2019-03", pd.Timestamp("2019-03-01"))
    out, _ = F.run_fold(d, f, ("boosting", "ridge"))
    assert (out[["benchmark", "boosting", "ridge"]] >= 0).all().all()
    assert out["timestamp"].min() >= pd.Timestamp("2019-03-01")


def test_episode_needs_consecutive_hours():
    ts = pd.date_range("2019-09-02 08:00", periods=6, freq="h")
    s = pd.DataFrame({"floor": 2, "zone": 1, "timestamp": ts, "kw": [5, 20, 20, 5, 20, 5], "lo": 0.0, "hi": 10.0})
    scored, ep = detect_episodes(s, margin_kw=0.25, min_hours=2)
    assert len(ep) == 1 and ep.iloc[0]["duration_h"] == 2
    assert kpis(scored, ep).iloc[0]["duration_burden"] == 100 * 2 / 6

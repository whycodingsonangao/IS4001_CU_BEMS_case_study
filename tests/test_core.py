import numpy as np
import pandas as pd

from cubems import config
from cubems import forecast as F
from cubems.aggregate import valid_kwh, zone_group
from cubems.anomaly import daily_exceptions, daily_kpis
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
    dates = pd.date_range("2019-01-01", "2019-03-31", freq="D")
    y = np.where(dates.day_of_week < 5, 90.0, 0.0)
    d = pd.DataFrame({"date": dates, "ac_kwh": y, "ac_hours": 24, "ac_hours_strict": 24,
                         "group": "C1", "zone_id": "F2Z1", "floor": 2, "zone": 1,
                         "dow": dates.day_of_week, "is_holiday": False,
                         "is_weekend": dates.day_of_week >= 5,
                         "day_type": np.where(dates.day_of_week >= 5, "Weekend/holiday", "Workday")})
    for col in config.QB_WEATHER_FEATURES:
        d[col] = np.arange(len(dates), dtype=float)
    return d


def test_daily_design_uses_complete_days_and_prior_information():
    zd = _toy().iloc[:9].copy()
    zd.loc[2, "ac_hours"] = 23
    zd.loc[2, "ac_kwh"] = 40.0
    d = F.design(zd)
    assert d.loc[2, "y"] != d.loc[2, "y"]  # incomplete target is missing
    assert np.isnan(d.loc[3, "lag_1d"])  # missing day cannot become a lag
    assert d.loc[7, "lag_7d"] == 90.0
    assert "wx_temp_mean" not in F.FEATURES  # observed day-D weather is unavailable at issue time
    assert d.loc[3, "wx_prev_temp_mean"] == 2.0


def test_daily_design_strict_rule_requires_60_minute_hours():
    zd = _toy().iloc[:2].copy()
    zd.loc[0, "ac_hours_strict"] = 23
    assert np.isnan(F.design(zd, config.STRICT_HOUR_MINUTES).loc[0, "y"])


def test_leakage_checks_and_nonnegative_predictions():
    d = F.design(_toy())
    f = F.Fold("development", "2019-03", pd.Timestamp("2019-03-01"))
    out, _ = F.run_fold(d, f, ("boosting", "ridge"))
    assert (out[["benchmark", "boosting", "ridge"]] >= 0).all().all()
    assert out["date"].min() >= pd.Timestamp("2019-03-01")
    assert out["date"].nunique() == 31
    assert "typical_kwh" in out.columns
    assert out["typical_kwh"].notna().all()


def test_daily_exception_flags_and_zone_rate():
    s = pd.DataFrame({"zone_id": ["F2Z1"] * 3, "floor": 2, "zone": 1,
                      "date": pd.date_range("2019-09-02", periods=3),
                      "y": [12.0, 31.0, 1.0], "lo": [5.0] * 3, "hi": [20.0] * 3})
    scored = daily_exceptions(s)
    assert scored["direction"].tolist() == ["Within range", "Above expected", "Below expected"]
    assert daily_kpis(scored).iloc[0]["exception_days"] == 2


def test_priority_hit_rate_checks_the_highest_use_zone():
    d = pd.DataFrame({"date": [pd.Timestamp("2019-09-02")] * 3,
                      "zone_id": ["F2Z1", "F2Z2", "F2Z4"],
                      "y": [30.0, 20.0, 10.0],
                      "ridge": [31.0, 15.0, 12.0],
                      "benchmark": [5.0, 25.0, 10.0]})
    assert F.priority_hit_rate(d, "ridge", k=1) == 1.0
    assert F.priority_hit_rate(d, "benchmark", k=1) == 0.0

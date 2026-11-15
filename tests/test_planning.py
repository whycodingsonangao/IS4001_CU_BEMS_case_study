import numpy as np
import pandas as pd
import pytest

from cubems import config, planning
from cubems import forecast as F


def _design():
    dates = pd.date_range("2019-01-01", "2019-12-31", freq="D")
    frames = []
    for zid, level in (("F2Z1", 90.0), ("F3Z1", 200.0)):
        d = pd.DataFrame({"date": dates, "ac_hours": 24, "ac_hours_strict": 24, "group": "C1", "zone_id": zid,
                          "floor": int(zid[1]), "zone": 1, "dow": dates.day_of_week, "is_holiday": False,
                          "is_weekend": dates.day_of_week >= 5})
        d["ac_kwh"] = np.where(d["is_weekend"], 0.0, level) + (dates.day_of_year % 5)
        d["day_type"] = np.where(d["is_weekend"], "Weekend/holiday", "Workday")
        for col, base in zip(config.QB_WEATHER_FEATURES, (28.0, 33.0, 23.0, 5.0, 3.0), strict=True):
            d[col] = base + (dates.day_of_year % 7) / 7
        frames.append(d)
    return F.design(pd.concat(frames, ignore_index=True))


def test_training_window_respects_time_order():
    d = _design()
    assert planning.train_end_for("2019-10-17") == pd.Timestamp("2019-10-01")
    assert planning.training_rows(d, planning.train_end_for("2019-10-17"))["date"].max() < pd.Timestamp("2019-10-01")
    assert planning.train_end_for("2026-03-02") == pd.Timestamp("2020-01-01") and planning.is_illustrative("2026-03-02")
    assert not planning.is_illustrative("2019-12-31")


@pytest.mark.parametrize("chosen", ["ridge", "boosting"])
def test_what_if_reproduces_monthly_replay_and_ignores_the_target_actual(chosen):
    d = _design()
    fold = next(f for f in F.folds() if f.test_month == "2019-10")
    train, test = F.split(d, fold)
    replay = (F.predict_ridge(F.fit_ridge(train), test)[0] if chosen == "ridge"
              else F.predict_boosting(F.fit_boosting(train), test))
    target, zid = pd.Timestamp("2019-10-16"), "F2Z1"
    lags, weather = planning.historical_inputs(d, target, [zid])
    fitted = planning.fit(d, target, chosen)
    got = planning.predict(fitted, planning.build_rows(lags, weather, target))[0]
    want = replay[((test["date"] == target) & (test["zone_id"] == zid)).to_numpy()][0]
    assert got == pytest.approx(want, rel=1e-5)
    changed = d.copy()  # the target day's own reading must not move its forecast
    changed.loc[(changed["date"] == target) & (changed["zone_id"] == zid), "y"] = 9999.0
    again = planning.predict(planning.fit(changed, target, chosen), planning.build_rows(lags, weather, target))[0]
    assert again == pytest.approx(got, rel=1e-6)


def test_prefill_uses_an_earlier_day_and_never_its_actual():
    d = _design()
    target = pd.Timestamp("2019-10-16")
    cal = planning.calendar_fields(target)
    src = planning.similar_day(d, target, cal["day_type"], "F2Z1")
    assert src == pd.Timestamp("2019-10-09") and src < target
    lags, weather = planning.historical_inputs(d, src, ["F2Z1"])
    row = d[(d["date"] == src) & (d["zone_id"] == "F2Z1")].iloc[0]
    assert set(lags["F2Z1"]) == set(F.LAGS) and set(weather) == set(F.PRIOR_WEATHER)
    assert lags["F2Z1"]["lag_1d"] == pytest.approx(row["lag_1d"]) and lags["F2Z1"]["lag_1d"] != row["y"]
    assert planning.similar_day(d, "2026-03-02", "Workday") == pd.Timestamp("2019-12-30")  # future: within 2019


def test_validation_and_missing_lags():
    d = _design()
    fitted = planning.fit(d, "2019-10-16", "ridge")
    _, weather = planning.historical_inputs(d, "2019-10-09", ["F2Z1"])
    ok = {"F2Z1": {"lag_1d": 90.0, "lag_7d": 90.0, "lag_7d_mean": 70.0}}
    assert planning.validate(ok, weather, fitted) == ([], [])
    errors, _ = planning.validate({"F2Z1": {"lag_1d": -5.0, "lag_7d": 90.0, "lag_7d_mean": 1e6}}, weather, fitted)
    assert any("negative" in e for e in errors) and any("units" in e for e in errors)
    errors, _ = planning.validate(ok, {**weather, "wx_prev_temp_mean": None}, fitted)
    assert any("required" in e for e in errors)
    errors, _ = planning.validate(ok, {**weather, "wx_prev_precip_mm": -1.0}, fitted)
    assert any("between" in e for e in errors)
    _, warnings = planning.validate({"F2Z1": {"lag_1d": 400.0, "lag_7d": None, "lag_7d_mean": None}},
                                    {**weather, "wx_prev_temp_mean": 45.0, "wx_prev_temp_max": 50.0}, fitted)
    assert any("training maximum" in w for w in warnings) and any("no value" in w for w in warnings)
    assert any("training range" in w for w in warnings)
    rows = planning.build_rows({"F2Z1": dict.fromkeys(F.LAGS)}, weather, "2019-10-16")
    assert rows[F.INDICATORS].iloc[0].tolist() == [1, 1, 1] and rows[F.LAGS].isna().all(axis=None)
    assert np.isfinite(planning.predict(fitted, rows)).all() and (planning.predict(fitted, rows) >= 0).all()


def test_calendar_fields():
    assert planning.calendar_fields("2019-10-23")["day_type"] == "Weekend/holiday"  # Chulalongkorn Day
    assert planning.calendar_fields("2019-10-22")["day_type"] == "Workday"
    assert planning.calendar_fields("2026-03-02", holiday=True)["is_holiday"] == 1

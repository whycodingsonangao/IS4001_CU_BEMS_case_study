"""What-if next-day AC forecasts for planning, using the Question B model exactly as selected.

Everything here reuses src/cubems/forecast.py: the same feature list, the same preprocessing and the
model named in decision.json. Training respects time order: a target day is forecast by a model
fitted only on complete zone-days before the first day of the target's month (the monthly replay
rule). Targets after the dataset end use all of 2019 and are illustrative, not validated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from . import forecast as F
from .calendar import holiday_dates

DATA_END = pd.Timestamp(config.YEAR_WINDOW[1])
FIRST_TARGET = pd.Timestamp(f"{config.QB_DEV_TEST_MONTHS[0]}-01")  # earlier months have too little training data

LAG_LABELS = {
    "lag_1d": "AC total, previous day",
    "lag_7d": "AC total, 7 days earlier",
    "lag_7d_mean": "Mean AC total, previous 7 days",
}
WEATHER_LABELS = {
    "wx_prev_temp_mean": ("Mean outdoor temperature, previous day", "°C"),
    "wx_prev_temp_max": ("Maximum outdoor temperature, previous day", "°C"),
    "wx_prev_dew_point_mean": ("Mean dew point, previous day", "°C"),
    "wx_prev_radiation_kwh_m2": ("Solar radiation, previous day", "kWh/m²"),
    "wx_prev_precip_mm": ("Rainfall, previous day", "mm"),
}
# Hard physical limits [D]: values outside these are rejected, not just warned about.
HARD_LIMITS = {
    "wx_prev_temp_mean": (0.0, 50.0),
    "wx_prev_temp_max": (0.0, 55.0),
    "wx_prev_dew_point_mean": (-10.0, 35.0),
    "wx_prev_radiation_kwh_m2": (0.0, 12.0),
    "wx_prev_precip_mm": (0.0, 500.0),
}
MAX_DAILY_KWH = 5000.0  # reject obvious unit slips (the largest 2019 zone-day is about 2,000 kWh)
LAG_RANGE_MARGIN = 1.25  # warn when a lag exceeds 1.25 x the zone's training maximum


def train_end_for(target) -> pd.Timestamp:
    """Training stops before the first day of the target's month; after the dataset, before 2020."""
    target = pd.Timestamp(target)
    if target > DATA_END:
        return DATA_END + pd.Timedelta(days=1)
    return target.replace(day=1)


def is_illustrative(target) -> bool:
    return pd.Timestamp(target) > DATA_END


def calendar_fields(target, holiday: bool | None = None) -> dict:
    """dow / weekend / holiday / day type from the target date. `holiday` overrides the 2019 list."""
    target = pd.Timestamp(target).normalize()
    dow = int(target.day_of_week)
    is_holiday = bool(target in holiday_dates()) if holiday is None else bool(holiday)
    is_weekend = dow >= 5
    return {"dow": dow, "is_weekend": int(is_weekend), "is_holiday": int(is_holiday), "day_name": target.day_name(),
            "day_type": "Weekend/holiday" if is_weekend or is_holiday else "Workday"}


@dataclass
class Fitted:
    chosen: str
    model: object
    train_end: pd.Timestamp
    train_rows: int
    train_last: pd.Timestamp
    typical: pd.Series  # median y by (zone_id, day_type), training rows only
    zone_typical: pd.Series
    lag_max: pd.Series  # per-zone maximum complete daily total in training
    num_range: pd.DataFrame  # min / max of every numeric feature in training
    zones: list[str] = field(default_factory=list)


def training_rows(design: pd.DataFrame, train_end) -> pd.DataFrame:
    rows = design[design["y"].notna()]
    return rows[(rows["date"] >= pd.Timestamp(config.QB_TRAIN_START)) & (rows["date"] < pd.Timestamp(train_end))]


def fit(design: pd.DataFrame, target, chosen: str) -> Fitted:
    """Fit the chosen Question B model on the days allowed for this target."""
    train_end = train_end_for(target)
    train = training_rows(design, train_end)
    if train.empty:
        raise ValueError(f"No training days before {train_end:%d %b %Y}.")
    if chosen == "ridge":
        model = F.fit_ridge(train)
    elif chosen == "boosting":
        model = F.fit_boosting(train)
    else:
        raise ValueError(f"decision.json names an unsupported model: {chosen!r}")
    return Fitted(
        chosen=chosen, model=model, train_end=train_end, train_rows=len(train), train_last=train["date"].max(),
        typical=train.groupby(["zone_id", "day_type"])["y"].median(),
        zone_typical=train.groupby("zone_id")["y"].median(),
        lag_max=train.groupby("zone_id")["y"].max(),
        num_range=train[F.NUMERIC].agg(["min", "max"]).T,
        zones=sorted(train["zone_id"].unique()))


def build_rows(zone_inputs: dict[str, dict], weather: dict, target, holiday: bool | None = None) -> pd.DataFrame:
    """Model input rows. `zone_inputs` maps zone id -> {lag name: kWh or None}; None means missing."""
    cal = calendar_fields(target, holiday)
    rows = []
    for zid, lags in zone_inputs.items():
        row = {"zone_id": zid, "date": pd.Timestamp(target), **cal}
        for col in F.LAGS:
            v = lags.get(col)
            v = np.nan if v is None or pd.isna(v) else float(v)
            row[col] = v
            row[f"{col}_missing"] = int(np.isnan(v))
        for col in F.PRIOR_WEATHER:
            row[col] = float(weather[col])
        rows.append(row)
    return pd.DataFrame(rows)


def predict(fitted: Fitted, rows: pd.DataFrame) -> np.ndarray:
    """Forecast kWh for prepared rows with the fitted chosen model (never negative)."""
    if fitted.chosen == "ridge":
        return F.predict_ridge(fitted.model, rows)[0]
    return np.clip(F.predict_boosting(fitted.model, rows), 0, None)


def typical_kwh(fitted: Fitted, zid: str, day_type: str) -> float | None:
    if (zid, day_type) in fitted.typical.index:
        return float(fitted.typical.loc[(zid, day_type)])
    if zid in fitted.zone_typical.index:
        return float(fitted.zone_typical.loc[zid])
    return None


def validate(zone_inputs: dict[str, dict], weather: dict, fitted: Fitted) -> tuple[list[str], list[str]]:
    """(errors, warnings). Errors block the forecast; warnings flag inputs far from the training data."""
    errors, warnings = [], []
    for col, (label, unit) in WEATHER_LABELS.items():
        v = weather.get(col)
        if v is None or pd.isna(v):
            errors.append(f"{label} is required ({unit}).")
            continue
        lo, hi = HARD_LIMITS[col]
        if not lo <= v <= hi:
            errors.append(f"{label} must be between {lo:g} and {hi:g} {unit}; got {v:g}.")
            continue
        tlo, thi = fitted.num_range.loc[col, "min"], fitted.num_range.loc[col, "max"]
        if v < tlo or v > thi:
            warnings.append(f"{label} of {v:g} {unit} is outside the training range ({tlo:.1f} to {thi:.1f} {unit}). "
                            "The model is extrapolating.")
    t_mean, t_max = weather.get("wx_prev_temp_mean"), weather.get("wx_prev_temp_max")
    if t_mean is not None and t_max is not None and not pd.isna(t_mean) and not pd.isna(t_max) and t_max < t_mean:
        errors.append("Maximum temperature cannot be below the mean temperature.")
    for zid, lags in zone_inputs.items():
        if zid not in fitted.zones:
            errors.append(f"{zid} is not an AC zone the model was trained on.")
            continue
        zmax = float(fitted.lag_max.get(zid, np.nan))
        missing = []
        for col in F.LAGS:
            v = lags.get(col)
            if v is None or pd.isna(v):
                missing.append(LAG_LABELS[col])
                continue
            if v < 0:
                errors.append(f"{zid}: {LAG_LABELS[col]} cannot be negative (got {v:g} kWh).")
            elif v > MAX_DAILY_KWH:
                errors.append(f"{zid}: {LAG_LABELS[col]} of {v:,.0f} kWh is not a plausible daily zone total. "
                              "Check the units (kWh per day).")
            elif not np.isnan(zmax) and v > LAG_RANGE_MARGIN * zmax:
                warnings.append(f"{zid}: {LAG_LABELS[col]} of {v:,.1f} kWh is well above this zone's training "
                                f"maximum ({zmax:,.1f} kWh). The model is extrapolating.")
        if missing:
            warnings.append(f"{zid}: no value for {', '.join(missing)}. The model uses its missing-value handling "
                            "(training average plus a missing flag), so this forecast is less specific.")
    return errors, warnings


# ---------------------------------------------------------------- historical prefill
def similar_day(design: pd.DataFrame, target, day_type: str, zone_id: str | None = None) -> pd.Timestamp | None:
    """Most recent earlier 2019 day with the same weekday and day type (falls back to day type only).

    Source days always precede the target. With a zone, the day must have that zone's previous-day lag.
    """
    target = pd.Timestamp(target)
    d = design[design["date"] < min(target, DATA_END + pd.Timedelta(days=1))]
    d = d[d["date"] >= pd.Timestamp(config.QB_TRAIN_START)]
    if zone_id is not None:
        d = d[(d["zone_id"] == zone_id) & d["lag_1d"].notna()]
    d = d[d["day_type"] == day_type]
    same_dow = d[d["dow"] == target.day_of_week]
    pool = same_dow if len(same_dow) else d
    return None if pool.empty else pd.Timestamp(pool["date"].max())


def historical_inputs(design: pd.DataFrame, source_date, zone_ids: list[str]) -> tuple[dict[str, dict], dict]:
    """Model inputs as they stood at 00:00 on `source_date`: lags and previous-day weather only.

    The source day's own AC total (`y`) is deliberately not returned and never becomes an input.
    """
    day = design[design["date"] == pd.Timestamp(source_date)].set_index("zone_id")
    lags = {}
    for zid in zone_ids:
        if zid in day.index:
            lags[zid] = {c: (None if pd.isna(day.loc[zid, c]) else float(day.loc[zid, c])) for c in F.LAGS}
        else:
            lags[zid] = dict.fromkeys(F.LAGS)
    weather = {}
    if len(day):
        first = day.iloc[0]
        weather = {c: (None if pd.isna(first[c]) else float(first[c])) for c in F.PRIOR_WEATHER}
    return lags, weather

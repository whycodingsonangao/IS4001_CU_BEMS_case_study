"""Question B: day-ahead prediction of each C1 zone's hourly AC kWh (methodology plan, section 5).

Forecast issued at 00:00 on day D for all 24 hours of D, so every input is known by then:
zone, hour, day of week, official holiday flags; AC kWh at the same hour 1 and 7 days earlier and
yesterday's hourly mean (missing stays missing + indicator, never zero); day-D weather (ERA5 stands in
for a forecast, so accuracy is an upper bound). Same-day indoor sensors, same-day lighting/plug/AC and
Question A's cluster labels are never inputs.

Evaluation: chronological, expanding window, monthly refit. Development folds test May-Aug; the
chosen model then goes to the final test Sep-Dec. All encoders, scalers and fallback means are fitted on
training rows only (see :func:`check_leakage`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from . import config
from .aggregate import valid_kwh

LAGS = ["lag_1d", "lag_7d", "yday_mean"]
CAT = ["zone_id", "hour", "dow"]
FLAGS = ["is_holiday", "is_weekend"]
NUMERIC = LAGS + config.QB_WEATHER_FEATURES
INDICATORS = [f"{c}_missing" for c in LAGS]
FEATURES = CAT + FLAGS + NUMERIC + INDICATORS
# [A] monotone: more AC yesterday / last week / on hotter days -> more AC predicted
MONOTONE = {"lag_1d": 1, "lag_7d": 1, "yday_mean": 1, "wx_temp_mean": 1, "wx_temp_max": 1}


# ---------------------------------------------------------------- design matrix
def design(hourly: pd.DataFrame, zone_day: pd.DataFrame, min_minutes: int = config.MIN_HOUR_MINUTES) -> pd.DataFrame:
    """Full zone-hour grid for C1 zones in 2019 with target ``y`` (NaN where the hour is invalid)."""
    zones = zone_day.loc[zone_day["group"] == "C1", ["floor", "zone"]].drop_duplicates()
    h = hourly[hourly["timestamp"].dt.year == config.YEAR].merge(zones, on=["floor", "zone"])
    h = h[["floor", "zone", "timestamp"]].assign(y=valid_kwh(h, "ac", min_minutes).to_numpy())
    h = h.sort_values(["floor", "zone", "timestamp"]).reset_index(drop=True)
    h["zone_id"] = "F" + h["floor"].astype(str) + "Z" + h["zone"].astype(str)
    h["date"] = h["timestamp"].dt.normalize()
    h["hour"] = h["timestamp"].dt.hour

    g = h.groupby("zone_id")["y"]
    h["lag_1d"] = g.shift(24)  # grid is complete per zone, so shift = same hour D-1 / D-7
    h["lag_7d"] = g.shift(168)
    ymean = h.groupby(["zone_id", "date"])["y"].mean().rename("yday_mean").reset_index()
    ymean["date"] = ymean["date"] + pd.Timedelta(days=1)
    h = h.merge(ymean, on=["zone_id", "date"], how="left")
    for c in LAGS:
        h[f"{c}_missing"] = h[c].isna().astype(int)

    cal = zone_day[["floor", "zone", "date", "dow", "is_holiday", "is_weekend", "day_type",
                    *config.QB_WEATHER_FEATURES]]
    h = h.merge(cal, on=["floor", "zone", "date"], how="left")
    h["is_holiday"] = h["is_holiday"].astype(int)
    h["is_weekend"] = h["is_weekend"].astype(int)
    h["month"] = h["timestamp"].dt.strftime("%Y-%m")
    return h


# ---------------------------------------------------------------- folds
@dataclass(frozen=True)
class Fold:
    stage: str  # "development" | "final"
    test_month: str
    train_end: pd.Timestamp  # last training timestamp (exclusive bound)


def folds() -> list[Fold]:
    out = []
    for stage, months in (("development", config.QB_DEV_TEST_MONTHS), ("final", config.QB_FINAL_TEST_MONTHS)):
        for m in months:
            out.append(Fold(stage, m, pd.Timestamp(f"{m}-01")))
    return out


def split(d: pd.DataFrame, f: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = d[d["y"].notna()]
    train = rows[(rows["timestamp"] >= pd.Timestamp(config.QB_TRAIN_START)) & (rows["timestamp"] < f.train_end)]
    test = rows[rows["month"] == f.test_month]
    return train, test


# ---------------------------------------------------------------- models
def fallback_means(train: pd.DataFrame) -> dict[str, pd.Series]:
    return {"zhd": train.groupby(["zone_id", "hour", "day_type"])["y"].mean(),
            "zh": train.groupby(["zone_id", "hour"])["y"].mean(),
            "z": train.groupby("zone_id")["y"].mean()}


def predict_benchmark(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Same hour 7 days earlier; fallback to training-only means when that hour is missing."""
    fb = fallback_means(train)
    p = test["lag_7d"].copy()
    for key, cols in (("zhd", ["zone_id", "hour", "day_type"]), ("zh", ["zone_id", "hour"]), ("z", ["zone_id"])):
        miss = p.isna()
        if miss.any():
            idx = pd.MultiIndex.from_frame(test.loc[miss, cols]) if len(cols) > 1 else test.loc[miss, cols[0]]
            p[miss] = fb[key].reindex(idx).to_numpy()
    return p.fillna(float(train["y"].mean())).to_numpy()


def _xgb_frame(df: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["zone_id"] = pd.Categorical(X["zone_id"], categories=zones)
    return X


def fit_boosting(train: pd.DataFrame, seed: int = 0) -> tuple[XGBRegressor, list[str]]:
    zones = sorted(train["zone_id"].unique())
    mono = tuple(MONOTONE.get(c, 0) for c in FEATURES)
    model = XGBRegressor(objective="count:poisson", n_estimators=500, learning_rate=0.05, max_depth=7,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         enable_categorical=True, max_cat_to_onehot=1, monotone_constraints=mono,
                         random_state=seed, n_jobs=-1)
    model.fit(_xgb_frame(train, zones), train["y"])
    return model, zones


def predict_boosting(fitted: tuple[XGBRegressor, list[str]], test: pd.DataFrame) -> np.ndarray:
    model, zones = fitted
    return model.predict(_xgb_frame(test, zones))


def _ridge_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    X["zone_hour"] = X["zone_id"] + "_" + X["hour"].astype(str)
    X["zone_daytype"] = X["zone_id"] + "_" + df["day_type"].astype(str)
    return X


def fit_ridge(train: pd.DataFrame, alpha: float = 1.0):
    cats = CAT + ["zone_hour", "zone_daytype"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), NUMERIC),
        ("bin", "passthrough", FLAGS + INDICATORS),
    ])
    model = make_pipeline(pre, Ridge(alpha=alpha))
    model.fit(_ridge_frame(train), train["y"])
    return model


def predict_ridge(model, test: pd.DataFrame) -> tuple[np.ndarray, float]:
    raw = model.predict(_ridge_frame(test))
    return np.clip(raw, 0, None), float((raw < 0).mean())


# ---------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    rmse = float(np.sqrt((e ** 2).mean()))
    return {"mae": float(np.abs(e).mean()), "rmse": rmse, "cvrmse": 100 * rmse / float(y.mean()),
            "nmbe": float(100 * e.sum() / y.sum()), "n": int(len(y))}


def score(pred: pd.DataFrame, model: str) -> dict:
    """Zone-hour metrics plus zone-day and building-hour total errors and skill vs benchmark."""
    m = metrics(pred["y"].to_numpy(), pred[model].to_numpy())
    zd_ = pred.groupby(["zone_id", "date"])[["y", model]].sum()
    bh = pred.groupby("timestamp")[["y", model]].sum()
    m["zone_day_mae"] = float((zd_[model] - zd_["y"]).abs().mean())
    m["building_hour_mae"] = float((bh[model] - bh["y"]).abs().mean())
    if model != "benchmark":
        m["skill_vs_benchmark"] = 1 - m["mae"] / metrics(pred["y"].to_numpy(), pred["benchmark"].to_numpy())["mae"]
    return m


# ---------------------------------------------------------------- leakage checks
def check_leakage(train: pd.DataFrame, test: pd.DataFrame, preds: dict[str, np.ndarray]) -> None:
    assert train["timestamp"].max() < test["timestamp"].min(), "training rows must precede test rows"
    assert len({len(v) for v in preds.values()} | {len(test)}) == 1, "every model scored on identical targets"
    for k, v in preds.items():
        assert np.isfinite(v).all(), f"{k}: non-finite predictions"
        assert (v >= 0).all(), f"{k}: negative predictions"
    # lags may only look backwards: D-1 / D-7 values come from before the forecast issue time (00:00 D)
    assert test["lag_1d_missing"].isin([0, 1]).all()


# ---------------------------------------------------------------- run one fold
def run_fold(d: pd.DataFrame, f: Fold, models: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    train, test = split(d, f)
    preds: dict[str, np.ndarray] = {"benchmark": predict_benchmark(train, test)}
    info: dict = {"stage": f.stage, "test_month": f.test_month, "train_rows": len(train), "test_rows": len(test),
                  "train_last": str(train["timestamp"].max())}
    if "boosting" in models:
        preds["boosting"] = predict_boosting(fit_boosting(train), test)
    if "ridge" in models:
        preds["ridge"], info["ridge_negative_share"] = predict_ridge(fit_ridge(train), test)
    check_leakage(train, test, preds)
    out = test[["zone_id", "floor", "timestamp", "date", "hour", "day_type", "month", "y"]].copy()
    for k, v in preds.items():
        out[k] = v
    out["stage"] = f.stage
    return out, info


def weekly_block_diagnostic(d: pd.DataFrame, weeks_per_season: int = 2) -> pd.DataFrame:
    """Diagnostic only: seasonally stratified weekly-block CV inside the development period,
    purging the 7 days after each test week and embargoing the 1 day before it."""
    dev_end = pd.Timestamp(f"{config.QB_DEV_TEST_MONTHS[-1]}-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
    rows = d[d["y"].notna() & (d["timestamp"] >= pd.Timestamp(config.QB_TRAIN_START)) & (d["timestamp"] <= dev_end)]
    mondays = pd.date_range(config.QB_TRAIN_START, dev_end, freq="W-MON")
    out = []
    for season in ("Cool", "Hot", "Rainy"):
        cand = [m for m in mondays if config.SEASON_BY_MONTH[m.month] == season and m + pd.Timedelta(days=6) <= dev_end]
        pick = [cand[i] for i in np.linspace(0, len(cand) - 1, weeks_per_season).round().astype(int)] if cand else []
        for start in pick:
            end = start + pd.Timedelta(days=7)
            test = rows[(rows["timestamp"] >= start) & (rows["timestamp"] < end)]
            embargo, purge = start - pd.Timedelta(days=1), end + pd.Timedelta(days=7)
            keep = (rows["timestamp"] < embargo) | (rows["timestamp"] >= purge)
            train = rows[keep]
            p_b = predict_boosting(fit_boosting(train), test)
            p_0 = predict_benchmark(train, test)
            out.append({"season": season, "week_start": start.date().isoformat(),
                        "boosting_mae": metrics(test["y"].to_numpy(), p_b)["mae"],
                        "benchmark_mae": metrics(test["y"].to_numpy(), p_0)["mae"], "test_rows": len(test)})
    return pd.DataFrame(out)


def residual_bands(dev_pred: pd.DataFrame, model: str, q: tuple[float, float] = (0.05, 0.95),
                   min_n: int = 20) -> pd.DataFrame:
    """Locked exception thresholds from out-of-sample development residuals (actual - predicted)
    per zone x hour x day type, falling back to zone x day type when a cell is thin."""
    r = dev_pred.assign(res=dev_pred["y"] - dev_pred[model])
    fine = r.groupby(["zone_id", "hour", "day_type"])["res"].agg(
        res_lo=lambda s: s.quantile(q[0]), res_hi=lambda s: s.quantile(q[1]), n="size").reset_index()
    coarse = r.groupby(["zone_id", "day_type"])["res"].agg(
        c_lo=lambda s: s.quantile(q[0]), c_hi=lambda s: s.quantile(q[1])).reset_index()
    b = fine.merge(coarse, on=["zone_id", "day_type"], how="left")
    thin = b["n"] < min_n
    b.loc[thin, "res_lo"] = b.loc[thin, "c_lo"]
    b.loc[thin, "res_hi"] = b.loc[thin, "c_hi"]
    return b[["zone_id", "hour", "day_type", "res_lo", "res_hi", "n"]]

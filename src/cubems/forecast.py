"""Next-day total AC kWh for each C1 zone, using information available at 00:00 on that day."""
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

LAGS = ["lag_1d", "lag_7d", "lag_7d_mean"]
PRIOR_WEATHER = [f"wx_prev_{c.removeprefix('wx_')}" for c in config.QB_WEATHER_FEATURES]
CAT = ["zone_id", "dow"]
FLAGS = ["is_holiday", "is_weekend"]
NUMERIC = LAGS + PRIOR_WEATHER
INDICATORS = [f"{c}_missing" for c in LAGS]
FEATURES = CAT + FLAGS + NUMERIC + INDICATORS
MONOTONE = {c: 1 for c in LAGS}


def design(zone_day: pd.DataFrame, min_minutes: int = config.MIN_HOUR_MINUTES) -> pd.DataFrame:
    """One calendar row per C1 zone-day; target exists only for 24 valid AC hours."""
    d = zone_day[(zone_day["group"] == "C1") & (zone_day["date"].dt.year == config.YEAR)].copy()
    d = d.sort_values(["zone_id", "date"]).reset_index(drop=True)
    complete_col = "ac_hours_strict" if min_minutes == config.STRICT_HOUR_MINUTES else "ac_hours"
    d["y"] = d["ac_kwh"].where(d[complete_col] == 24)
    g = d.groupby("zone_id")["y"]
    d["lag_1d"] = g.shift(1)
    d["lag_7d"] = g.shift(7)
    d["lag_7d_mean"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    for col in LAGS:
        d[f"{col}_missing"] = d[col].isna().astype(int)
    for source, name in zip(config.QB_WEATHER_FEATURES, PRIOR_WEATHER, strict=True):
        d[name] = d.groupby("zone_id")[source].shift(1)
    d["is_holiday"] = d["is_holiday"].astype(int)
    d["is_weekend"] = d["is_weekend"].astype(int)
    d["month"] = d["date"].dt.strftime("%Y-%m")
    return d


@dataclass(frozen=True)
class Fold:
    stage: str
    test_month: str
    train_end: pd.Timestamp


def folds() -> list[Fold]:
    return [Fold(stage, month, pd.Timestamp(f"{month}-01"))
            for stage, months in (("development", config.QB_DEV_TEST_MONTHS),
                                  ("final", config.QB_FINAL_TEST_MONTHS)) for month in months]


def split(d: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = d[d["y"].notna()]
    train = rows[(rows["date"] >= pd.Timestamp(config.QB_TRAIN_START)) & (rows["date"] < fold.train_end)]
    test = rows[rows["month"] == fold.test_month]
    return train, test


def predict_benchmark(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Same zone's complete daily total seven days earlier; training-only mean fallback."""
    p = test["lag_7d"].copy()
    for cols in (["zone_id", "day_type"], ["zone_id"]):
        miss = p.isna()
        if not miss.any():
            break
        means = train.groupby(cols)["y"].mean()
        keys = pd.MultiIndex.from_frame(test.loc[miss, cols]) if len(cols) > 1 else test.loc[miss, cols[0]]
        p.loc[miss] = means.reindex(keys).to_numpy()
    return p.fillna(float(train["y"].mean())).to_numpy()


def _xgb_frame(d: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    x = d[FEATURES].copy()
    x["zone_id"] = pd.Categorical(x["zone_id"], categories=zones)
    return x


def fit_boosting(train: pd.DataFrame, seed: int = 0) -> tuple[XGBRegressor, list[str]]:
    zones = sorted(train["zone_id"].unique())
    model = XGBRegressor(objective="count:poisson", n_estimators=500, learning_rate=0.05, max_depth=7,
                         min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                         enable_categorical=True, max_cat_to_onehot=1,
                         monotone_constraints=tuple(MONOTONE.get(c, 0) for c in FEATURES),
                         random_state=seed, n_jobs=-1)
    model.fit(_xgb_frame(train, zones), train["y"])
    return model, zones


def predict_boosting(fitted: tuple[XGBRegressor, list[str]], test: pd.DataFrame) -> np.ndarray:
    model, zones = fitted
    return model.predict(_xgb_frame(test, zones))


def _ridge_frame(d: pd.DataFrame) -> pd.DataFrame:
    x = d[FEATURES].copy()
    x["zone_daytype"] = x["zone_id"] + "_" + d["day_type"].astype(str)
    return x


def fit_ridge(train: pd.DataFrame, alpha: float = 1.0):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT + ["zone_daytype"]),
        ("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), NUMERIC),
        ("bin", "passthrough", FLAGS + INDICATORS),
    ])
    model = make_pipeline(pre, Ridge(alpha=alpha))
    model.fit(_ridge_frame(train), train["y"])
    return model


def predict_ridge(model, test: pd.DataFrame) -> tuple[np.ndarray, float]:
    raw = model.predict(_ridge_frame(test))
    return np.clip(raw, 0, None), float((raw < 0).mean())


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    error = p - y
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    mean = float(y.mean())
    return {"mae": mae, "rmse": rmse, "cvrmse": 100 * rmse / mean if mean else np.nan,
            "nmbe": 100 * float(error.mean()) / mean if mean else np.nan, "n": len(y)}


def priority_hit_rate(pred: pd.DataFrame, model: str, k: int = 5) -> float:
    """Mean share of the actual highest-use k zones found in each day's predicted top k."""
    hits = []
    for _, day in pred.groupby("date"):
        if len(day) < k:
            continue
        predicted = set(day.nlargest(k, model)["zone_id"])
        actual = set(day.nlargest(k, "y")["zone_id"])
        hits.append(len(predicted & actual) / k)
    return float(np.mean(hits)) if hits else np.nan


def score(pred: pd.DataFrame, model: str) -> dict:
    m = metrics(pred["y"].to_numpy(), pred[model].to_numpy())
    m["top5_hit_rate"] = priority_hit_rate(pred, model)
    by_day = pred.groupby("date")[["y", model]].sum()
    m["building_day_mae"] = float((by_day[model] - by_day["y"]).abs().mean())
    if model != "benchmark":
        base = metrics(pred["y"].to_numpy(), pred["benchmark"].to_numpy())["mae"]
        m["skill_vs_benchmark"] = 1 - m["mae"] / base if base else np.nan
    return m


def check_leakage(train: pd.DataFrame, test: pd.DataFrame, preds: dict[str, np.ndarray]) -> None:
    assert train["date"].max() < test["date"].min(), "training days must precede test days"
    assert len({len(p) for p in preds.values()} | {len(test)}) == 1, "models must score identical days"
    for name, values in preds.items():
        assert np.isfinite(values).all(), f"{name}: non-finite predictions"
        assert (values >= 0).all(), f"{name}: negative predictions"
    assert test["lag_1d_missing"].isin([0, 1]).all()


def run_fold(d: pd.DataFrame, fold: Fold, models: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    train, test = split(d, fold)
    typical = train.groupby(["zone_id", "day_type"])["y"].median()
    keys = pd.MultiIndex.from_frame(test[["zone_id", "day_type"]])
    typical_kwh = pd.Series(typical.reindex(keys).to_numpy(), index=test.index)
    typical_kwh = typical_kwh.fillna(test["zone_id"].map(train.groupby("zone_id")["y"].median()))
    typical_kwh = typical_kwh.fillna(float(train["y"].median()))
    preds: dict[str, np.ndarray] = {"benchmark": predict_benchmark(train, test)}
    info = {"stage": fold.stage, "test_month": fold.test_month, "train_rows": len(train),
            "test_rows": len(test), "train_last": str(train["date"].max())}
    if "boosting" in models:
        preds["boosting"] = predict_boosting(fit_boosting(train), test)
    if "ridge" in models:
        preds["ridge"], info["ridge_negative_share"] = predict_ridge(fit_ridge(train), test)
    check_leakage(train, test, preds)
    out = test[["zone_id", "floor", "zone", "date", "day_type", "month", "y"]].copy()
    out["typical_kwh"] = typical_kwh.to_numpy()
    for name, values in preds.items():
        out[name] = values
    out["stage"] = fold.stage
    return out, info


def weekly_block_diagnostic(d: pd.DataFrame, weeks_per_season: int = 2) -> pd.DataFrame:
    """Secondary seasonally stratified week test; later dates can enter training."""
    end = pd.Timestamp(f"{config.QB_DEV_TEST_MONTHS[-1]}-01") + pd.offsets.MonthEnd(1)
    rows = d[d["y"].notna() & d["date"].between(pd.Timestamp(config.QB_TRAIN_START), end)]
    mondays = pd.date_range(config.QB_TRAIN_START, end, freq="W-MON")
    result = []
    for season in ("Cool", "Hot", "Rainy"):
        candidates = [m for m in mondays if config.SEASON_BY_MONTH[m.month] == season
                      and m + pd.Timedelta(days=6) <= end]
        selected = ([candidates[i] for i in np.linspace(0, len(candidates) - 1,
                                                       weeks_per_season).round().astype(int)] if candidates else [])
        for start in selected:
            stop = start + pd.Timedelta(days=7)
            test = rows[(rows["date"] >= start) & (rows["date"] < stop)]
            train = rows[(rows["date"] < start - pd.Timedelta(days=1)) |
                         (rows["date"] >= stop + pd.Timedelta(days=7))]
            boosted = predict_boosting(fit_boosting(train), test)
            base = predict_benchmark(train, test)
            result.append({"season": season, "week_start": start.date().isoformat(),
                           "boosting_mae": metrics(test["y"].to_numpy(), boosted)["mae"],
                           "benchmark_mae": metrics(test["y"].to_numpy(), base)["mae"],
                           "test_rows": len(test)})
    return pd.DataFrame(result)


def residual_bands(dev_pred: pd.DataFrame, model: str) -> pd.DataFrame:
    """Daily residual 5th–95th percentiles from development predictions only."""
    r = dev_pred.assign(residual=dev_pred["y"] - dev_pred[model])
    return r.groupby(["zone_id", "day_type"])["residual"].agg(
        res_lo=lambda s: s.quantile(0.05), res_hi=lambda s: s.quantile(0.95), n="size").reset_index()

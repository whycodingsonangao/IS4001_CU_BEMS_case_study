"""Within-season, leave-one-month-out folds (replaces the Jan-Aug / Sep-Dec block split).

Seasons use whole months (see config.SEASON_BY_MONTH). For each season, each month is held
out once and the model is trained on the *other months of the same season*, so a test month
is never judged against a different climate regime.

2018 Jul-Dec can be added as extra rainy/cool months with ``include_2018=True``. That is a
methodology change from the proposal and needs team agreement before it is used for results.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config


@dataclass(frozen=True)
class Fold:
    season: str
    test: str  # "YYYY-MM"
    train: tuple[str, ...]


def season_months(include_2018: bool = False) -> dict[str, list[str]]:
    months = pd.period_range("2019-01", "2019-12", freq="M")
    if include_2018:
        months = pd.period_range("2018-07", "2019-12", freq="M")
    out: dict[str, list[str]] = {}
    for p in months:
        out.setdefault(config.SEASON_BY_MONTH[p.month], []).append(str(p))
    return out


def make_folds(include_2018: bool = False) -> list[Fold]:
    folds = []
    for season, months in season_months(include_2018).items():
        for m in months:
            folds.append(Fold(season, m, tuple(x for x in months if x != m)))
    return folds


def fold_coverage(zone_day: pd.DataFrame, group: str = "Model 1",
                  include_2018: bool = False) -> pd.DataFrame:
    """Usable zone-days per (fold test month, zone) - shows which folds are trustworthy."""
    zd = zone_day[zone_day["group"] == group].copy()
    zd["ym"] = zd["date"].dt.to_period("M").astype(str)
    rows = []
    for f in make_folds(include_2018):
        t = zd[zd["ym"] == f.test]
        tr = zd[zd["ym"].isin(f.train)]
        rows.append({
            "season": f.season,
            "test_month": f.test,
            "train_months": ", ".join(f.train),
            "test_usable_days": int(t["usable"].sum()),
            "train_usable_days": int(tr["usable"].sum()),
            "test_usable_pct": 100 * t["usable"].mean() if len(t) else 0.0,
        })
    return pd.DataFrame(rows)

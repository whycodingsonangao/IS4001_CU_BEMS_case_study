"""Question A: k-means clustering of complete zone-days (methodology plan, section 4).

* One row = one zone on one day. Features = 24 hourly kWh per end use (C1: 72, C2: 48).
* Zones in ``config.QA_SEPARATE_ZONES`` (F1Z2, 24/7) are not clustered; see :func:`separate_profiles`.
* A day is used only if all 24 hours are valid for every end use in its group.
* Scaling: each zone x end-use block is divided by its 95th-percentile hourly kWh (floor 0.1 kWh).
* Calendar variables are never features; they label clusters afterwards (as does weather).
* k in 3..8 chosen by, in order: (1) every cluster >= 3 % of zone-days and in >= 3 zones,
  (2) half-year ARI >= 0.8, (3) highest silhouette, (4) interpretable centroids (human check).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from . import config
from .aggregate import valid_kwh

SEED = 0
SIL_SAMPLE = 6000
WARD_SAMPLE = 4000


@dataclass
class Matrix:
    group: str
    uses: list[str]
    meta: pd.DataFrame  # floor, zone, zone_id, date + calendar/weather metadata, one row per zone-day
    kwh: np.ndarray  # actual kWh, shape (n, 24 * len(uses))
    columns: list[str]


def feature_matrix(hourly: pd.DataFrame, zone_day: pd.DataFrame, group: str,
                   min_minutes: int = config.MIN_HOUR_MINUTES, year: int = config.YEAR) -> Matrix:
    uses = config.GROUP_USES[group]
    flag = "usable" if min_minutes == config.MIN_HOUR_MINUTES else "usable_strict"
    days = zone_day[(zone_day["group"] == group) & zone_day[flag] & (zone_day["year"] == year)
                    & ~zone_day["zone_id"].isin(config.QA_SEPARATE_ZONES)]
    keys = days[["floor", "zone", "date"]]
    h = hourly[hourly["timestamp"].dt.year == year].copy()
    h["date"] = h["timestamp"].dt.normalize()
    h["hour"] = h["timestamp"].dt.hour
    h = h.merge(keys, on=["floor", "zone", "date"])
    blocks, cols = [], []
    for use in uses:
        h[use] = valid_kwh(h, use, min_minutes)
        p = h.pivot_table(index=["floor", "zone", "date"], columns="hour", values=use)
        blocks.append(p)
        cols += [f"{use}_{hr:02d}" for hr in range(24)]
    wide = pd.concat(blocks, axis=1).dropna()
    meta = wide.index.to_frame(index=False).merge(days, on=["floor", "zone", "date"], how="left")
    return Matrix(group, uses, meta, wide.to_numpy(dtype=float), cols)


def separate_profiles(hourly: pd.DataFrame, zone_day: pd.DataFrame,
                      min_minutes: int = config.MIN_HOUR_MINUTES, year: int = config.YEAR) -> pd.DataFrame:
    """Mean hourly kWh by day type and end use for zones reported outside the clustering, plus day totals."""
    flag = "usable" if min_minutes == config.MIN_HOUR_MINUTES else "usable_strict"
    days = zone_day[zone_day["zone_id"].isin(config.QA_SEPARATE_ZONES) & zone_day[flag]
                    & (zone_day["year"] == year)]
    h = hourly[hourly["timestamp"].dt.year == year].assign(date=lambda d: d["timestamp"].dt.normalize(),
                                                           hour=lambda d: d["timestamp"].dt.hour)
    h = h.merge(days[["floor", "zone", "date", "zone_id", "group", "day_type"]], on=["floor", "zone", "date"])
    rows = []
    for use in config.END_USES:
        if f"{use.lower()}_kwh" not in h:
            continue
        v = h.assign(kwh=valid_kwh(h, use, min_minutes)).dropna(subset=["kwh"])
        g = v.groupby(["zone_id", "group", "day_type", "hour"]).agg(mean_kwh=("kwh", "mean"),
                                                                      days=("date", "nunique")).reset_index()
        rows.append(g.assign(use=use))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def p95_scale(m: Matrix) -> tuple[np.ndarray, pd.DataFrame]:
    """Divide each zone x end-use block by its 95th-percentile hourly kWh (floored)."""
    X = m.kwh.copy()
    rows = []
    for zid, idx in m.meta.groupby("zone_id").indices.items():
        for j, use in enumerate(m.uses):
            sl = slice(24 * j, 24 * (j + 1))
            p95 = float(np.percentile(m.kwh[idx, sl], 95))
            s = max(p95, config.QA_P95_FLOOR_KWH)
            X[idx, sl] = m.kwh[idx, sl] / s
            rows.append({"zone_id": zid, "use": use, "p95_kwh": p95, "scale_kwh": s, "floored": p95 < s})
    return X, pd.DataFrame(rows)


def zscore_scale(m: Matrix) -> np.ndarray:
    """z-score within each zone x end-use block (the plan's screening comparator)."""
    X = m.kwh.copy()
    for idx in m.meta.groupby("zone_id").indices.values():
        for j in range(len(m.uses)):
            sl = slice(24 * j, 24 * (j + 1))
            b = m.kwh[idx, sl]
            X[idx, sl] = (b - b.mean()) / (b.std() or 1.0)
    return X


def _kmeans(X: np.ndarray, k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(X, labels, sample_size=min(SIL_SAMPLE, len(X)), random_state=SEED))


def half_year_ari(X: np.ndarray, meta: pd.DataFrame, k: int) -> float:
    """Fit on Jan-Jun and on Jul-Dec separately, assign every day with both, compare."""
    h1 = (meta["half"] == "H1").to_numpy()
    if h1.sum() < k or (~h1).sum() < k:
        return float("nan")
    a = _kmeans(X[h1], k).predict(X)
    b = _kmeans(X[~h1], k).predict(X)
    return float(adjusted_rand_score(a, b))


def evaluate_k(X: np.ndarray, meta: pd.DataFrame, k: int) -> dict:
    km = _kmeans(X, k)
    lab = km.labels_
    share = np.bincount(lab, minlength=k) / len(lab)
    zones_per = meta.assign(c=lab).groupby("c")["zone_id"].nunique().reindex(range(k), fill_value=0)
    return {
        "k": k,
        "min_share": float(share.min()),
        "min_zones": int(zones_per.min()),
        "rule1_size": bool(share.min() >= config.QA_MIN_CLUSTER_SHARE
                           and zones_per.min() >= config.QA_MIN_CLUSTER_ZONES),
        "half_year_ari": half_year_ari(X, meta, k),
        "silhouette": _silhouette(X, lab),
        "inertia": float(km.inertia_),
    }


def choose_k(table: pd.DataFrame) -> tuple[int, str]:
    t = table.copy()
    t["rule2_stable"] = t["half_year_ari"] >= config.QA_MIN_HALF_YEAR_ARI
    both = t[t["rule1_size"] & t["rule2_stable"]]
    if len(both):
        return int(both.loc[both["silhouette"].idxmax(), "k"]), "passes rules 1-2; highest silhouette"
    r1 = t[t["rule1_size"]]
    if len(r1):
        return int(r1.loc[r1["silhouette"].idxmax(), "k"]), "no k passes rule 2; best silhouette among rule-1 passes"
    return int(t.loc[t["silhouette"].idxmax(), "k"]), "no k passes rule 1; highest silhouette (review!)"


def screening(m: Matrix, k: int = 4) -> pd.DataFrame:
    """Replicates the plan's scaling screen: raw vs z-score vs 95th percentile at fixed k."""
    out = []
    for name, X in [("Raw kWh", m.kwh), ("z-score", zscore_scale(m)), ("95th percentile", p95_scale(m)[0])]:
        lab = _kmeans(X, k).labels_
        dom = m.meta.assign(c=lab).groupby("zone_id")["c"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
        out.append({"scaling": name, "k": k, "zones_90pct_one_cluster": int((dom >= 0.9).sum()),
                    "n_zones": int(dom.size), "silhouette": _silhouette(X, lab),
                    "half_year_ari": half_year_ari(X, m.meta, k)})
    return pd.DataFrame(out)


def ward_ari(X: np.ndarray, labels: np.ndarray, k: int) -> float:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X), size=min(WARD_SAMPLE, len(X)), replace=False)
    w = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X[idx])
    return float(adjusted_rand_score(labels[idx], w))


def run(hourly: pd.DataFrame, zone_day: pd.DataFrame, group: str,
        min_minutes: int = config.MIN_HOUR_MINUTES) -> dict[str, pd.DataFrame]:
    m = feature_matrix(hourly, zone_day, group, min_minutes)
    X, scalers = p95_scale(m)
    ktab = pd.DataFrame([evaluate_k(X, m.meta, k) for k in config.QA_K_RANGE])
    k, why = choose_k(ktab)
    ktab["chosen"] = ktab["k"] == k
    ktab["rule2_stable"] = ktab["half_year_ari"] >= config.QA_MIN_HALF_YEAR_ARI

    km = _kmeans(X, k)
    # order clusters by size so "Cluster 1" is the most common pattern
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    lab = remap[km.labels_]
    centres = km.cluster_centers_[order]
    dist = np.linalg.norm(X - centres[lab], axis=1)

    labels = m.meta[["floor", "zone", "zone_id", "date", "month", "dow", "day_type", "is_holiday", "season",
                     "half", *[c for c in m.meta.columns if c.startswith("wx_")]]].copy()
    labels["cluster"] = lab + 1
    labels["distance"] = dist
    thr = labels.groupby("cluster")["distance"].transform(lambda s: np.percentile(s, config.QA_ATYPICAL_PCTL))
    labels["atypical"] = labels["distance"] > thr
    labels["group"] = group
    for j, use in enumerate(m.uses):
        labels[f"{use.lower()}_kwh"] = m.kwh[:, 24 * j:24 * (j + 1)].sum(axis=1)

    # centroids: scaled (what k-means sees) and actual mean kWh profile per cluster
    rows = []
    for c in range(k):
        sel = lab == c
        for j, use in enumerate(m.uses):
            sl = slice(24 * j, 24 * (j + 1))
            for hr in range(24):
                rows.append({"group": group, "cluster": c + 1, "use": use, "hour": hr,
                             "scaled": float(centres[c, sl][hr]),
                             "mean_kwh": float(m.kwh[sel, sl][:, hr].mean())})
    cent = pd.DataFrame(rows)

    # actual kWh per cluster x zone x floor x end use
    long = labels.melt(id_vars=["group", "cluster", "zone_id", "floor", "date"],
                       value_vars=[f"{u.lower()}_kwh" for u in m.uses], var_name="use", value_name="kwh")
    long["use"] = long["use"].str.replace("_kwh", "").str.capitalize().replace({"Ac": "AC"})
    long["in_floor_window"] = long["date"] <= pd.Timestamp(config.FLOOR_COMPARE_END)
    kwh = long.groupby(["group", "cluster", "zone_id", "floor", "use", "in_floor_window"], as_index=False).agg(
        kwh=("kwh", "sum"), days=("date", "nunique"))

    ktab["group"] = group
    ktab["reason"] = np.where(ktab["chosen"], why, "")
    ktab["ward_ari"] = np.where(ktab["chosen"], ward_ari(X, lab, k), np.nan)
    scr = screening(m).assign(group=group)
    scalers["group"] = group
    return {"labels": labels, "centroids": cent, "kselect": ktab, "kwh": kwh, "screening": scr,
            "scalers": scalers}

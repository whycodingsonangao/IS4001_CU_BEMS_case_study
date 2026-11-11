"""Question A: cluster complete zone-days for C1 and C2 and write the result tables.

    python scripts/run_question_a.py            # 57/60 rule, then the strict 60/60 sensitivity run
    python scripts/run_question_a.py --main-only

Outputs: data/processed/qa/ and data/processed/qa_strict/ (labels, centroids, kselect, kwh, screening, scalers,
separate = hourly profiles of zones reported outside the clustering, e.g. F1Z2).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from cubems import config
from cubems.clustering import run, separate_profiles


def one(min_minutes: int, hourly: pd.DataFrame, zd: pd.DataFrame) -> None:
    out = config.results_dir("qa", min_minutes)
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, list[pd.DataFrame]] = {}
    for group in ("C1", "C2"):
        t = time.perf_counter()
        res = run(hourly, zd, group, min_minutes)
        for name, df in res.items():
            tables.setdefault(name, []).append(df)
        ch = res["kselect"].loc[res["kselect"]["chosen"]].iloc[0]
        print(f"  [{min_minutes}/60] {group}: {len(res['labels']):,} zone-days, k={int(ch['k'])} "
              f"(silhouette {ch['silhouette']:.2f}, half-year ARI {ch['half_year_ari']:.2f}, "
              f"Ward ARI {ch['ward_ari']:.2f}) - {ch['reason']}  ({time.perf_counter() - t:.0f}s)")
    for name, dfs in tables.items():
        pd.concat(dfs, ignore_index=True).to_parquet(out / f"{name}.parquet", index=False)
    separate_profiles(hourly, zd, min_minutes).to_parquet(out / "separate.parquet", index=False)
    print(f"  [{min_minutes}/60] reported separately: {', '.join(config.QA_SEPARATE_ZONES)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true", help="skip the strict 60/60 sensitivity run")
    args = ap.parse_args()
    hourly = pd.read_parquet(config.HOURLY_PATH)
    zd = pd.read_parquet(config.ZONE_DAY_PATH)
    one(config.MIN_HOUR_MINUTES, hourly, zd)
    if not args.main_only:
        one(config.STRICT_HOUR_MINUTES, hourly, zd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

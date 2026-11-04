"""Question B: day-ahead AC prediction, model decision, final test and the exception layer.

    python scripts/run_question_b.py                 # 57/60 rule, then the strict 60/60 sensitivity run
    python scripts/run_question_b.py --main-only
    python scripts/run_question_b.py --skip-diagnostic

Outputs (data/processed/qb/ and qb_strict/): predictions, metrics, zone_metrics, decision.json,
diagnostic, bands, scored, episodes, kpis.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd

from cubems import config
from cubems import forecast as F
from cubems.anomaly import detect_episodes, kpis

MARGIN_KWH = 0.25  # locked before the final test
MIN_HOURS = 2


def summarise(pred: pd.DataFrame, models: list[str], label: str) -> list[dict]:
    rows = []
    for m in models:
        rows.append({"subset": label, "model": m, **F.score(pred, m)})
    return rows


def one(min_minutes: int, hourly: pd.DataFrame, zd: pd.DataFrame, diagnostic: bool) -> None:
    out = config.results_dir("qb", min_minutes)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"[{min_minutes}/60]"
    d = F.design(hourly, zd, min_minutes)
    print(f"  {tag} valid AC zone-hours: {int(d['y'].notna().sum()):,}")

    # -------- development folds: benchmark, boosting, ridge
    preds, infos = [], []
    for f in [f for f in F.folds() if f.stage == "development"]:
        t = time.perf_counter()
        p, info = F.run_fold(d, f, ("boosting", "ridge"))
        preds.append(p)
        infos.append(info)
        print(f"  {tag} dev {f.test_month}: {len(p):,} rows ({time.perf_counter() - t:.0f}s)")
    dev = pd.concat(preds, ignore_index=True)
    rows = summarise(dev, ["benchmark", "boosting", "ridge"], "development")
    for f_m in dev["month"].unique():
        rows += summarise(dev[dev["month"] == f_m], ["benchmark", "boosting", "ridge"], f"development {f_m}")
    dm = {r["model"]: r for r in rows if r["subset"] == "development"}

    # -------- decision rule (locked before the final test)
    boost_ok = dm["boosting"]["mae"] < dm["ridge"]["mae"] and abs(dm["boosting"]["nmbe"]) <= config.QB_NMBE_LIMIT
    chosen = "boosting" if boost_ok else "ridge"
    decision = {
        "chosen": chosen,
        "rule": f"boosting primary iff dev MAE < Ridge dev MAE and |NMBE| <= {config.QB_NMBE_LIMIT}%",
        "dev_mae": {k: v["mae"] for k, v in dm.items()},
        "dev_nmbe": {k: v["nmbe"] for k, v in dm.items()},
        "ridge_negative_share": float(pd.Series([i.get("ridge_negative_share", 0) for i in infos]).mean()),
        "min_minutes": min_minutes,
        "exception_margin_kwh": MARGIN_KWH,
        "exception_min_hours": MIN_HOURS,
    }
    bands = F.residual_bands(dev, chosen)

    # -------- final test: only the chosen model (plus the benchmark it must beat)
    fin = []
    for f in [f for f in F.folds() if f.stage == "final"]:
        p, info = F.run_fold(d, f, (chosen,))
        fin.append(p)
        infos.append(info)
        print(f"  {tag} final {f.test_month}: {len(p):,} rows")
    final = pd.concat(fin, ignore_index=True)
    rows += summarise(final, ["benchmark", chosen], "final (all floors)")
    rows += summarise(final[final["floor"] != 6], ["benchmark", chosen], "final (excl. Floor 6)")
    metrics = pd.DataFrame(rows)
    fm = metrics[(metrics["subset"] == "final (all floors)") & (metrics["model"] == chosen)].iloc[0]
    decision["final_mae"] = float(fm["mae"])
    decision["final_skill_vs_benchmark"] = float(fm["skill_vs_benchmark"])

    zone_rows = []
    for stage, df, models in (("development", dev, ["benchmark", "boosting", "ridge"]),
                              ("final", final, ["benchmark", chosen])):
        for (zid, fl), g in df.groupby(["zone_id", "floor"]):
            for m in models:
                zone_rows.append({"stage": stage, "zone_id": zid, "floor": fl, "model": m,
                                  **F.metrics(g["y"].to_numpy(), g[m].to_numpy())})

    # -------- exception layer on the final test
    sc = final.merge(bands, on=["zone_id", "hour", "day_type"], how="left")
    sc["pred"] = sc[chosen]
    sc["lo"] = (sc["pred"] + sc["res_lo"]).clip(lower=0)
    sc["hi"] = sc["pred"] + sc["res_hi"]
    sc = sc.dropna(subset=["lo", "hi"])
    sc["zone"] = sc["zone_id"].str.split("Z").str[1].astype(int)
    scored, episodes = detect_episodes(sc, MARGIN_KWH, MIN_HOURS, value="y")
    k = kpis(scored, episodes)
    decision["episodes"] = int(len(episodes))
    decision["zones_with_episodes"] = int(episodes["zone_id"].nunique())
    print(f"  {tag} chosen: {chosen} | dev MAE boost {dm['boosting']['mae']:.3f} ridge {dm['ridge']['mae']:.3f} "
          f"bench {dm['benchmark']['mae']:.3f} | final MAE {fm['mae']:.3f} skill {fm['skill_vs_benchmark']:.1%} | "
          f"{len(episodes)} exception episodes")

    if diagnostic and min_minutes == config.MIN_HOUR_MINUTES:
        t = time.perf_counter()
        F.weekly_block_diagnostic(d).to_parquet(out / "diagnostic.parquet", index=False)
        print(f"  {tag} weekly-block diagnostic done ({time.perf_counter() - t:.0f}s)")

    pd.concat([dev, final], ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    metrics.to_parquet(out / "metrics.parquet", index=False)
    pd.DataFrame(zone_rows).to_parquet(out / "zone_metrics.parquet", index=False)
    pd.DataFrame(infos).to_parquet(out / "folds.parquet", index=False)
    bands.to_parquet(out / "bands.parquet", index=False)
    scored.drop(columns=["run"]).to_parquet(out / "scored.parquet", index=False)
    episodes.to_parquet(out / "episodes.parquet", index=False)
    k.to_parquet(out / "kpis.parquet", index=False)
    (out / "decision.json").write_text(json.dumps(decision, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-only", action="store_true", help="skip the strict 60/60 sensitivity run")
    ap.add_argument("--skip-diagnostic", action="store_true", help="skip the weekly-block CV diagnostic")
    args = ap.parse_args()
    hourly = pd.read_parquet(config.HOURLY_PATH)
    zd = pd.read_parquet(config.ZONE_DAY_PATH)
    one(config.MIN_HOUR_MINUTES, hourly, zd, not args.skip_diagnostic)
    if not args.main_only:
        one(config.STRICT_HOUR_MINUTES, hourly, zd, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

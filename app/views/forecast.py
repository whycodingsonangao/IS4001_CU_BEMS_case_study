import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import EMPHASIS, MUTED, SERIES, card, load_qb, load_qb_decision, need, page_header, show, style, zone_key

from cubems import config

page_header("Day-ahead AC forecast",
            "How well can each zone's hourly AC use for the next day be predicted? Forecast issued 00:00 on day D for all "
            "24 hours of D. ERA5 weather stands in for a forecast, so accuracy is an upper bound.",
            eyebrow="Question B · prediction")

dec = need(load_qb_decision(), "run_question_b.py")
metrics, zm, pred = load_qb("metrics"), load_qb("zone_metrics"), load_qb("predictions")
diag, folds = load_qb("diagnostic"), load_qb("folds")
chosen = dec["chosen"]
LBL = {"benchmark": "Same hour, 7 days earlier", "boosting": "Gradient boosting (Poisson, monotone)",
       "ridge": "Ridge (floored at 0)"}

if "cvrmse" not in metrics.columns:
    metrics = metrics.copy()
    pred_dates = pd.to_datetime(pred["date"])

    def mean_actual(subset: str) -> float:
        if subset.startswith("development"):
            selected = pred[pred["stage"] == "development"]
            if subset != "development":
                month = subset.removeprefix("development ")
                selected = selected[pred_dates.loc[selected.index].dt.strftime("%Y-%m") == month]
        elif subset == "final (excl. Floor 6)":
            selected = pred[(pred["stage"] == "final") & (pred["floor"] != 6)]
        else:
            selected = pred[pred["stage"] == "final"]
        return float(selected["y"].mean())

    metrics["cvrmse"] = [
        100 * rmse / mean_actual(subset)
        for subset, rmse in zip(metrics["subset"], metrics["rmse"])
    ]

fin = metrics[(metrics["subset"] == "final (all floors)") & (metrics["model"] == chosen)].iloc[0]
fin_b = metrics[(metrics["subset"] == "final (all floors)") & (metrics["model"] == "benchmark")].iloc[0]
c = st.columns(4)
c[0].metric("Model sent to final test", chosen.capitalize(), border=True, help=dec["rule"])
c[1].metric("Final-test MAE (kWh per zone-hour)", f"{fin['mae']:.2f}", f"{fin['mae'] - fin_b['mae']:+.2f} vs benchmark",
            delta_color="inverse", border=True)
c[2].metric("Skill vs benchmark", f"{fin['skill_vs_benchmark']:.1%}", border=True,
            help="1 − MAE / benchmark MAE on identical targets")
c[3].metric("Final-test bias (NMBE)", f"{fin['nmbe']:+.1f}%", border=True)

# ---------------------------------------------------------------- decision
dev = metrics[metrics["subset"] == "development"].set_index("model")
ok_mae = dev.loc["boosting", "mae"] < dev.loc["ridge", "mae"]
ok_bias = abs(dev.loc["boosting", "nmbe"]) <= config.QB_NMBE_LIMIT
with card("Model decision on the development folds (May–Aug, expanding window, monthly refit)",
          f"Boosting {'beats' if ok_mae else 'does not beat'} Ridge on MAE ({dev.loc['boosting', 'mae']:.3f} vs "
          f"{dev.loc['ridge', 'mae']:.3f}) and its |NMBE| is {abs(dev.loc['boosting', 'nmbe']):.1f}% "
          f"({'≤' if ok_bias else '>'} {config.QB_NMBE_LIMIT:.0f}%), so <b>{chosen}</b> goes to the final test. "
          f"Against the benchmark it is {1 - dev.loc['boosting', 'mae'] / dev.loc['benchmark', 'mae']:+.1%} on MAE.",
          f"Ridge produced negative predictions for {dec['ridge_negative_share']:.0%} of rows before flooring at 0."):
    t = metrics[metrics["subset"].str.startswith("development")].copy()
    t["model"] = t["model"].map(LBL)
    t["skill_vs_benchmark"] = 100 * t["skill_vs_benchmark"]
    st.dataframe(t[["subset", "model", "mae", "cvrmse", "nmbe", "skill_vs_benchmark", "zone_day_mae",
                    "building_hour_mae", "n"]], hide_index=True, width="stretch", height=300, column_config={
        "subset": "Fold", "model": "Model",
        "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
        "cvrmse": st.column_config.NumberColumn("CV(RMSE) %", format="%.1f%%"),
        "nmbe": st.column_config.NumberColumn("NMBE %", format="%+.1f"),
        "skill_vs_benchmark": st.column_config.NumberColumn("Skill vs bench.", format="%+.1f%%"),
        "zone_day_mae": st.column_config.NumberColumn("Zone-day MAE", format="%.1f"),
        "building_hour_mae": st.column_config.NumberColumn("Building-hour MAE", format="%.1f"),
        "n": st.column_config.NumberColumn("Zone-hours")})

with card("Final test (Sep–Dec, frozen model choice)", "Reported with and without Floor 6, whose data ends 22 Oct."):
    t = metrics[metrics["subset"].str.startswith("final")].copy()
    t["model"] = t["model"].map(LBL)
    t["skill_vs_benchmark"] = 100 * t["skill_vs_benchmark"]
    st.dataframe(t[["subset", "model", "mae", "cvrmse", "nmbe", "skill_vs_benchmark", "zone_day_mae",
                    "building_hour_mae", "n"]], hide_index=True, width="stretch", column_config={
        "subset": "Subset", "model": "Model",
        "mae": st.column_config.NumberColumn("MAE", format="%.3f"),
        "cvrmse": st.column_config.NumberColumn("CV(RMSE) %", format="%.1f%%"),
        "nmbe": st.column_config.NumberColumn("NMBE %", format="%+.1f"),
        "skill_vs_benchmark": st.column_config.NumberColumn("Skill vs bench.", format="%+.1f%%"),
        "zone_day_mae": st.column_config.NumberColumn("Zone-day MAE", format="%.1f"),
        "building_hour_mae": st.column_config.NumberColumn("Building-hour MAE", format="%.1f"),
        "n": st.column_config.NumberColumn("Zone-hours")})

# ---------------------------------------------------------------- per-zone
z = zm[zm["stage"] == "final"].pivot_table(index="zone_id", columns="model", values="mae")
z["skill"] = 1 - z[chosen] / z["benchmark"]
z = z.loc[sorted(z.index, key=zone_key)].reset_index()
worse = z[z["skill"] < 0]["zone_id"].tolist()
with card("Final-test accuracy by zone", f"The model beats the benchmark in <b>{(z['skill'] > 0).sum()} of {len(z)}</b> "
          "zones" + (f"; it is worse in {', '.join(worse)}." if worse else ".")):
    fig = go.Figure(go.Bar(x=z["zone_id"], y=z["skill"], marker_color=[EMPHASIS if s > 0 else "#e34948" for s in z["skill"]],
                           customdata=z[[chosen, "benchmark"]],
                           hovertemplate="<b>%{x}</b><br>Skill %{y:.1%}<br>MAE %{customdata[0]:.2f} vs "
                                         "benchmark %{customdata[1]:.2f}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", title="Skill vs benchmark")
    fig.update_xaxes(tickangle=-90)
    show(style(fig, height=300, legend=False))

# ---------------------------------------------------------------- actual vs predicted
with card("Actual vs predicted for one zone"):
    f1, f2 = st.columns([1, 2])
    zones = sorted(pred["zone_id"].unique(), key=zone_key)
    zone = f1.selectbox("Zone", zones, index=zones.index("F2Z1") if "F2Z1" in zones else 0)
    zp = pred[pred["zone_id"] == zone]
    start = f2.date_input("Week starting", pd.Timestamp("2019-09-02"), min_value=zp["date"].min().date(),
                          max_value=zp["date"].max().date())
    t0 = pd.Timestamp(start)
    grid = pd.DataFrame({"timestamp": pd.date_range(t0, t0 + pd.Timedelta(hours=167), freq="h")})
    w = grid.merge(zp, on="timestamp", how="left")
    fig = go.Figure()
    fig.add_scatter(x=w["timestamp"], y=w["y"], name="Actual", mode="lines", line=dict(width=2, color=EMPHASIS),
                    hovertemplate="Actual %{y:.1f} kWh<extra></extra>")
    fig.add_scatter(x=w["timestamp"], y=w[chosen], name=f"Predicted ({chosen})", mode="lines",
                    line=dict(width=2, color=SERIES[1]), hovertemplate="Predicted %{y:.1f} kWh<extra></extra>")
    fig.add_scatter(x=w["timestamp"], y=w["benchmark"], name="Benchmark (7 days earlier)", mode="lines",
                    line=dict(width=1.2, color=MUTED, dash="dot"), hovertemplate="Benchmark %{y:.1f} kWh<extra></extra>")
    fig.update_layout(yaxis_title="AC kWh per hour", hovermode="x unified")
    fig.update_xaxes(hoverformat="%a %d %b, %H:00", tickformat="%a %d %b")
    show(style(fig, height=360))
    st.caption("Gaps are hours that were not valid (fewer than 57 recorded minutes); they are never filled.")

with st.expander("Diagnostic: weekly-block CV (seasonally stratified, 7-day purge, 1-day embargo)",
                 icon=":material/science:"):
    if diag is not None and len(diag):
        st.markdown("Runs once as a diagnostic only. It trains on later months, which a day-ahead forecaster cannot see, "
                    "so it is **not** used to choose the model.")
        st.dataframe(diag, hide_index=True, width="stretch", column_config={
            "boosting_mae": st.column_config.NumberColumn("Boosting MAE", format="%.3f"),
            "benchmark_mae": st.column_config.NumberColumn("Benchmark MAE", format="%.3f")})
with st.expander("Fold log (training rows, last training timestamp)"):
    st.dataframe(folds, hide_index=True, width="stretch")

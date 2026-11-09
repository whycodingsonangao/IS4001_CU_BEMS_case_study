"""Question B: next-day zone AC totals and the resulting priority list."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import EMPHASIS, MUTED, SERIES, card, load_qb, load_qb_decision, need, page_header, show, style, zone_key

page_header("Next-day AC priorities",
            "Forecast each AC zone's total kWh for the next operating day, issued at 00:00 before its readings exist. "
            "Use the forecast to decide which zones merit a schedule or equipment review first.",
            eyebrow="Question B · prediction")

dec = need(load_qb_decision(), "run_question_b.py")
metrics = need(load_qb("metrics"), "run_question_b.py")
zm = need(load_qb("zone_metrics"), "run_question_b.py")
pred = need(load_qb("predictions"), "run_question_b.py")
diag, folds = load_qb("diagnostic"), load_qb("folds")
chosen = dec["chosen"]
labels = {"benchmark": "Same zone, seven days earlier", "boosting": "Gradient boosting", "ridge": "Ridge"}
fin = metrics[(metrics.subset == "final (all floors)") & (metrics.model == chosen)].iloc[0]
base = metrics[(metrics.subset == "final (all floors)") & (metrics.model == "benchmark")].iloc[0]

c = st.columns(4)
c[0].metric("Selected daily model", labels[chosen], border=True)
c[1].metric("Final MAE", f"{fin.mae:.1f} kWh / zone-day", f"{fin.mae-base.mae:+.1f} vs last week",
            delta_color="inverse", border=True)
c[2].metric("Skill vs last week", f"{fin.skill_vs_benchmark:.1%}", border=True,
            help="1 − model MAE / seven-day benchmark MAE on the same complete zone-days")
c[3].metric("Top-five zone recall", f"{fin.top5_hit_rate:.1%}", border=True,
            help=f"Last-week benchmark: {base.top5_hit_rate:.1%}. Average overlap with the actual five "
                 "highest-use zones on each final-test day.")

final = pred[pred.stage == "final"].copy()
with card("Which zones need attention?", "Replay a test-day forecast. Rank zones by forecast AC kWh; "
          "the difference from each zone's historical typical day provides context.",
          "The typical value is computed from training days available before that test month. Actual use is not used "
          "to create this priority list."):
    pick = st.date_input("Forecast date", pd.Timestamp("2019-09-02"),
                         min_value=final.date.min().date(), max_value=final.date.max().date())
    day = final[final.date == pd.Timestamp(pick)].copy()
    day["above_typical_kwh"] = day[chosen] - day["typical_kwh"]
    day = day.sort_values(chosen, ascending=False)
    st.dataframe(day[["zone_id", chosen, "typical_kwh", "above_typical_kwh", "benchmark"]],
                 hide_index=True, width="stretch", height=400, column_config={
                     "zone_id": "Priority zone", chosen: st.column_config.NumberColumn("Forecast kWh", format="%.1f"),
                     "typical_kwh": st.column_config.NumberColumn("Typical kWh", format="%.1f"),
                     "above_typical_kwh": st.column_config.NumberColumn("Above typical", format="%+.1f"),
                     "benchmark": st.column_config.NumberColumn("Last-week kWh", format="%.1f")})

with card("Evaluation: complete zone-days", "The model is selected using May–Aug development folds. "
          "Sep–Dec is reported with and without Floor 6, whose data ends in October."):
    t = metrics[metrics.subset.isin(["development", "final (all floors)", "final (excl. Floor 6)"])].copy()
    t["model"] = t.model.map(labels)
    t["skill_vs_benchmark"] = 100 * t["skill_vs_benchmark"]
    st.dataframe(t[["subset", "model", "mae", "skill_vs_benchmark", "top5_hit_rate", "n"]],
                 hide_index=True, width="stretch",
                 column_config={"subset": "Period", "model": "Model",
                                "mae": st.column_config.NumberColumn("MAE · kWh/zone-day", format="%.1f"),
                                "skill_vs_benchmark": st.column_config.NumberColumn("Skill vs last week", format="%+.1f%%"),
                                "top5_hit_rate": st.column_config.NumberColumn("Recall@5", format="%.1f%%"),
                                "n": st.column_config.NumberColumn("Complete zone-days")})
    with st.expander("Secondary metrics and monthly folds"):
        s = metrics.copy()
        s["model"] = s.model.map(labels)
        st.dataframe(s[["subset", "model", "rmse", "cvrmse", "nmbe", "building_day_mae"]],
                     hide_index=True, width="stretch", column_config={
                         "subset": "Period", "model": "Model",
                         "rmse": st.column_config.NumberColumn("RMSE · kWh/zone-day", format="%.1f"),
                         "cvrmse": st.column_config.NumberColumn("CV(RMSE)", format="%.1f%%"),
                         "nmbe": st.column_config.NumberColumn("Bias (NMBE)", format="%+.1f%%"),
                         "building_day_mae": st.column_config.NumberColumn("Building-day MAE · kWh", format="%.1f")})
        st.caption("CV(RMSE) divides RMSE by mean measured daily AC use. Use it to describe spread, not as a "
                   "pass/fail threshold for zone prioritization. Building-day totals cover the scored zones on each date.")

z = zm[zm.stage == "final"].pivot_table(index="zone_id", columns="model", values="mae")
z["skill"] = 1 - z[chosen] / z["benchmark"]
z = z.loc[sorted(z.index, key=zone_key)].reset_index()
with card("Does the model help across zones?", f"It beats last week's daily total in "
          f"<b>{(z.skill > 0).sum()} of {len(z)}</b> AC zones on final-test MAE."):
    fig = go.Figure(go.Bar(x=z.zone_id, y=z.skill, marker_color=[EMPHASIS if v > 0 else SERIES[1] for v in z.skill]))
    fig.update_xaxes(tickangle=-90)
    fig.update_yaxes(tickformat=".0%", title="MAE skill vs last week")
    show(style(fig, height=300, legend=False))

with card("Actual and predicted daily AC total"):
    zones = sorted(final.zone_id.unique(), key=zone_key)
    zone = st.selectbox("Zone", zones, index=zones.index("F2Z1") if "F2Z1" in zones else 0)
    q = final[final.zone_id == zone].sort_values("date")
    fig = go.Figure()
    fig.add_scatter(x=q.date, y=q.y, name="Actual", line=dict(color=EMPHASIS))
    fig.add_scatter(x=q.date, y=q[chosen], name="Forecast", line=dict(color=SERIES[1]))
    fig.add_scatter(x=q.date, y=q.benchmark, name="Last-week benchmark", line=dict(color=MUTED, dash="dot"))
    fig.update_layout(yaxis_title="AC kWh per complete day", hovermode="x unified")
    show(style(fig, height=340))

with st.expander("Diagnostic: seasonally stratified weekly blocks"):
    st.caption("This checks excluded weeks using both earlier and later development data; the chronological folds "
               "above are used for model selection and the final result.")
    if diag is not None:
        st.dataframe(diag, hide_index=True, width="stretch")
with st.expander("Chronological fold log"):
    if folds is not None:
        st.dataframe(folds, hide_index=True, width="stretch")

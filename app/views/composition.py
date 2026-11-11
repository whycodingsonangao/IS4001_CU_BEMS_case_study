import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import (
    DEEMPH,
    EMPHASIS,
    MONTHS,
    SEQ_BLUE,
    USE_COLORS,
    USE_LABELS,
    card,
    load_hourly,
    load_zone_day,
    page_header,
    show,
    style,
    usable_note,
)

from cubems import config

page_header("Where the electricity goes",
            "What drives consumption, and how does its composition vary across floors, end uses and "
            "recurring operating patterns?", eyebrow="Question A · descriptive")

zd = load_zone_day()

f1, f2, f3 = st.columns([3, 2, 2])
months = f1.select_slider("Months (2019)", options=list(range(1, 13)), value=(1, 12),
                          format_func=lambda m: MONTHS[m - 1])
day_types = f2.segmented_control("Day type", ["Workday", "Weekend/holiday"], selection_mode="multi",
                                 default=["Workday", "Weekend/holiday"])

d = zd[(zd["year"] == 2019) & zd["usable"] & zd["month"].between(*months) & zd["day_type"].isin(day_types or [])]
if d.empty:
    st.warning("No usable zone-days for this filter.")
    st.stop()
st.caption(usable_note() + (" Floor 6 data ends 22 Oct 2019." if months[1] >= 10 else ""))

uses = ["ac_kwh", "light_kwh", "plug_kwh"]
names = {"ac_kwh": "AC", "light_kwh": "Light", "plug_kwh": "Plug"}
tot_all = d[uses].sum()

overview_tab, priorities_tab, operations_tab = st.tabs(
    ["Overview", "Zone priorities", "Operating patterns"]
)

with overview_tab:
    k = st.columns(4)
    k[0].metric("Recorded electricity", f"{d['total_kwh'].sum() / 1e3:,.0f} MWh", border=True)
    for i, u in enumerate(uses, start=1):
        k[i].metric(f"{USE_LABELS[names[u]]} share", f"{tot_all[u] / tot_all.sum():.0%}", border=True)

    # ---------------------------------------------------------------- composition by floor
    dfl = d[d["date"] <= pd.Timestamp(config.FLOOR_COMPARE_END)]
    fl = dfl.groupby("floor")[uses].sum().div(1000).rename(columns=names)
    if fl.empty:
        st.info(f"The floor comparison covers 1 Jan – {pd.Timestamp(config.FLOOR_COMPARE_END):%d %b} only "
                "(Floor 6's last full day). Include a month up to October to see it.")
    else:
        top_floor = fl.sum(axis=1).idxmax()
        ac_floor = fl["AC"] / fl.sum(axis=1)
        with card("Electricity composition by floor (1 Jan – 21 Oct)",
                  f"Floor {top_floor} records the most electricity ({fl.sum(axis=1).max():,.0f} MWh). "
                  f"AC is {ac_floor.min():.0%}–{ac_floor.max():.0%} of each floor's total.",
                  f"1 Jan – {pd.Timestamp(config.FLOOR_COMPARE_END):%d %b} (Floor 6's last full day). F1Z4 (lighting only) "
                  "is in the totals; F1Z2 is Floor 1's only AC zone."):
            fig = go.Figure()
            for use in ["AC", "Light", "Plug"]:
                fig.add_bar(y=[f"Floor {f}" for f in fl.index], x=fl[use], name=USE_LABELS[use], orientation="h",
                            marker=dict(color=USE_COLORS[use], line=dict(width=1.5, color="rgba(255,255,255,0.9)")),
                            hovertemplate="%{y} · " + USE_LABELS[use] + ": %{x:,.1f} MWh<extra></extra>")
            tot = fl.sum(axis=1)
            fig.add_scatter(y=[f"Floor {f}" for f in fl.index], x=tot, mode="text",
                            text=[f"  {v:,.0f}" for v in tot],
                            textposition="middle right", showlegend=False, hoverinfo="skip")
            fig.update_layout(barmode="stack", xaxis_title="Recorded electricity (MWh)", bargap=0.35)
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(range=[0, tot.max() * 1.08])
            show(style(fig, height=330))

# ---------------------------------------------------------------- Pareto ranking
with priorities_tab:
    with card("Priority ranking: which zones carry the load",
              note="This control affects only the priority ranking chart and table below. "
                   "kWh per complete day normalises zones with different coverage, so Floor 6 is not under-ranked "
                   "just for having fewer days."):
        ranking_control, _ = st.columns([2, 1])
        with ranking_control:
            use_sel = st.segmented_control(
                "Rank zones by end use",
                ["Total", "AC", "Light", "Plug"],
                default="Total",
                key="ranking_end_use",
                help="Changes only the priority ranking chart and table in this card.",
            ) or "Total"

        col = "total_kwh" if use_sel == "Total" else f"{use_sel.lower()}_kwh"
        z = d.groupby("zone_id").agg(kwh=(col, "sum"), days=("usable", "sum")).dropna()
        z = z[z["kwh"] > 0]
        z["kwh_per_day"] = z["kwh"] / z["days"]
        z = z.sort_values("kwh", ascending=False)
        z["share"] = z["kwh"] / z["kwh"].sum()
        z["cum_share"] = z["share"].cumsum()
        n80 = int((z["cum_share"] < 0.8).sum() + 1)
        st.markdown(
            f"**{n80} of {len(z)} zones** account for 80% of recorded {use_sel.lower()} electricity. "
            f"Start with {', '.join(z.index[:3])}."
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            zz = z.reset_index()
            zz["group"] = ["Top 80% of load" if i < n80 else "Remaining zones" for i in range(len(zz))]
            fig = px.bar(zz, x="zone_id", y="share", color="group",
                         color_discrete_map={"Top 80% of load": EMPHASIS, "Remaining zones": DEEMPH},
                         labels={"zone_id": "", "share": "Share of recorded kWh"},
                         custom_data=["kwh", "cum_share", "kwh_per_day"])
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:.1%} of total · %{customdata[0]:,.0f} kWh"
                                            "<br>Cumulative %{customdata[1]:.0%}"
                                            "<br>%{customdata[2]:,.0f} kWh per usable day<extra></extra>")
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickangle=-90, tickfont=dict(size=11))
            show(style(fig, height=340))
        with c2:
            st.dataframe(
                z.reset_index()[["zone_id", "kwh", "kwh_per_day", "share", "cum_share"]]
                .assign(share=lambda t: 100 * t["share"], cum_share=lambda t: 100 * t["cum_share"]),
                hide_index=True, width="stretch", height=340,
                column_config={
                    "zone_id": st.column_config.TextColumn("Zone", width="small"),
                    "kwh": st.column_config.NumberColumn("kWh", format="%,.0f"),
                    "kwh_per_day": st.column_config.NumberColumn("kWh / day", format="%.0f"),
                    "share": st.column_config.ProgressColumn("Share", format="%.1f%%", min_value=0,
                                                             max_value=float(100 * z["share"].max())),
                    "cum_share": st.column_config.NumberColumn("Cumulative", format="%.0f%%"),
                })

# ---------------------------------------------------------------- recurring patterns
with operations_tab:
    h = load_hourly()
    with card("Recurring operating pattern: average load by hour and weekday") as _:
        c1, c2 = st.columns([2.2, 5.3])
        use_h = c1.segmented_control("End use", ["AC", "Light", "Plug"], default="AC", key="heat_use") or "AC"
        floors = c1.pills("Floors", list(range(1, 8)), selection_mode="multi", default=list(range(1, 8)),
                          format_func=lambda f: f"F{f}") or []
        keep = d.loc[d["floor"].isin(floors), ["floor", "zone", "date"]]
        hh = h[["floor", "zone", "timestamp", f"{use_h.lower()}_kwh"]].copy()
        hh["date"] = hh["timestamp"].dt.normalize()
        hh = hh.merge(keep, on=["floor", "zone", "date"])
        hh = hh.groupby("timestamp", as_index=False)[f"{use_h.lower()}_kwh"].sum()
        hh["hour"] = hh["timestamp"].dt.hour
        hh["dow"] = hh["timestamp"].dt.day_name().str[:3]
        piv = hh.pivot_table(index="dow", columns="hour", values=f"{use_h.lower()}_kwh", aggfunc="mean")
        piv = piv.reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        piv.columns = [f"{c:02d}:00" for c in piv.columns]
        with c2:
            if piv.notna().any().any():
                peak = piv.loc[["Mon", "Tue", "Wed", "Thu", "Fri"]].mean().idxmax()
                wkd = piv.loc[["Mon", "Tue", "Wed", "Thu", "Fri"]].to_numpy().mean()
                wk = piv.loc[["Sat", "Sun"]].to_numpy().mean() / wkd
                lbl = USE_LABELS[use_h] if use_h == "AC" else USE_LABELS[use_h].lower()
                st.markdown(f'<div class="takeaway">Workday {lbl} load peaks around <b>{peak}</b>; '
                            f'weekends run at <b>{wk:.0%}</b> of the workday average.</div>', unsafe_allow_html=True)
                fig = px.imshow(piv, color_continuous_scale=SEQ_BLUE, aspect="auto",
                                labels=dict(x="", y="", color="kW"))
                fig.update_traces(xgap=2, ygap=2,
                                  hovertemplate="%{y} %{x}<br>%{z:,.1f} kW average<extra></extra>")
                fig.update_xaxes(tickvals=[f"{i:02d}:00" for i in range(0, 24, 3)])
                show(style(fig, height=280, legend=False))
        st.caption("Each cell sums the hourly kWh of every selected zone, averaged over days in the filter. "
                   "Days where some zones were unusable contribute fewer zones, so read the shape, not the level.")

    # ---------------------------------------------------------------- after-hours load
    hz = h[["floor", "zone", "timestamp", *uses]].copy()
    hz["date"] = hz["timestamp"].dt.normalize()
    hz = hz.merge(d[["floor", "zone", "date", "zone_id", "day_type"]], on=["floor", "zone", "date"])
    hr = hz["timestamp"].dt.hour
    hz["after"] = (hz["day_type"] != "Workday") | (hr < 8) | (hr >= 18)
    hz["kwh"] = hz[uses].sum(axis=1, min_count=1)
    hz["after_kwh"] = hz["kwh"].where(hz["after"], 0)
    ah = hz.groupby("zone_id")[["after_kwh", "kwh"]].sum()
    ah["share"] = ah["after_kwh"] / ah["kwh"]
    ah = ah.sort_values("after_kwh", ascending=False).head(12).iloc[::-1]
    top = ah.index[-1]
    with card("After-hours load: outside 08:00–18:00 on workdays, all day on weekends and holidays",
              f"**{top}** uses {ah.loc[top, 'after_kwh'] / 1e3:,.0f} MWh after hours, "
              f"{ah.loc[top, 'share']:.0%} of its own total.",
              "Top 12 zones by after-hours kWh, the candidates for equipment-schedule flagging and "
              "essential-load review."):
        fig = go.Figure(go.Bar(
            y=ah.index, x=ah["after_kwh"] / 1e3, orientation="h", marker_color=EMPHASIS,
            text=[f"{s:.0%} of zone total" for s in ah["share"]], textposition="outside",
            textfont=dict(size=11), cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,.1f} MWh after hours<extra></extra>"))
        fig.update_layout(xaxis_title="After-hours electricity (MWh)", bargap=0.35)
        fig.update_xaxes(range=[0, ah["after_kwh"].max() / 1e3 * 1.25])
        show(style(fig, height=380, legend=False))

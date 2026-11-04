import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import SERIES, USE_LABELS, card, load_qa, need, page_header, show, style
from plotly.subplots import make_subplots

from cubems import config

page_header("Recurring daily patterns",
            "Which daily electricity patterns recur, and how much kWh does each account for? K-means on complete "
            "zone-days; 24 hourly kWh per end use, each zone × end use scaled by its own 95th percentile.",
            eyebrow="Question A · k-means")

kselect = need(load_qa("kselect"), "run_question_a.py")
labels, cent, kwh = load_qa("labels"), load_qa("centroids"), load_qa("kwh")
screen, scalers = load_qa("screening"), load_qa("scalers")

group = st.segmented_control("Zone group", ["C1", "C2"], default="C1",
                             help="C1: AC + lighting + plugs (24 zones, 72 features). "
                                  "C2: lighting + plugs (8 zones, 48 features).") or "C1"
ks = kselect[kselect["group"] == group].sort_values("k")
ch = ks[ks["chosen"]].iloc[0]
lab = labels[labels["group"] == group]
k = int(ch["k"])
names = [f"Cluster {i}" for i in range(1, k + 1)]

c = st.columns(4)
c[0].metric("Zone-days clustered", f"{len(lab):,}", border=True)
c[1].metric("Chosen k", k, border=True, help=ch["reason"])
c[2].metric("Silhouette", f"{ch['silhouette']:.2f}", border=True)
c[3].metric("Half-year ARI", f"{ch['half_year_ari']:.2f}", border=True,
            help=f"Ward cross-check ARI {ch['ward_ari']:.2f}")
if "review" in ch["reason"]:
    solo = lab.groupby("cluster")["zone_id"].nunique()
    solo_c = solo[solo < config.QA_MIN_CLUSTER_ZONES]
    zs = lab[lab["cluster"].isin(solo_c.index)]["zone_id"].unique()
    st.warning(f"No k in 3–8 passes rule 1 (every cluster ≥3% of zone-days **and** in ≥3 zones). "
               f"{', '.join(zs)} forms its own cluster at every k, so the most stable k with the best silhouette is "
               "shown. Plan open item 5: inspect this zone.", icon=":material/rule:")

# ---------------------------------------------------------------- centroids
cg = cent[cent["group"] == group]
uses = list(dict.fromkeys(cg["use"]))
share = lab["cluster"].value_counts(normalize=True).sort_index()
dt = pd.crosstab(lab["cluster"], lab["day_type"], normalize="index")
with card("What each pattern looks like (mean actual kWh per hour)",
          " · ".join(f"<b>Cluster {i}</b>: {share[i]:.0%} of days, "
                     f"{dt.loc[i].get('Workday', 0):.0%} workdays" for i in share.index)):
    fig = make_subplots(rows=1, cols=len(uses), subplot_titles=[USE_LABELS[u] for u in uses], shared_yaxes=False)
    for j, use in enumerate(uses, start=1):
        for i in range(k):
            s = cg[(cg["use"] == use) & (cg["cluster"] == i + 1)]
            fig.add_scatter(x=s["hour"], y=s["mean_kwh"], mode="lines", name=names[i], legendgroup=names[i],
                            showlegend=j == 1, line=dict(width=2, color=SERIES[i]),
                            hovertemplate=f"{names[i]} · {USE_LABELS[use]}<br>%{{x:02d}}:00 · %{{y:.2f}} kWh<extra></extra>",
                            row=1, col=j)
    fig.update_xaxes(tickvals=list(range(0, 24, 6)), ticktext=["00:00", "06:00", "12:00", "18:00"])
    fig.update_yaxes(title_text="kWh per hour", row=1, col=1)
    style(fig, height=360)
    fig.update_layout(legend=dict(y=-0.14, yanchor="top"), margin=dict(t=30, b=40))
    show(fig)

# ---------------------------------------------------------------- kWh by cluster
kg = kwh[kwh["group"] == group]
c1, c2 = st.columns(2)
with c1, card("Electricity by cluster and end use", "Actual kWh in each pattern, all of 2019 (complete days)."):
    t = kg.groupby(["cluster", "use"])["kwh"].sum().unstack().fillna(0) / 1e3
    fig = go.Figure()
    for use in uses:
        fig.add_bar(x=[f"Cluster {i}" for i in t.index], y=t[use], name=USE_LABELS[use],
                    marker_color={"AC": "#2a78d6", "Light": "#eb6834", "Plug": "#1baf7a"}[use],
                    hovertemplate="%{x} · " + USE_LABELS[use] + ": %{y:,.1f} MWh<extra></extra>")
    fig.update_layout(barmode="stack", yaxis_title="MWh")
    show(style(fig, height=320))
with c2, card("Floor comparison by cluster", f"1 Jan – {pd.Timestamp(config.FLOOR_COMPARE_END):%d %b} 2019 so Floor 6 "
              "is compared over the same days as the others."):
    f = kg[kg["in_floor_window"]].groupby(["floor", "cluster"])["kwh"].sum().unstack().fillna(0) / 1e3
    fig = go.Figure()
    for i in f.columns:
        fig.add_bar(y=[f"Floor {x}" for x in f.index], x=f[i], name=f"Cluster {i}", orientation="h",
                    marker_color=SERIES[i - 1], hovertemplate="%{y} · Cluster " + str(i) + ": %{x:,.1f} MWh<extra></extra>")
    fig.update_layout(barmode="stack", xaxis_title="MWh")
    fig.update_yaxes(autorange="reversed")
    show(style(fig, height=320))

with card("kWh per cluster, zone and end use", None, "Download for the report tables."):
    piv = kg.groupby(["zone_id", "use", "cluster"])["kwh"].sum().unstack("cluster").fillna(0)
    piv.columns = [f"Cluster {c}" for c in piv.columns]
    piv["Total"] = piv.sum(axis=1)
    piv = piv.reset_index().rename(columns={"zone_id": "Zone", "use": "End use"})
    st.dataframe(piv, hide_index=True, width="stretch", height=300,
                 column_config={c: st.column_config.NumberColumn(f"{c} (kWh)", format="%.0f") for c in piv.columns[2:]})
    st.download_button("Download (CSV)", piv.to_csv(index=False), f"qa_kwh_{group}.csv", "text/csv",
                       icon=":material/download:")

# ---------------------------------------------------------------- calendar & weather labels (after clustering)
c1, c2 = st.columns(2)
with c1, card("Calendar labels (attached after clustering)", "Calendar variables are not features."):
    m = pd.crosstab(lab["cluster"], lab["month"]).reindex(columns=range(1, 13), fill_value=0)
    m.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    st.dataframe(m.reset_index().rename(columns={"cluster": "Cluster"}), hide_index=True, width="stretch")
    st.dataframe(pd.crosstab(lab["cluster"], lab["day_type"]).reset_index().rename(columns={"cluster": "Cluster"}),
                 hide_index=True, width="stretch")
with c2, card("Weather labels (C1 only)", "Day-level Open-Meteo weather averaged per cluster: context, not cause."):
    if group == "C1" and "wx_temp_mean" in lab:
        w = lab.groupby("cluster")[["wx_temp_mean", "wx_temp_max", "wx_dew_point_mean", "wx_precip_mm"]].mean()
        st.dataframe(w.reset_index(), hide_index=True, width="stretch", column_config={
            "cluster": "Cluster", "wx_temp_mean": st.column_config.NumberColumn("Mean °C", format="%.1f"),
            "wx_temp_max": st.column_config.NumberColumn("Max °C", format="%.1f"),
            "wx_dew_point_mean": st.column_config.NumberColumn("Dew point °C", format="%.1f"),
            "wx_precip_mm": st.column_config.NumberColumn("Rain mm/day", format="%.1f")})
    else:
        st.caption("Weather is applied to AC-metered (C1) zones only.")

# ---------------------------------------------------------------- k selection & screening
with st.expander("How k was chosen (rules applied in order)", icon=":material/rule:"):
    st.dataframe(ks[["k", "min_share", "min_zones", "rule1_size", "half_year_ari", "rule2_stable", "silhouette",
                     "chosen"]], hide_index=True, width="stretch", column_config={
        "min_share": st.column_config.NumberColumn("Smallest cluster", format="percent"),
        "min_zones": st.column_config.NumberColumn("Fewest zones"),
        "rule1_size": st.column_config.CheckboxColumn("Rule 1 (≥3%, ≥3 zones)"),
        "half_year_ari": st.column_config.NumberColumn("Half-year ARI", format="%.2f"),
        "rule2_stable": st.column_config.CheckboxColumn("Rule 2 (ARI ≥0.8)"),
        "silhouette": st.column_config.NumberColumn("Silhouette", format="%.2f"),
        "chosen": st.column_config.CheckboxColumn("Chosen")})
    st.caption("Rule 4 (interpretable centroids) is a human check: read the profiles above.")
with st.expander("Scaling screen at k = 4 (replicates the plan's pilot)", icon=":material/science:"):
    st.dataframe(screen[screen["group"] == group].drop(columns="group"), hide_index=True, width="stretch",
                 column_config={"silhouette": st.column_config.NumberColumn(format="%.2f"),
                                "half_year_ari": st.column_config.NumberColumn(format="%.2f")})
    fl = scalers[(scalers["group"] == group) & scalers["floored"]]
    if len(fl):
        st.caption("Channels lifted to the 0.1 kWh floor: " + ", ".join(f"{r.zone_id} {r.use} (P95 {r.p95_kwh:.2f})"
                                                                      for r in fl.itertuples()))
with st.expander("Atypical days (farther than the 99th percentile from their centroid; kept, not removed)"):
    a = lab[lab["atypical"]].sort_values("distance", ascending=False)
    st.dataframe(a[["zone_id", "date", "cluster", "day_type", "distance"]], hide_index=True, width="stretch",
                 column_config={"date": st.column_config.DateColumn("Date"),
                                "distance": st.column_config.NumberColumn(format="%.2f")})

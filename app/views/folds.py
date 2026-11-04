import plotly.express as px
import streamlit as st
from common import SEQ_BLUE, card, heat_text, load_zone_day, page_header, show, style, zone_key

from cubems import config
from cubems.folds import fold_coverage, season_months

page_header("Within-season month folds",
            "Hold out one month, train on the other months of the same season, so a test month is never "
            "judged against a different climate regime.", eyebrow="Question B · validation design")

zd = load_zone_day()
f1, f2 = st.columns([1, 3])
group = f1.segmented_control("Model", ["Model 1", "Model 2"], default="Model 1") or "Model 1"
inc18 = f2.toggle("Include 2018 Jul–Dec months", value=False,
                  help="Adds rainy and cool folds. Changes the methodology (2018 was a historical check only), "
                       "so agree as a team first.")

sm = season_months(inc18)
cols = st.columns(len(sm))
for col, (season, months) in zip(cols, sm.items(), strict=True):
    with col.container(border=True):
        st.badge(season, color={"Hot": "orange", "Rainy": "blue", "Cool": "green"}[season])
        st.markdown(f"**{len(months)} folds** · {', '.join(months)}")

fc = fold_coverage(zd, group, inc18)
weak = fc[fc["test_usable_pct"] < 85]
with card("Fold coverage",
          f"<b>{len(weak)} of {len(fc)}</b> test months have under 85% usable zone-days: "
          + (", ".join(weak["test_month"]) if len(weak) else "none") + ".",
          "Seasons use whole months, approximating Thai Meteorological Department dates."):
    st.dataframe(fc, hide_index=True, width="stretch", column_config={
        "season": st.column_config.TextColumn("Season", width="small"),
        "test_month": st.column_config.TextColumn("Test month", width="small"),
        "train_months": st.column_config.TextColumn("Train months"),
        "test_usable_days": st.column_config.NumberColumn("Test zone-days"),
        "train_usable_days": st.column_config.NumberColumn("Train zone-days"),
        "test_usable_pct": st.column_config.ProgressColumn("Test usable", format="%.0f%%", min_value=0, max_value=100),
    })

d = zd[zd["group"] == group].copy()
if not inc18:
    d = d[d["year"] == 2019]
d["ym"] = d["date"].dt.strftime("%Y-%m")
piv = 100 - d.pivot_table(index="zone_id", columns="ym", values="usable", aggfunc="mean") * 100
piv = piv.loc[sorted(piv.index, key=zone_key)]
piv.columns = [f"{c}<br>{config.SEASON_BY_MONTH[int(c[-2:])]}" for c in piv.columns]
with card("Days lost per zone and month: which folds can you trust?",
          "Darker cells lose more days. Rainy (Jun–Oct) is the cleanest season; Cool is weakest "
          "(Jan–Feb damaged on Floors 2, 3, 5 and 7; no Floor 6 data in Nov–Dec)."):
    fig = px.imshow(piv, color_continuous_scale=SEQ_BLUE, zmin=0, zmax=100, aspect="auto",
                    labels=dict(color="% unusable", x="", y=""))
    fig.update_traces(xgap=2, ygap=2, text=heat_text(piv), texttemplate="%{text}",
                      hovertemplate="%{y} · %{x}<br>%{z:.0f}% of days unusable<extra></extra>")
    show(style(fig, height=max(360, 20 * len(piv)), legend=False))

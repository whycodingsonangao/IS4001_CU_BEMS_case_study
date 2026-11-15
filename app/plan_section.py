"""'Plan AC use' section of the Next-day priorities page: historical replay and what-if planning."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from common import load_zone_day, zone_key

from cubems import forecast as F
from cubems import planning

MODEL_LABELS = {"benchmark": "Same zone, seven days earlier", "boosting": "Gradient boosting", "ridge": "Ridge"}
REPLAY, WHATIF = "Historical replay", "What-if planning"
ONE, ALL = "One zone", "All AC zones"

_CSS = """
<style>
.plan-tag {display:inline-block; font-size:0.72rem; font-weight:650; letter-spacing:0.07em; text-transform:uppercase;
           border:1px solid currentColor; border-radius:999px; padding:2px 10px; opacity:0.75;}
.plan-big {font-size:4.2rem; font-weight:700; letter-spacing:-0.03em; line-height:1.05; margin:0.5rem 0 0.1rem 0;}
.plan-big small {font-size:1.5rem; font-weight:600; opacity:0.6; margin-left:0.35rem;}
.plan-sub {font-size:0.95rem; opacity:0.72; margin-bottom:1.1rem;}
.plan-card h6 {font-size:0.74rem; font-weight:650; letter-spacing:0.07em; text-transform:uppercase; opacity:0.62;
               margin:0 0 0.35rem 0; padding:0;}
.plan-card .v {font-size:1.25rem; font-weight:650; line-height:1.25;}
.plan-card .n {font-size:0.86rem; opacity:0.75; margin-top:0.2rem;}
</style>
"""


def _big(value: str, unit: str, sub: str, tag: str) -> None:
    st.markdown(f'<span class="plan-tag">{tag}</span><div class="plan-big">{value}<small>{unit}</small></div>'
                f'<div class="plan-sub">{sub}</div>', unsafe_allow_html=True)


def _cards(cards: list[tuple[str, str, str]]) -> None:
    for col, (heading, value, note) in zip(st.columns(len(cards)), cards, strict=True):
        with col.container(border=True):
            st.markdown(f'<div class="plan-card"><h6>{heading}</h6><div class="v">{value}</div>'
                        f'<div class="n">{note}</div></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner="Preparing model inputs...")
def _design() -> pd.DataFrame:
    return F.design(load_zone_day())


@st.cache_resource(show_spinner="Fitting the forecast model on earlier days...")
def _fitted(train_end: str, chosen: str) -> planning.Fitted:
    # keyed by the training cut-off, so every target in the same month reuses one fit
    # (train_end_for is idempotent on a cut-off date, so passing the cut-off as the target is safe)
    return planning.fit(_design(), pd.Timestamp(train_end), chosen)


# ---------------------------------------------------------------- A. historical replay
def _replay(pred: pd.DataFrame, chosen: str) -> None:
    scored = pred[pred[chosen].notna()]
    dates = sorted(scored["date"].unique())
    default = pd.Timestamp("2019-10-27")
    pick = st.date_input("Day to replay (scored 2019 days)", default if default in dates else dates[-1],
                         min_value=dates[0], max_value=dates[-1], key="plan_replay_date")
    day = scored[scored["date"] == pd.Timestamp(pick)].copy()
    if day.empty:
        st.info("No zone has a complete scored day on this date. Pick another day.")
        return
    # the advance ranking uses the forecast only; the actual is joined afterwards for display
    day = day.sort_values([chosen, "zone_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    day["rank"] = range(1, len(day) + 1)
    day["vs_typical"] = day[chosen] - day["typical_kwh"]
    day["actual_minus_forecast"] = day["y"] - day[chosen]
    f_total, a_total = float(day[chosen].sum()), float(day["y"].sum())
    top = day.iloc[0]

    _big(f"{f_total:,.0f}", "kWh", f"Forecast AC total for {pd.Timestamp(pick):%a %d %b %Y} · {len(day)} scored AC zones · "
         "issued at 00:00, before any reading from that day", "Historical replay · forecast")
    diff = a_total - f_total
    _cards([
        ("Actual AC · known after the day", f"{a_total:,.0f} kWh",
         "Measured once the day had ended. Not an input to the forecast or the ranking."),
        ("Actual minus forecast", f"{diff:+,.0f} kWh",
         f"{100 * diff / f_total:+.0f}% of the forecast total." if f_total else "Forecast total is zero."),
        ("Top forecast zone", f"{top.zone_id} · {top[chosen]:,.0f} kWh",
         f"Ranked first on the forecast alone. Its actual was {top.y:,.0f} kWh."),
    ])
    st.dataframe(
        day[["rank", "zone_id", chosen, "typical_kwh", "vs_typical", "benchmark", "y", "actual_minus_forecast"]],
        hide_index=True, width="stretch", height=min(420, 38 + 35 * len(day)), column_config={
            "rank": st.column_config.NumberColumn("Forecast rank", help="Order by forecast kWh only"),
            "zone_id": "Zone",
            chosen: st.column_config.NumberColumn("Forecast kWh", format="%.1f"),
            "typical_kwh": st.column_config.NumberColumn("Typical kWh", format="%.1f",
                                                         help="Median of earlier training days of the same day type"),
            "vs_typical": st.column_config.NumberColumn("Forecast vs typical", format="%+.1f"),
            "benchmark": st.column_config.NumberColumn("Last-week kWh", format="%.1f",
                                                       help="Same zone seven days earlier (the benchmark)"),
            "y": st.column_config.NumberColumn("Actual kWh (after the day)", format="%.1f"),
            "actual_minus_forecast": st.column_config.NumberColumn("Actual − forecast", format="%+.1f")})
    st.caption(f"Stage: {top.stage} fold. The forecast and its ranking were fixed before the day; the actual column is the "
               "historical meter reading shown for comparison only. Zones without a complete AC day are not scored.")


# ---------------------------------------------------------------- B. what-if planning
def _lag_key(col: str) -> str:
    return f"plan_lag_{col}"


def _wx_key(col: str) -> str:
    return f"plan_wx_{col}"


def _prefill(source: pd.Timestamp, zones: list[str], scope: str) -> None:
    """Button callback: copy the inputs known at 00:00 on the source day. Its own AC total is never copied."""
    ss = st.session_state
    lags, weather = planning.historical_inputs(_design(), source, zones)
    for col in F.PRIOR_WEATHER:
        ss[_wx_key(col)] = weather.get(col)
    if scope == ONE:
        for col in F.LAGS:
            ss[_lag_key(col)] = lags[zones[0]][col]
    else:
        ss.plan_all = pd.DataFrame([{"zone_id": z, **lags[z]} for z in zones])
        ss.plan_all_nonce = ss.get("plan_all_nonce", 0) + 1
    ss.plan_source = {"date": source, "zones": zones, "scope": scope, "lags": lags, "weather": weather}


def _same(a, b) -> bool:
    if a is None or pd.isna(a):
        return b is None or pd.isna(b)
    return b is not None and not pd.isna(b) and abs(float(a) - float(b)) < 1e-6


def _source_text(zone_inputs: dict, weather: dict, scope: str) -> tuple[str, str]:
    src = st.session_state.get("plan_source")
    if not src or src["scope"] != scope or not set(zone_inputs) <= set(src["zones"]):
        return "Manual entry", "Values typed on this page."
    edited = (any(not _same(weather.get(c), src["weather"].get(c)) for c in F.PRIOR_WEATHER)
              or any(not _same(zone_inputs[z].get(c), src["lags"][z].get(c)) for z in zone_inputs for c in F.LAGS))
    who = src["zones"][0] if len(src["zones"]) == 1 else f"{len(zone_inputs)} AC zones"
    head = f"{who} · {src['date']:%a %d %b %Y}"
    note = "Inputs as known at 00:00 that day; its own AC total is not used."
    return (head + " · edited", "Prefilled from that day, then changed by hand.") if edited else (head, note)


def _whatif(dec: dict) -> None:
    ss = st.session_state
    chosen = dec["chosen"]
    design = _design()
    mae = float(dec.get("final_mae", float("nan")))

    c = st.columns([1.2, 1, 1.4])
    target = pd.Timestamp(c[0].date_input("Target day", pd.Timestamp("2019-11-04"), min_value=planning.FIRST_TARGET,
                                          key="plan_target", help="Any day from May 2019 onward, including days after "
                                          "the dataset ends (illustrative only)."))
    scope = c[1].segmented_control("Scope", [ONE, ALL], default=ONE, key="plan_scope") or ONE
    fitted = _fitted(f"{planning.train_end_for(target):%Y-%m-%d}", chosen)
    zones = sorted(fitted.zones, key=zone_key)
    zone = c[2].selectbox("Zone", zones, index=zones.index("F2Z2") if "F2Z2" in zones else 0, key="plan_zone",
                          disabled=scope == ALL)
    illustrative = planning.is_illustrative(target)
    holiday = None
    if target.year != 2019:
        holiday = st.checkbox("Target day is a public holiday", key="plan_holiday",
                              help="The project's holiday list covers 2019 only, so state it here for other years.")
    cal = planning.calendar_fields(target, holiday)
    st.caption(f"Calendar fields from the target day: {cal['day_name']}, {cal['day_type'].lower()}"
               + (", listed public holiday" if cal["is_holiday"] else "") + ". "
               f"Model: {MODEL_LABELS.get(chosen, chosen)} (from decision.json), fitted on {fitted.train_rows:,} complete "
               f"zone-days up to {fitted.train_last:%d %b %Y}.")
    if illustrative:
        st.warning("Illustrative what-if forecast from a model trained on 2019 data. It is not a validated live forecast "
                   "for this date, and there is no live weather or meter feed: every input below is yours to supply.",
                   icon=":material/science:")

    # ---- inputs
    active = [zone] if scope == ONE else zones
    suggested = planning.similar_day(design, target, cal["day_type"], zone if scope == ONE else None)
    with st.container(border=True):
        st.markdown("###### Inputs the model uses")
        p = st.columns([1.2, 1.5, 2], vertical_alignment="bottom")
        last_source = min(target - pd.Timedelta(days=1), planning.DATA_END)
        source = pd.Timestamp(p[0].date_input(
            "Historical source day", suggested or last_source, min_value=pd.Timestamp("2019-01-08"),
            max_value=last_source, key=f"plan_src_{target:%Y%m%d}_{scope}_{zone if scope == ONE else 'all'}",
            help="Must be earlier than the target day. Defaults to the most recent earlier day with the same weekday "
                 "and day type."))
        p[1].button("Use a similar historical day", icon=":material/history:", type="primary", width="stretch",
                    on_click=_prefill, args=(source, active, scope))
        p[2].caption("Fills the fields with what was known at 00:00 on the source day: earlier AC totals and the "
                     "previous day's weather. The source day's own AC use is not copied. Edit anything afterwards.")

        if scope == ONE:
            st.markdown(f"**Earlier AC use for {zone}** · kWh per complete day · leave blank if unknown")
            cols = st.columns(3)
            lags = {col: cols[i].number_input(planning.LAG_LABELS[col], min_value=0.0, value=None, step=1.0,
                                              format="%.1f", key=_lag_key(col), placeholder="unknown")
                    for i, col in enumerate(F.LAGS)}
            zone_inputs = {zone: lags}
        else:
            st.markdown("**Earlier AC use per zone** · kWh per complete day · blank cells are treated as unknown")
            base = ss.get("plan_all")
            if base is None or list(base["zone_id"]) != zones:
                base = pd.DataFrame({"zone_id": zones, **{col: [None] * len(zones) for col in F.LAGS}})
                ss.plan_all = base
            edited = st.data_editor(
                base, hide_index=True, width="stretch", height=280, key=f"plan_all_editor_{ss.get('plan_all_nonce', 0)}",
                disabled=["zone_id"], column_config={"zone_id": "Zone", **{
                    col: st.column_config.NumberColumn(planning.LAG_LABELS[col] + " · kWh", min_value=0.0, format="%.1f")
                    for col in F.LAGS}})
            zone_inputs = {r["zone_id"]: {col: r[col] for col in F.LAGS} for r in edited.to_dict("records")}
            # a zone with no earlier AC value at all has no input, so it is left out of the sum
            skipped = [z for z, lags in zone_inputs.items() if all(v is None or pd.isna(v) for v in lags.values())]
            zone_inputs = {z: lags for z, lags in zone_inputs.items() if z not in skipped}

        st.markdown("**Previous day's weather** · required")
        cols = st.columns(5)
        weather = {col: cols[i].number_input(f"{label} · {unit}", value=None, step=0.1, format="%.1f", key=_wx_key(col),
                                             placeholder="required")
                   for i, (col, (label, unit)) in enumerate(planning.WEATHER_LABELS.items())}

    # ---- result
    if not zone_inputs:
        st.info("Enter earlier AC use for at least one zone, or use a similar historical day, to get a summed forecast.",
                icon=":material/edit:")
        return
    errors, warnings = planning.validate(zone_inputs, weather, fitted)
    tag = "What-if scenario · illustrative, not validated" if illustrative else "What-if scenario · forecast"
    if errors:
        st.error("**No forecast yet.** " + " ".join(errors), icon=":material/block:")
        return
    rows = planning.build_rows(zone_inputs, weather, target, holiday)
    rows["forecast"] = planning.predict(fitted, rows)
    rows["typical"] = [planning.typical_kwh(fitted, z, cal["day_type"]) for z in rows["zone_id"]]
    src_head, src_note = _source_text(zone_inputs, weather, scope)
    err_line = f"Typical error on the 2019 test was about ±{mae:.0f} kWh per zone-day." if mae == mae else ""

    if scope == ONE:
        f, t = float(rows["forecast"].iloc[0]), rows["typical"].iloc[0]
        _big(f"{f:,.0f}", "kWh", f"Forecast AC use for {zone} on {target:%a %d %b %Y} · one zone, AC only", tag)
        if t is None or pd.isna(t):
            typ = ("No typical value", "This zone has no earlier days of this day type.")
            use = "Use the forecast on its own to size tomorrow's check."
        else:
            d = f - t
            typ = (f"{d:+,.0f} kWh", f"Typical {cal['day_type'].lower()} for {zone}: {t:,.0f} kWh "
                                     "(median of earlier training days).")
            if mae == mae and d > mae:
                use = "Higher than typical. Confirm the schedule and bookings, and check this zone early in the round."
            elif mae == mae and d < -mae:
                use = "Lower than typical. No extra attention unless the space is meant to be in use."
            else:
                use = "Close to typical. Routine checks are enough."
        _cards([("Compared with typical", *typ), ("Input source", src_head, src_note),
                ("Planning use", use, f"{err_line} An energy estimate for planning, not an equipment-fault detector.")])
    else:
        total, typical_total = float(rows["forecast"].sum()), float(pd.to_numeric(rows["typical"]).sum())
        ranked = rows.sort_values(["forecast", "zone_id"], ascending=[False, True]).reset_index(drop=True)
        top3 = ", ".join(ranked["zone_id"].head(3))
        _big(f"{total:,.0f}", "kWh", f"Summed AC forecast for {target:%a %d %b %Y} · {len(rows)} of {len(zones)} AC zones "
             "with inputs · AC only, not whole-building electricity", tag)
        if skipped:
            st.caption(f"Not in the sum (no earlier AC value entered): {', '.join(skipped)}.")
        _cards([("Compared with typical", f"{total - typical_total:+,.0f} kWh",
                 f"Sum of each zone's typical {cal['day_type'].lower()}: {typical_total:,.0f} kWh."),
                ("Input source", src_head, src_note),
                ("Planning use", f"Start with {top3}",
                 f"Largest forecast loads. {err_line} For planning review effort, not for detecting faults.")])
        ranked["vs_typical"] = ranked["forecast"] - pd.to_numeric(ranked["typical"])
        st.dataframe(ranked[["zone_id", "forecast", "typical", "vs_typical"]].head(8), hide_index=True, width="stretch",
                     column_config={"zone_id": "Top zones by forecast",
                                    "forecast": st.column_config.NumberColumn("What-if forecast kWh", format="%.1f"),
                                    "typical": st.column_config.NumberColumn("Typical kWh", format="%.1f"),
                                    "vs_typical": st.column_config.NumberColumn("Forecast vs typical", format="%+.1f")})
    if len(warnings) <= 3:
        for w in warnings:
            st.warning(w, icon=":material/warning:")
    else:
        with st.expander(f"{len(warnings)} input warnings (missing or unusual values)", icon=":material/warning:"):
            for w in warnings:
                st.markdown(f"- {w}")


def render(dec: dict, pred: pd.DataFrame) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("#### Plan AC use")
        mode = st.segmented_control("Mode", [REPLAY, WHATIF], default=REPLAY, key="plan_mode",
                                    label_visibility="collapsed") or REPLAY
        if mode == REPLAY:
            st.caption("Replay what the forecast said at 00:00 on a past day, next to the reading that came in afterwards.")
            _replay(pred, dec["chosen"])
        else:
            st.caption("Enter tomorrow's inputs, or start from a similar past day, to get a planning estimate. "
                       "Scenario values are yours; nothing here is a meter reading.")
            _whatif(dec)

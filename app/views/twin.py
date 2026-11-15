"""Operations Twin (proof of concept): from a retrospective daily AC alert to an assigned inspection work order."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    FLAG_COLOR,
    MUTED,
    load_qb,
    load_qc,
    load_zone_day,
    load_zones,
    need,
    page_header,
    show,
    style,
)
from twin_component import BUILDING, twin_view
from twin_shared import STICKY_CSS, order_timeline, stamp_store_version, watch_store

from cubems import config, twin
from cubems import workorders as wo

DEMO_DATE = pd.Timestamp("2019-10-27")
STATE_ICON = {"ok": ":material/check_circle:", "caution": ":material/error:", "missing": ":material/do_not_disturb_on:"}
STATE_WORD = {"ok": "Available", "caution": "Check", "missing": "Not available"}

page_header("Operations Twin",
            "Pick a past day, see which AC zones fell outside their expected daily range and where they sit in the "
            "building, check the evidence, then raise and assign an inspection work order. Historical data only: "
            "this is a review after the day ends, with no live monitoring.",
            eyebrow="Proof of concept · alert to assigned inspection")

scored = need(load_qb("scored"), "run_question_b.py")
exceptions = need(load_qb("exceptions"), "run_question_b.py")
zone_day = load_zone_day()
zones = load_zones()
qc_events = load_qc("events")
holidays = pd.read_csv(config.HOLIDAYS_PATH) if config.HOLIDAYS_PATH.exists() else None

dates = twin.available_dates(scored)
counts = twin.findings_per_date(exceptions)
ss = st.session_state
if ss.get("twin_date") not in dates:
    ss.twin_date = DEMO_DATE if DEMO_DATE in dates else dates[0]
ss.setdefault("twin_zone", None)


# ---------------------------------------------------------------- state callbacks
def _set_date(d) -> None:
    ss.twin_date, ss.twin_zone, ss.twin_date_widget = d, None, d


def _date_changed() -> None:
    ss.twin_date, ss.twin_zone = ss.twin_date_widget, None


def _step(delta: int) -> None:
    i = dates.index(ss.twin_date) + delta
    if 0 <= i < len(dates):
        _set_date(dates[i])


def _select(zid: str | None) -> None:
    ss.twin_zone = zid


def _picked() -> None:
    state = ss.get("ops_twin")
    value = state.get("pick") if isinstance(state, dict) else getattr(state, "pick", None)
    if value:
        ss.twin_zone = None if value == BUILDING else value


def _open_order() -> None:
    rows = ss.get("twin_wo_table", {}).get("selection", {}).get("rows", [])
    orders = wo.work_orders()
    if rows and rows[0] < len(orders):
        o = orders[::-1][rows[0]]
        d = pd.Timestamp(o["alert_date"])
        if d in dates:
            _set_date(d)
            ss.twin_zone = o["zone_id"]


def _date_label(d: pd.Timestamp) -> str:
    n = counts.get(d, 0)
    return f"{d:%a %d %b %Y}  ·  {n} finding{'s' if n != 1 else ''}" if n else f"{d:%a %d %b %Y}  ·  no findings"


# ---------------------------------------------------------------- day selection
c = st.columns([2.2, 0.9, 0.8, 2.6], vertical_alignment="bottom")
c[0].selectbox("Day to review (only days with scored data)", dates, index=dates.index(ss.twin_date),
               format_func=_date_label, key="twin_date_widget", on_change=_date_changed)
c[1].button("Previous", icon=":material/chevron_left:", on_click=_step, args=(-1,), width="stretch",
            disabled=ss.twin_date == dates[0])
c[2].button("Next", icon=":material/chevron_right:", on_click=_step, args=(1,), width="stretch",
            disabled=ss.twin_date == dates[-1])
c[3].caption(f"Scored days run {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y} ({len(dates)} days, final test period). "
             "Each finding is known only after its day is complete.")

date = ss.twin_date
found = twin.day_findings(exceptions, date)
status = twin.day_status(scored, exceptions, date)
if ss.twin_zone not in status:
    ss.twin_zone = None
sel = ss.twin_zone

orders_all = wo.work_orders()
stamp_store_version()
order_by_finding = {o["finding_id"]: o for o in orders_all}  # latest order per finding

n_nodata = twin.day_summary(status)["no_data"]
n_within = twin.day_summary(status)["within"]
summary = (f"{len(found)} finding{'s' if len(found) != 1 else ''} · {n_within} within range · {n_nodata} without data"
           + (" · click a highlighted zone" if len(found) else ""))

WORKFLOW, HISTORICAL = "Workflow status", "Historical alert only"
has_orders_today = any(twin.finding_id(date, z) in order_by_finding for z in status)

left, right = st.columns([1.5, 1], gap="medium")
# keep the building in view while the (longer) side panel scrolls
st.markdown(STICKY_CSS, unsafe_allow_html=True)
left = left.container(key="twin_stage")

with left:
    overlay = WORKFLOW
    if has_orders_today:
        overlay = st.segmented_control(
            "Building overlay", [WORKFLOW, HISTORICAL], default=WORKFLOW, key="twin_overlay",
            help="Workflow status adds the demo work-order state on top of the historical alert. "
                 "Historical alert only shows the original red and orange deviation colours.") or WORKFLOW
    payload_zones = twin.component_zones(status, date, order_by_finding, show_workflow=overlay == WORKFLOW)
    watch_store()
    twin_view(key="ops_twin", on_pick_change=_picked,
              data={"zones": payload_zones, "selected": sel, "date_label": f"{date:%a %d %b %Y}", "summary": summary})
    if n_nodata:
        f6 = [z for z, s in status.items() if s["status"] == twin.STATUS_NO_DATA and s["floor"] == 6]
        note = f"{n_nodata} AC zone{'s have' if n_nodata != 1 else ' has'} no complete scored day on this date (hatched)."
        if len(f6) == 4 and date > pd.Timestamp(config.FLOOR_COMPARE_END):
            note += f" Floor 6 has no data after {pd.Timestamp(config.FLOOR_COMPARE_END):%d %b %Y}."
        st.caption(note + " No data is not the same as within range.")


# ---------------------------------------------------------------- side panel
def _finding_panel(f: dict) -> None:
    fid = f["finding_id"]
    near_zero = bool(f["near_zero_range"])
    items = twin.evidence(zone_day, scored, exceptions, zones, date, f["zone_id"], qc_events, holidays)

    # 1 finding
    with st.container(border=True):
        st.markdown(f"###### Finding · review priority {f['priority']} of {len(found)}")
        st.markdown(f"#### Floor {f['floor']} · Zone {f['zone']}  \n"
                    f"<span class='subtle'>{f['zone_id']} · {date:%A %d %B %Y} · {f['direction']}</span>",
                    unsafe_allow_html=True)
        m = st.columns(2)
        m[0].metric("Actual AC · kWh", f"{f['y']:,.1f}")
        m[1].metric("Predicted AC · kWh", f"{f['pred']:,.1f}")
        st.metric("Expected range · kWh", f"{f['lo']:,.1f} to {f['hi']:,.1f}")
        st.metric("Beyond the range · kWh", f"{f['departure_kwh']:+,.1f}", f["direction"], delta_color="off",
                  delta_arrow="up" if f["departure_kwh"] > 0 else "down")
        st.caption("Review priority ranks the day's findings by kWh beyond the expected range. "
                   "It is not a fault severity and does not identify a cause.")
        st.page_link("views/exceptions.py", label=f"Open {f['zone_id']} in Daily AC deviations",
                     icon=":material/crisis_alert:", query_params={"zone": f["zone_id"]})

    # 2 evidence
    with st.container(border=True):
        st.markdown("###### Check before assigning work")
        if near_zero:
            st.warning(next(i["text"] for i in items if i["key"] == "near_zero"), icon=":material/error:")
        hist = twin.recent_history(zone_day, f["zone_id"], date)
        if len(hist) > 1:
            fig = go.Figure()
            is_day = hist["date"] == date
            fig.add_bar(x=hist["date"], y=hist["ac_kwh"],
                        marker_color=[FLAG_COLOR if d else MUTED for d in is_day],
                        marker_opacity=[1.0 if d else (0.75 if t == "Workday" else 0.35)
                                        for d, t in zip(is_day, hist["day_type"], strict=True)],
                        customdata=hist["day_type"], hovertemplate="%{x|%a %d %b}: %{y:.1f} kWh<br>%{customdata}<extra></extra>")
            fig.add_hrect(y0=f["lo"], y1=f["hi"], fillcolor=MUTED, opacity=0.14, line_width=0)
            fig.update_layout(yaxis_title="AC kWh", bargap=0.25)
            show(style(fig, height=150, legend=False))
            st.caption(f"Daily AC for {f['zone_id']}, previous {twin.HISTORY_DAYS} days. Red: this day. Dark bars: "
                       "workdays; light bars: weekends and holidays. Band: this day's expected range.")
        for i in items:
            if i["key"] == "near_zero":
                continue
            st.markdown(f"{STATE_ICON[i['state']]} **{i['label']}** · {STATE_WORD[i['state']]}  \n{i['text']}")
        st.caption("These checks describe the data. An energy deviation alone does not show an equipment fault "
                   "or its cause; that needs an on-site inspection.")

    # 3 disposition
    existing = wo.orders_for(fid)
    recorded = wo.disposition_for(fid)
    with st.container(border=True):
        st.markdown("###### Manager disposition")
        if recorded:
            st.markdown(f":material/task_alt: Recorded: **{recorded['disposition']}** · "
                        f"{pd.Timestamp(recorded['at']):%d %b %Y %H:%M}"
                        + (f"  \n“{recorded['note']}”" if recorded.get("note") else ""))
        default = recorded["disposition"] if recorded else None
        choice = st.radio("Decision for this finding", wo.DISPOSITIONS, key=f"disp_{fid}",
                          index=wo.DISPOSITIONS.index(default) if default else None,
                          captions=["Cause known or acceptable. No record of work.", "Watch the next days. No work record.",
                                    "Draft a work order for an on-site inspection."])
        if choice in wo.DISPOSITIONS[:2]:
            note = st.text_input("Note (optional)", key=f"note_{fid}",
                                 placeholder="e.g. weekend event booked in this zone")
            if st.button("Record decision", key=f"rec_{fid}", icon=":material/save:"):
                wo.record_disposition(fid, choice, note)
                st.rerun()
            st.caption("This records the decision only. No work order is created.")
        elif choice == wo.DISPOSITIONS[2]:
            another = False
            if existing:
                st.info(f"{existing[-1]['id']} already covers this finding ({existing[-1]['status']}).",
                        icon=":material/assignment_turned_in:")
                another = st.checkbox("Create another work order for this finding anyway", key=f"dup_{fid}")
            if not existing or another:
                with st.form(f"draft_{fid}", border=False):
                    st.markdown("**Work-order draft** (prefilled from the finding)")
                    st.markdown(
                        f"- **Alert date (historical):** {date:%a %d %b %Y}\n"
                        f"- **Location:** Floor {f['floor']}, Zone {f['zone']} ({f['zone_id']}), Chamchuri 5\n"
                        f"- **Measured deviation:** {f['y']:,.1f} kWh actual against an expected "
                        f"{f['lo']:,.1f} to {f['hi']:,.1f} kWh ({f['departure_kwh']:+,.1f} kWh, {f['direction'].lower()})")
                    summary_text = st.text_area("Evidence summary", twin.evidence_summary(items), height=170,
                                                key=f"evi_{fid}_{len(existing)}")
                    request = st.text_area("Inspection request (edit before creating)", twin.default_request(f),
                                           height=150, key=f"req_{fid}_{len(existing)}")
                    if st.form_submit_button("Create work order (demo)", type="primary", icon=":material/assignment_add:"):
                        try:
                            wo.create_work_order(f, summary_text, request, allow_duplicate=another)
                        except wo.DuplicateWorkOrder as dup:
                            st.warning(f"{dup.existing['id']} already covers this finding.")
                        else:
                            ss.pop(f"dup_{fid}", None)
                            st.rerun()

    # 4 work-order status
    with st.container(border=True):
        st.markdown("###### Demo work-order status")
        if not existing:
            st.markdown("<span class='subtle'>No work order for this finding.</span>", unsafe_allow_html=True)
        for o in existing:
            st.markdown(f"**{o['id']}** · {o['status']}")
            st.markdown(order_timeline(o))
            if o["status"] == wo.STATUS_COMPLETED:
                st.success(f"{wo.COMPLETED_LABEL}. This is a workflow status. The historical reading of "
                           f"{o['actual_kwh']:,.1f} kWh and the deviation are unchanged.", icon=":material/task_alt:")
                if o.get("technician_note"):
                    st.markdown(f"Technician note (demo): “{o['technician_note']}”")
            if o["status"] == wo.STATUS_CREATED:
                a = st.columns([2, 1.3], vertical_alignment="bottom")
                team = a[0].selectbox("Assign to (demo queue)", wo.DEMO_TEAMS, key=f"team_{o['id']}")
                if a[1].button("Assign", key=f"assign_{o['id']}", type="primary", icon=":material/engineering:",
                               width="stretch"):
                    try:
                        wo.assign(o["id"], team)
                    except wo.InvalidTransition as e:
                        st.warning(str(e))
                    else:
                        st.rerun()
            with st.expander("Request text"):
                st.write(o["request"])
        st.caption("Demo records in a local store shared with the technician demo app. Nobody is notified and no "
                   "maintenance system is contacted. Completion is recorded there as a demo status.")


def _other_zone_panel(s: dict) -> None:
    with st.container(border=True):
        st.markdown(f"#### Floor {s['floor']} · Zone {s['zone']}  \n"
                    f"<span class='subtle'>{s['id']} · {date:%A %d %B %Y}</span>", unsafe_allow_html=True)
        if s["status"] == twin.STATUS_WITHIN:
            m = st.columns(2)
            m[0].metric("Actual AC · kWh", f"{s['y']:,.1f}")
            m[1].metric("Predicted AC · kWh", f"{s['pred']:,.1f}")
            st.metric("Expected range · kWh", f"{s['lo']:,.1f} to {s['hi']:,.1f}")
            st.success("Within the expected range. No finding and no decision needed.", icon=":material/check:")
            st.page_link("views/exceptions.py", label=f"Open {s['id']} in Daily AC deviations",
                         icon=":material/crisis_alert:", query_params={"zone": s["id"]})
        else:
            msg = "No complete scored AC day for this zone on this date, so it cannot be checked against a range."
            if s["floor"] == 6 and date > pd.Timestamp(config.FLOOR_COMPARE_END):
                msg += f" Floor 6 has no data after {pd.Timestamp(config.FLOOR_COMPARE_END):%d %b %Y}."
            st.info(msg + " This is missing data, not a sign that the zone was within range.",
                    icon=":material/do_not_disturb_on:")


def _overview_panel() -> None:
    with st.container(border=True):
        st.markdown(f"###### Review priority · {date:%a %d %b %Y}")
        if found.empty:
            st.success("No monitored zone fell outside its expected range on this date.", icon=":material/check:")
            st.caption(f"{n_within} zones within range, {n_nodata} AC zones without data. "
                       "Pick another day to walk through the alert-to-assignment flow.")
            return
        st.caption("Ranked by kWh beyond the expected range, largest first. Select a zone here or on the building.")
        for f in found.to_dict("records"):
            o = order_by_finding.get(f["finding_id"])
            tag = f"  ·  {o['id']} {o['status'].lower()}" if o else ""
            flag = "  ·  near-zero range" if f["near_zero_range"] else ""
            st.button(f"{f['priority']}  ·  Floor {f['floor']} Zone {f['zone']}  ·  {f['departure_kwh']:+,.1f} kWh{flag}{tag}",
                      key=f"pick_{f['finding_id']}", on_click=_select, args=(f["zone_id"],), width="stretch",
                      type="primary" if f["priority"] == 1 else "secondary")


with right:
    if sel is None:
        _overview_panel()
    else:
        st.button("Whole building", icon=":material/arrow_back:", on_click=_select, args=(None,), key="twin_back")
        row = found[found["zone_id"] == sel]
        if len(row):
            rec = row.iloc[0].to_dict()
            rec["date"] = f"{date:%Y-%m-%d}"
            _finding_panel(rec)
        else:
            _other_zone_panel(status[sel])

# ---------------------------------------------------------------- work-order list
with left:
    with st.container(border=True):
        st.markdown(f"###### Demo work orders ({len(orders_all)})")
        if not orders_all:
            st.markdown("<span class='subtle'>None yet. Select a finding and choose “Create inspection work order”.</span>",
                        unsafe_allow_html=True)
        else:
            table = pd.DataFrame([{"Work order": o["id"], "Alert date": pd.Timestamp(o["alert_date"]),
                                   "Zone": o["zone_id"], "Beyond range · kWh": o["departure_kwh"],
                                   "Status": o["status"], "Assigned to": o["team"] or "Unassigned"}
                                  for o in orders_all[::-1]])
            st.dataframe(table, hide_index=True, width="stretch", key="twin_wo_table", on_select=_open_order,
                         selection_mode="single-row", height=min(248, 38 + 35 * len(table)),
                         column_config={"Alert date": st.column_config.DateColumn(format="DD MMM YYYY"),
                                        "Beyond range · kWh": st.column_config.NumberColumn(format="%+.1f")})
            st.caption("Select a row to open that finding. Stored in `demo_state/operations_twin.sqlite3` "
                       "(git-ignored), shared with the technician app and separate from the analysis outputs.")
            with st.popover("Reset demo records", icon=":material/delete:"):
                st.write("Delete every demo disposition and work order on this machine?")
                if st.button("Delete demo records", type="primary"):
                    wo.reset()
                    st.rerun()

with st.expander("What is real and what is simulated"):
    st.markdown(
        "- **Real (historical):** the 2019 meter readings, the forecast, the expected range and the list of "
        "out-of-range zone-days. They come unchanged from the Question B outputs.\n"
        "- **Schematic:** the building drawing. Zone shapes are approximate, traced by eye from the published floor plan.\n"
        "- **Simulated:** the manager's decision, the work order, the HVAC teams, the assignment and the technician's "
        "completion. They exist only in a local demo store.\n"
        "- **Not shown:** any site visit, repair, AC adjustment, energy saving or change to a meter reading. "
        "Marking an inspection complete changes the workflow colour only.")

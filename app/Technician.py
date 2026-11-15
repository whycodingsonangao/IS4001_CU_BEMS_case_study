"""Technician demo dashboard (separate local app):

    streamlit run app/Technician.py --server.port 8502

Shows only work orders the manager has assigned in the Operations Twin, on the same building twin.
Demo only: no login, no notifications, no maintenance system. State is the shared local store in
src/cubems/workorders.py; nothing here changes a historical reading or any analysis output.
"""

import pandas as pd
import streamlit as st
from common import apply_style, load_qb, need
from twin_component import BUILDING, twin_view
from twin_shared import STICKY_CSS, order_timeline, stamp_store_version, watch_store

from cubems import twin
from cubems import workorders as wo

st.set_page_config(page_title="Technician queue · CU-BEMS demo", page_icon=":material/engineering:", layout="wide",
                   initial_sidebar_state="collapsed")
apply_style()
st.markdown("""<style>
.tech-empty {text-align:center; padding:4.5rem 1rem 4rem 1rem;}
.tech-empty .t {font-size:1.5rem; font-weight:650; letter-spacing:-0.01em; margin-top:0.6rem;}
.tech-empty .s {opacity:0.65; margin-top:0.5rem; font-size:0.95rem;}
.tech-empty .i {font-size:2.4rem; opacity:0.35;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Technician demo · assigned inspections</div>', unsafe_allow_html=True)
st.title("Inspection queue")
st.markdown('<p class="subtle">Work orders assigned by the facility manager in the Operations Twin. Demo workflow on '
            "historical 2019 data: nothing here is a live alert, and marking a job complete records a demo status only.</p>",
            unsafe_allow_html=True)

scored = need(load_qb("scored"), "run_question_b.py")
exceptions = need(load_qb("exceptions"), "run_question_b.py")
ss = st.session_state
ss.setdefault("tech_order", None)
ss.setdefault("tech_nonce", 0)

# ---------------------------------------------------------------- queue (assigned and completed only)
ALL_TEAMS = "All demo teams"
f = st.columns([1.4, 1.2, 3], vertical_alignment="bottom")
team = f[0].selectbox("Queue for", [ALL_TEAMS, *wo.DEMO_TEAMS], key="tech_team")
show_done = f[1].toggle("Show completed", value=True, key="tech_show_done")
queue = wo.technician_queue(None if team == ALL_TEAMS else team)
stamp_store_version()
watch_store()
n_open = sum(o["status"] == wo.STATUS_ASSIGNED for o in queue)
n_done = len(queue) - n_open
if not show_done:
    queue = [o for o in queue if o["status"] == wo.STATUS_ASSIGNED]
queue = sorted(queue, key=lambda o: (o["status"] != wo.STATUS_ASSIGNED, o["assigned_at"] or ""))
ids = [o["id"] for o in queue]
if ss.tech_order not in ids:
    ss.tech_order = None
f[2].caption(f"{n_open} open · {n_done} completed (demo). The list refreshes on its own every few seconds. "
             "Created but unassigned drafts stay with the manager.")


def _table_key() -> str:
    return f"tech_table_{ss.tech_nonce}"


def _row_selected() -> None:
    rows = ss.get(_table_key(), {}).get("selection", {}).get("rows", [])
    ss.tech_order = ids[rows[0]] if rows and rows[0] < len(ids) else None


def _select(order_id: str | None) -> None:
    """Programmatic selection (from the twin): remount the table so its highlighted row follows."""
    ss.tech_order = order_id
    ss.tech_nonce += 1


def _picked() -> None:
    state = ss.get("tech_twin")
    value = state.get("pick") if isinstance(state, dict) else getattr(state, "pick", None)
    if not value:
        return
    if value == BUILDING:
        _select(None)
        return
    current = next((o for o in queue if o["id"] == ss.tech_order), None)
    if current:  # another zone on the same alert date that also has an assigned work order
        match = next((o for o in queue if o["zone_id"] == value and o["alert_date"] == current["alert_date"]), None)
        if match:
            _select(match["id"])


if not queue:
    with st.container(border=True):
        st.markdown('<div class="tech-empty"><div class="i">◻</div><div class="t">No inspection work orders assigned.</div>'
                    '<div class="s">Work orders appear here after the facility manager creates one in the Operations '
                    "Twin and assigns it to a demo HVAC team.</div></div>", unsafe_allow_html=True)
    st.stop()

table = pd.DataFrame([{
    "Work order": o["id"], "Status": o["status"], "Location": f"Floor {o['floor']} · Zone {o['zone']}",
    "Alert date (historical)": pd.Timestamp(o["alert_date"]), "Beyond range · kWh": o["departure_kwh"],
    "Assigned to": o["team"], "Assigned": pd.Timestamp(o["assigned_at"])} for o in queue])
with st.container(border=True):
    st.markdown(f"###### Assigned work orders ({len(queue)})")
    default = {"selection": {"rows": [ids.index(ss.tech_order)]}} if ss.tech_order else None
    st.dataframe(table, hide_index=True, width="stretch", key=_table_key(), on_select=_row_selected,
                 selection_mode="single-row", selection_default=default, height=min(248, 38 + 35 * len(table)),
                 column_config={"Alert date (historical)": st.column_config.DateColumn(format="DD MMM YYYY"),
                                "Beyond range · kWh": st.column_config.NumberColumn(format="%+.1f"),
                                "Assigned": st.column_config.DatetimeColumn(format="DD MMM, HH:mm")})
    st.caption("Select a row to open its location in the building and the inspection details.")

order = next((o for o in queue if o["id"] == ss.tech_order), None)
if order is None:
    with st.container(border=True):
        st.markdown('<div class="tech-empty" style="padding:2.5rem 1rem;"><div class="s">Select a work order above to see '
                    "where it is in the building.</div></div>", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------- shared twin + details
date = pd.Timestamp(order["alert_date"])
status = twin.day_status(scored, exceptions, date)
by_finding = {o["finding_id"]: o for o in wo.technician_queue()}  # the technician never sees unassigned drafts
zones_payload = twin.component_zones(status, date, by_finding)
counts = twin.day_summary(status)

left, right = st.columns([1.5, 1], gap="medium")
st.markdown(STICKY_CSS, unsafe_allow_html=True)
with left.container(key="twin_stage"):
    twin_view(key="tech_twin", on_pick_change=_picked,
              data={"zones": zones_payload, "selected": order["zone_id"], "date_label": f"{date:%a %d %b %Y}",
                    "summary": f"Historical alert date · {counts['findings']} finding"
                               f"{'s' if counts['findings'] != 1 else ''} that day"})
    st.caption(f"Same building twin as the manager's Operations Twin, set to this work order's historical alert date "
               f"({date:%d %b %Y}). Red and orange are that day's deviations; green marks a demo work order already "
               "marked complete.")

with right:
    done = order["status"] == wo.STATUS_COMPLETED
    with st.container(border=True):
        st.markdown(f"###### Work order · {order['status']}")
        st.markdown(f"#### {order['id']}  \n<span class='subtle'>Floor {order['floor']} · Zone {order['zone']} "
                    f"({order['zone_id']}), Chamchuri 5 · {order['team']}</span>", unsafe_allow_html=True)
        st.markdown(order_timeline(order))

    with st.container(border=True):
        st.markdown("###### Original finding · historical reading")
        st.caption(f"{date:%A %d %B %Y} · {order['direction']} · review priority {order['review_priority']} that day")
        m = st.columns(2)
        m[0].metric("Actual AC · kWh", f"{order['actual_kwh']:,.1f}")
        m[1].metric("Predicted AC · kWh", f"{order['predicted_kwh']:,.1f}")
        st.metric("Expected range · kWh", f"{order['range_low_kwh']:,.1f} to {order['range_high_kwh']:,.1f}")
        st.metric("Beyond the range · kWh", f"{order['departure_kwh']:+,.1f}")
        st.caption("Recorded 2019 meter data and forecast, copied when the work order was created. "
                   "An energy deviation does not say what caused it.")

    with st.container(border=True):
        st.markdown("###### Inspection request")
        st.write(order["request"])
        with st.expander("Evidence the manager checked"):
            st.text(order["evidence_summary"])

    with st.container(border=True):
        st.markdown("###### Demo checklist")
        if done:
            for item, ticked in order.get("checklist", {}).items():
                st.markdown(f"{':material/check_box:' if ticked else ':material/check_box_outline_blank:'} {item}")
            if order.get("technician_note"):
                st.markdown(f"**Note:** {order['technician_note']}")
            st.success(f"{wo.COMPLETED_LABEL} · {order['completed_by_role']} · "
                       f"{pd.Timestamp(order['completed_at']):%d %b %Y %H:%M}", icon=":material/task_alt:")
        else:
            ticks = {item: st.checkbox(item, key=f"chk_{order['id']}_{i}") for i, item in enumerate(wo.CHECKLIST)}
            note = st.text_area("Technician note", key=f"note_{order['id']}", height=100,
                                placeholder="What was checked and found (demo text)")
            ready = all(ticks.values())
            if st.button("Mark inspection complete (demo)", type="primary", icon=":material/task_alt:",
                         disabled=not ready, key=f"done_{order['id']}"):
                try:
                    wo.complete(order["id"], ticks, note)
                except wo.InvalidTransition as e:
                    st.warning(str(e))
                else:
                    st.rerun()
            if not ready:
                st.caption("Tick every step to enable completion.")
        st.caption(f"Completion is recorded as “{wo.TECHNICIAN_ROLE}” with a timestamp. It is a demo status: it does not "
                   "show that anyone visited the building, and it changes no meter reading or energy figure.")

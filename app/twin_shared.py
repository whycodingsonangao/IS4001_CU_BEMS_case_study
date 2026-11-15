"""Helpers shared by the manager's Operations Twin page and the technician app."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from cubems import workorders as wo

STICKY_CSS = ("<style>[data-testid='stLayoutWrapper']:has(> .st-key-twin_stage) "
              "{position: sticky; top: 3.75rem;}</style>")
POLL_SECONDS = 4


def stamp_store_version() -> None:
    """Call once per full run, after reading the store, so the poll only fires on later changes."""
    st.session_state["_store_version"] = wo.version()


@st.fragment(run_every=POLL_SECONDS)
def watch_store() -> None:
    """Short status poll: rerun the page when the other local app has written to the shared store."""
    if wo.version() != st.session_state.get("_store_version"):
        st.rerun()


def order_timeline(o: dict) -> str:
    """Markdown status steps for one demo work order."""
    done, todo = ":material/check_circle:", ":material/radio_button_unchecked:"
    fmt = lambda t: f"{pd.Timestamp(t):%d %b %Y %H:%M}"  # noqa: E731
    lines = [f"{done} Created · {fmt(o['created_at'])}"]
    if o.get("assigned_at"):
        lines.append(f"{done} Assigned for inspection · {o['team']} · {fmt(o['assigned_at'])}")
    else:
        lines.append(f"{todo} Assigned for inspection · not yet")
    if o.get("completed_at"):
        lines.append(f"{done} {wo.COMPLETED_LABEL} · {o.get('completed_by_role', wo.TECHNICIAN_ROLE)} · "
                     f"{fmt(o['completed_at'])}")
    else:
        lines.append(f"{todo} {wo.COMPLETED_LABEL} · not yet")
    return "  \n".join(lines)

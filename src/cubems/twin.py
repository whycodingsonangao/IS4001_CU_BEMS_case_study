"""Operations Twin: turn one day's retrospective AC deviations into located, evidence-backed findings.

Pure functions over the existing Question B outputs (scored / exceptions) and the zone-day table.
Nothing here changes the model, the expected range or any historical output. The ranking is
abs(departure_kwh) and is a review priority, not a fault severity.
"""

from __future__ import annotations

import pandas as pd

from . import config

# ---------------------------------------------------------------- display constants [D]
# An expected upper bound at or below this is treated as "the model expected the AC to be off".
# Display rule for the twin only; it does not feed back into the Question B range.
NEAR_ZERO_RANGE_KWH = 5.0
RECENT_SAME_TYPE_DAYS = 4  # recent same-day-type days used as plain history context
HISTORY_DAYS = 21  # days of AC history drawn in the side panel

STATUS_TOP = "top"  # review priority 1 on the selected date
STATUS_FINDING = "finding"  # other out-of-range zones on the selected date
STATUS_WITHIN = "within"  # scored, inside the expected range
STATUS_NO_DATA = "no_data"  # AC zone without a scored complete day on this date
STATUS_NO_AC = "no_ac"  # zone has no AC meter, so Question B never monitors it

# ---------------------------------------------------------------- schematic floor plan
# Approximate zone footprints traced from the published floor plan (Pipattanasomporn et al. 2020,
# Sci. Data 7:241, Fig. 2). Plan units: x runs west to east (0-100), y runs north to south (0-64).
# Schematic only: not surveyed dimensions, and a zone is a metering area, not necessarily one room.
PLAN_W, PLAN_D = 100.0, 64.0
_CORES = [
    [(0, 10), (4, 10), (4, 54), (0, 54)],
    [(96, 9), (100, 9), (100, 54), (96, 54)],
]
PLAN_LOW = {  # Floors 1-2: four zones
    1: [(55, 0), (96, 0), (96, 64), (58, 64), (58, 34), (47, 34), (47, 17), (55, 9)],
    2: [(4, 0), (55, 0), (55, 9), (47, 17), (47, 34), (27, 34), (27, 29), (4, 29)],
    3: [(27, 34), (58, 34), (58, 64), (27, 64)],
    4: [(4, 29), (27, 29), (27, 64), (4, 64)],
}
PLAN_HIGH = {  # Floors 3-7: five zones
    1: [(39, 0), (96, 0), (96, 27), (55, 27), (55, 34), (49, 34), (49, 20), (39, 20)],
    2: [(55, 27), (96, 27), (96, 64), (55, 64)],
    3: [(24, 34), (55, 34), (55, 64), (24, 64)],
    4: [(4, 0), (39, 0), (39, 20), (49, 20), (49, 34), (24, 34), (24, 27), (4, 27)],
    5: [(4, 27), (24, 27), (24, 64), (4, 64)],
}
# Where each zone's label sits (the area centroid falls near a neighbour for the L-shaped zones).
LABEL_LOW = {1: (77, 30), 2: (25, 13), 3: (42, 50), 4: (15, 47)}
LABEL_HIGH = {1: (70, 12), 2: (76, 47), 3: (39, 50), 4: (20, 12), 5: (14, 47)}
FLOORS = list(range(1, 8))


def zone_id(floor: int, zone: int) -> str:
    return f"F{floor}Z{zone}"


def valid_zones() -> dict[int, list[int]]:
    """Zone numbers per floor from the project configuration (so F2Z5 never appears)."""
    out: dict[int, set[int]] = {f: set() for f in FLOORS}
    for group in (config.C1_ZONES, config.C2_ZONES, config.LIGHT_ONLY_ZONES):
        for floor, zones in group.items():
            out[floor].update(zones)
    return {f: sorted(z) for f, z in out.items()}


def ac_zone_ids() -> set[str]:
    return {zone_id(f, z) for f, zones in config.C1_ZONES.items() for z in zones}


def plan_for(floor: int) -> dict[int, list[tuple[float, float]]]:
    return PLAN_LOW if floor <= 2 else PLAN_HIGH


def layout() -> list[dict]:
    """Every valid zone with its schematic footprint, plus the unmetered service cores."""
    valid = valid_zones()
    ac = ac_zone_ids()
    out = []
    for floor in FLOORS:
        plan = plan_for(floor)
        for zone in valid[floor]:
            zid = zone_id(floor, zone)
            out.append({"id": zid, "floor": floor, "zone": zone, "poly": plan[zone], "has_ac": zid in ac,
                        "label": (LABEL_LOW if floor <= 2 else LABEL_HIGH)[zone]})
        for i, core in enumerate(_CORES):
            out.append({"id": f"F{floor}core{i}", "floor": floor, "zone": 0, "poly": core, "has_ac": False,
                        "core": True})
    return out


# ---------------------------------------------------------------- findings for a day
def finding_id(date, zid: str) -> str:
    return f"{pd.Timestamp(date):%Y-%m-%d}:{zid}"


def available_dates(scored: pd.DataFrame) -> list[pd.Timestamp]:
    """Only dates that actually have scored zone-days."""
    return sorted(pd.Timestamp(d) for d in scored["date"].unique())


def findings_per_date(exceptions: pd.DataFrame) -> dict[pd.Timestamp, int]:
    return {pd.Timestamp(d): int(n) for d, n in exceptions.groupby("date").size().items()}


def day_findings(exceptions: pd.DataFrame, date) -> pd.DataFrame:
    """Flagged zones on one date, ranked by abs(departure_kwh) descending.

    Ties break on floor, then zone number, so the order is deterministic. `priority` starts at 1.
    """
    d = exceptions[exceptions["date"] == pd.Timestamp(date)].copy()
    d["abs_departure_kwh"] = d["departure_kwh"].abs()
    d = d.sort_values(["abs_departure_kwh", "floor", "zone"], ascending=[False, True, True], kind="mergesort")
    d["priority"] = range(1, len(d) + 1)
    d["near_zero_range"] = d["hi"] <= NEAR_ZERO_RANGE_KWH
    d["finding_id"] = [finding_id(date, z) for z in d["zone_id"]]
    return d.reset_index(drop=True)


def day_status(scored: pd.DataFrame, exceptions: pd.DataFrame, date) -> dict[str, dict]:
    """Status of every valid zone on one date, keyed by zone id."""
    date = pd.Timestamp(date)
    found = day_findings(exceptions, date).set_index("zone_id")
    day = scored[scored["date"] == date].set_index("zone_id")
    ac = ac_zone_ids()
    out: dict[str, dict] = {}
    for floor, zones in valid_zones().items():
        for zone in zones:
            zid = zone_id(floor, zone)
            rec: dict = {"id": zid, "floor": floor, "zone": zone}
            if zid not in ac:
                rec["status"] = STATUS_NO_AC
            elif zid not in day.index:
                rec["status"] = STATUS_NO_DATA
            else:
                row = day.loc[zid]
                rec.update(y=float(row["y"]), pred=float(row["pred"]), lo=float(row["lo"]), hi=float(row["hi"]),
                           departure_kwh=float(row["departure_kwh"]), direction=str(row["direction"]))
                if zid in found.index:
                    rank = int(found.loc[zid, "priority"])
                    rec.update(status=STATUS_TOP if rank == 1 else STATUS_FINDING, priority=rank,
                               near_zero_range=bool(found.loc[zid, "near_zero_range"]))
                else:
                    rec["status"] = STATUS_WITHIN
            out[zid] = rec
    return out


def component_zones(status: dict[str, dict], date, order_by_finding: dict[str, dict] | None = None,
                    show_workflow: bool = True) -> list[dict]:
    """Zone records for the building component, shared by the manager and technician apps.

    `status` carries the historical alert state (never changed by the workflow). `wo` is the demo
    work-order status layered on top; pass show_workflow=False to see the historical alert alone.
    """
    orders = order_by_finding or {}
    out = []
    for z in layout():
        rec = {"id": z["id"], "floor": z["floor"], "zone": z["zone"], "poly": z["poly"], "core": bool(z.get("core")),
               "label": z.get("label")}
        if rec["core"]:
            rec["status"] = "core"
        else:
            s = status[z["id"]]
            rec.update(status=s["status"], priority=s.get("priority", 0), departure_kwh=s.get("departure_kwh", 0.0))
            o = orders.get(finding_id(date, z["id"])) if show_workflow else None
            rec["wo"] = o["status"] if o else None
        out.append(rec)
    return out


def day_summary(status: dict[str, dict]) -> dict[str, int]:
    count = lambda name: sum(1 for s in status.values() if s["status"] == name)  # noqa: E731
    return {"findings": count(STATUS_TOP) + count(STATUS_FINDING), "within": count(STATUS_WITHIN),
            "no_data": count(STATUS_NO_DATA)}


# ---------------------------------------------------------------- evidence
def _fmt(v: float, unit: str = "kWh") -> str:
    return f"{v:,.1f} {unit}"


def _days(n: int, day_type: str) -> str:
    """'4 workdays' / '4 weekend or holiday days'."""
    if day_type == "Workday":
        return f"{n} workday{'s' if n != 1 else ''}"
    return f"{n} weekend or holiday day{'s' if n != 1 else ''}"


def recent_history(zone_day: pd.DataFrame, zid: str, date, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """The zone's daily AC kWh for the `days` before the date and the date itself (complete AC days only)."""
    date = pd.Timestamp(date)
    q = zone_day[(zone_day["zone_id"] == zid) & (zone_day["date"] <= date)
                 & (zone_day["date"] >= date - pd.Timedelta(days=days))]
    q = q[q["ac_hours"] == 24]
    return q[["date", "ac_kwh", "day_type"]].sort_values("date").reset_index(drop=True)


def evidence(zone_day: pd.DataFrame, scored: pd.DataFrame, exceptions: pd.DataFrame, zones: pd.DataFrame,
             date, zid: str, qc_events: pd.DataFrame | None = None, holidays: pd.DataFrame | None = None) -> list[dict]:
    """Checks a manager can make from the data before assigning work.

    Each item is {key, label, state, text}; state is 'ok' (evidence available), 'caution' (read this
    before acting) or 'missing' (the evidence does not exist, said explicitly). Statements are
    descriptive only: nothing here names a cause.
    """
    date = pd.Timestamp(date)
    items: list[dict] = []
    row = scored[(scored["zone_id"] == zid) & (scored["date"] == date)]
    zd = zone_day[(zone_day["zone_id"] == zid) & (zone_day["date"] == date)]
    info = zones[zones["zone_id"] == zid]
    s = row.iloc[0] if len(row) else None
    z = zd.iloc[0] if len(zd) else None

    # 1 near-zero expected range
    if s is not None and s["hi"] <= NEAR_ZERO_RANGE_KWH and s["departure_kwh"] > 0:
        items.append({"key": "near_zero", "label": "Near-zero expected range", "state": "caution",
                      "text": f"The expected range is {s['lo']:.1f} to {s['hi']:.1f} kWh, so the forecast expected the "
                              "AC to be off. Any use at all is flagged and the amount beyond the range is simply "
                              "the day's AC use. Check whether the zone was scheduled to be occupied."})

    # 2 recent AC history
    hist = recent_history(zone_day, zid, date)
    prior = hist[hist["date"] < date]
    if s is not None and len(prior):
        same = prior[prior["day_type"] == s["day_type"]].tail(RECENT_SAME_TYPE_DAYS)
        other = prior[prior["day_type"] != s["day_type"]].tail(RECENT_SAME_TYPE_DAYS)
        parts = []
        if len(same):
            parts.append(f"Previous {_days(len(same), s['day_type'])}: median "
                         f"{_fmt(float(same['ac_kwh'].median()))}")
        if len(other):
            parts.append(f"previous {_days(len(other), other['day_type'].iloc[0])}: median "
                         f"{_fmt(float(other['ac_kwh'].median()))}")
        items.append({"key": "history", "label": "Recent AC history", "state": "ok",
                      "text": f"This day: {_fmt(float(s['y']))}. " + "; ".join(parts) + "."})
    else:
        items.append({"key": "history", "label": "Recent AC history", "state": "missing",
                      "text": f"No complete AC days for this zone in the {HISTORY_DAYS} days before this date."})

    # 3 zone sensor
    has_env = bool(info["has_env"].iloc[0]) if len(info) else False
    if not has_env:
        items.append({"key": "sensor", "label": "Zone temperature / humidity", "state": "missing",
                      "text": "This zone has no multi-sensor in the dataset, so there is no indoor reading to check."})
    elif z is None or pd.isna(z["temp_mean"]) or not z["temp_minutes"]:
        items.append({"key": "sensor", "label": "Zone temperature / humidity", "state": "missing",
                      "text": "The zone's multi-sensor has no valid readings on this date."})
    else:
        cover = 100 * float(z["temp_minutes"]) / config.MINUTES_PER_DAY
        rh = "" if pd.isna(z["rh_mean"]) else f", {z['rh_mean']:.0f}% RH"
        base = prior.merge(zone_day[zone_day["zone_id"] == zid][["date", "temp_mean"]], on="date")
        base = base[base["day_type"] == s["day_type"]].tail(RECENT_SAME_TYPE_DAYS) if s is not None else base.iloc[0:0]
        ref = (f" Previous {_days(len(base), s['day_type'])} averaged {base['temp_mean'].mean():.1f} °C."
               if len(base) and base["temp_mean"].notna().any() else "")
        items.append({"key": "sensor", "label": "Zone temperature / humidity", "state": "ok",
                      "text": f"Daily mean {z['temp_mean']:.1f} °C{rh} ({cover:.0f}% of minutes valid).{ref}"})

    # 4 weather and calendar
    if z is not None and not pd.isna(z.get("wx_temp_mean")):
        hol = ""
        if holidays is not None:
            h = holidays[pd.to_datetime(holidays["date"]) == date]
            if len(h):
                hol = f" Listed holiday: {h['name'].iloc[0]}."
        items.append({"key": "context", "label": "Weather and calendar", "state": "ok",
                      "text": f"{z['day_name']}, scored as {str(z['day_type']).lower()}.{hol} Outdoor mean "
                              f"{z['wx_temp_mean']:.1f} °C, max {z['wx_temp_max']:.1f} °C (ERA5 reanalysis, not an "
                              "on-site station)."})
    else:
        items.append({"key": "context", "label": "Weather and calendar", "state": "missing",
                      "text": "No weather record is joined for this date."})

    # 5 nearby zones and building-wide pattern
    day = scored[scored["date"] == date]
    flagged = exceptions[exceptions["date"] == date]
    floor = int(zid[1])
    peers = sorted(p for p in ac_zone_ids() if int(p[1]) == floor and p != zid)
    bits = []
    for p in peers:
        if p in set(flagged["zone_id"]):
            dep = float(flagged.loc[flagged["zone_id"] == p, "departure_kwh"].iloc[0])
            bits.append(f"{p} outside range ({dep:+.1f} kWh)")
        elif p in set(day["zone_id"]):
            bits.append(f"{p} within range")
        else:
            bits.append(f"{p} no data")
    n_flag, n_mon = len(flagged), len(day)
    wide = n_mon > 0 and n_flag >= max(3, round(0.25 * n_mon))
    text = (("Same floor: " + ", ".join(bits) + ". ") if bits else "No other AC-metered zone on this floor. ")
    text += (f"Building: {n_flag} of {n_mon} monitored zones {'is' if n_flag == 1 else 'are'} outside the expected "
             "range on this date.")
    if wide:
        text += (" Many zones moved together, so check the calendar and schedule before treating this as one "
                 "zone's issue.")
    items.append({"key": "nearby", "label": "Nearby zones", "state": "caution" if wide else "ok", "text": text})

    # 6 data quality
    if z is not None and s is not None:
        notes = [f"AC meter complete for {int(z['ac_hours'])}/24 hours",
                 f"expected range calibrated on {_days(int(s['n']), s['day_type'])} in the development period"]
        screened = int(z["spike_minutes"]) + int(z["flat_minutes"])
        n_events = 0
        if qc_events is not None and len(qc_events):
            e = qc_events[(qc_events["zone_id"] == zid) & (qc_events["start"] < date + pd.Timedelta(days=1))
                          & (qc_events["end"] >= date)]
            n_events = len(e)
        if screened or n_events:
            notes.append(f"{screened} minutes removed by spike/flatline screening ({n_events} episodes)")
        else:
            notes.append("no spike or flatline episodes screened out")
        units = int(info["n_ac_units"].iloc[0]) if len(info) else 0
        if units:
            notes.append(f"{units} AC circuit{'s' if units != 1 else ''} metered in this zone")
        items.append({"key": "quality", "label": "Data quality", "state": "caution" if screened or n_events else "ok",
                      "text": "; ".join(notes) + "."})
    else:
        items.append({"key": "quality", "label": "Data quality", "state": "missing",
                      "text": "No complete scored day for this zone on this date."})
    return items


def evidence_summary(items: list[dict]) -> str:
    """Plain-text evidence block used to prefill a work-order draft."""
    mark = {"ok": "", "caution": " [check]", "missing": " [not available]"}
    return "\n".join(f"- {i['label']}{mark[i['state']]}: {i['text']}" for i in items)


def default_request(f: dict) -> str:
    """Editable inspection request. Asks for an inspection; it does not assert a cause."""
    where = f"Floor {f['floor']}, Zone {f['zone']} ({f['zone_id']})"
    return (f"Please inspect the AC serving {where}. On {pd.Timestamp(f['date']):%a %d %b %Y} the zone's metered AC "
            f"energy was {f['y']:.1f} kWh against an expected range of {f['lo']:.1f} to {f['hi']:.1f} kWh "
            f"({abs(f['departure_kwh']):.1f} kWh {'above' if f['departure_kwh'] > 0 else 'below'} the range). "
            "The cause is not known from energy data alone. Check the operating schedule and setpoints for that day, "
            "whether the space was in use, and the condition of the units, then record what you find.")

import json
import multiprocessing as mp

import pandas as pd
import pytest

from cubems import twin
from cubems import workorders as wo


def _exc():
    return pd.DataFrame({"zone_id": ["F3Z2", "F2Z2", "F4Z1", "F3Z1"], "floor": [3, 2, 4, 3], "zone": [2, 2, 1, 1],
                         "date": pd.Timestamp("2019-09-14"), "departure_kwh": [50.0, -80.0, 50.0, 10.0],
                         "hi": [0.0, 170.0, 2.0, 90.0]})


def test_layout_uses_only_valid_zones():
    ids = {z["id"] for z in twin.layout() if not z.get("core")}
    assert len(ids) == 33 and "F2Z5" not in ids and "F3Z5" in ids
    assert len(twin.ac_zone_ids()) == 24


def test_review_priority_is_abs_departure_with_deterministic_ties():
    f = twin.day_findings(_exc(), "2019-09-14")
    assert f["zone_id"].tolist() == ["F2Z2", "F3Z2", "F4Z1", "F3Z1"]  # |-80| first; 50/50 tie -> lower floor
    assert f["priority"].tolist() == [1, 2, 3, 4]
    assert f["near_zero_range"].tolist() == [False, True, True, False]
    assert twin.day_findings(_exc(), "2019-09-15").empty


def test_no_data_is_distinct_from_within_range():
    scored = _exc().assign(y=1.0, pred=1.0, lo=0.0, direction="x")
    s = twin.day_status(scored, _exc(), "2019-09-14")
    assert s["F2Z2"]["status"] == twin.STATUS_TOP and s["F3Z2"]["status"] == twin.STATUS_FINDING
    assert s["F6Z1"]["status"] == twin.STATUS_NO_DATA and s["F1Z1"]["status"] == twin.STATUS_NO_AC


FINDING = {"finding_id": "2019-10-27:F2Z2", "date": "2019-10-27", "zone_id": "F2Z2", "floor": 2, "zone": 2,
           "y": 373.2, "pred": 135.7, "lo": 87.3, "hi": 170.0, "departure_kwh": 203.2, "direction": "Above expected",
           "priority": 1}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBEMS_DEMO_STORE", str(tmp_path / "operations_twin.sqlite3"))
    return tmp_path


def test_work_order_transitions(store):
    assert wo.technician_queue() == [] and wo.work_orders() == []  # fresh store: empty technician queue
    wo.record_disposition(FINDING["finding_id"], "Monitor")
    assert wo.work_orders() == []  # only "Create inspection work order" makes a work record
    o = wo.create_work_order(FINDING, "evidence", "inspect")
    assert o["status"] == wo.STATUS_CREATED
    assert wo.technician_queue() == []  # a created, unassigned draft never reaches the technician
    with pytest.raises(wo.InvalidTransition):
        wo.complete(o["id"], {}, "")
    with pytest.raises(wo.DuplicateWorkOrder):
        wo.create_work_order(FINDING, "evidence", "inspect")
    assert wo.create_work_order(FINDING, "e", "r", allow_duplicate=True)["id"] == "DEMO-WO-0002"
    v = wo.version()
    a = wo.assign(o["id"], wo.DEMO_TEAMS[0])
    assert a["status"] == wo.STATUS_ASSIGNED and wo.version() > v
    assert [q["id"] for q in wo.technician_queue()] == [o["id"]]
    assert wo.technician_queue(wo.DEMO_TEAMS[1]) == []
    with pytest.raises(wo.InvalidTransition):
        wo.assign(o["id"], wo.DEMO_TEAMS[1])
    done = wo.complete(o["id"], dict.fromkeys(wo.CHECKLIST, True), "  checked  ")
    assert done["status"] == wo.STATUS_COMPLETED and done["completed_by_role"] == wo.TECHNICIAN_ROLE
    assert done["technician_note"] == "checked" and done["completed_at"]
    assert done["actual_kwh"] == 373.2 and done["departure_kwh"] == 203.2  # the historical finding is untouched
    with pytest.raises(wo.InvalidTransition):
        wo.complete(o["id"], {}, "")
    assert [h["status"] for h in wo.get(o["id"])["history"]] == list(wo.STATUSES)
    wo.reset()
    assert wo.work_orders() == [] and wo.disposition_for(FINDING["finding_id"]) is None


def test_legacy_json_is_migrated_not_discarded(store):
    legacy = {"dispositions": {"2019-10-27:F2Z2": {"disposition": "Monitor", "note": "", "at": "2026-11-15T23:00:00"}},
              "work_orders": [{"id": "DEMO-WO-0003", "finding_id": "2019-10-27:F2Z2", "status": wo.STATUS_ASSIGNED,
                               "team": "HVAC Team A (demo)", "zone_id": "F2Z2", "alert_date": "2019-10-27",
                               "created_at": "2026-11-15T23:30:00", "assigned_at": "2026-11-15T23:31:00"}]}
    (store / wo.LEGACY_JSON).write_text(json.dumps(legacy))
    assert [o["id"] for o in wo.technician_queue()] == ["DEMO-WO-0003"]
    assert wo.disposition_for("2019-10-27:F2Z2")["disposition"] == "Monitor"
    assert not (store / wo.LEGACY_JSON).exists() and (store / "operations_twin.migrated.json").exists()
    other = dict(FINDING, finding_id="2019-09-14:F3Z2", zone_id="F3Z2")
    assert wo.create_work_order(other, "e", "r")["id"] == "DEMO-WO-0004"  # numbering continues after migration


def _create_in_other_process(path, finding, out):
    import os

    os.environ["CUBEMS_DEMO_STORE"] = path
    from cubems import workorders

    try:
        out.put(("created", workorders.create_work_order(finding, "e", "r")["id"]))
    except workorders.DuplicateWorkOrder as dup:
        out.put(("duplicate", dup.existing["id"]))


def test_concurrent_processes_cannot_duplicate_or_corrupt(store):
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    path = str(store / "operations_twin.sqlite3")
    same = [ctx.Process(target=_create_in_other_process, args=(path, FINDING, out)) for _ in range(6)]
    distinct = [ctx.Process(target=_create_in_other_process,
                            args=(path, dict(FINDING, finding_id=f"2019-09-{d:02d}:F3Z2"), out)) for d in range(1, 7)]
    for p in same + distinct:
        p.start()
    readers_ok = all(isinstance(wo.work_orders(), list) for _ in range(40))  # reads while other processes write
    for p in same + distinct:
        p.join(60)
    results = [out.get(timeout=10) for _ in same + distinct]
    assert readers_ok and sum(kind == "created" for kind, _ in results) == 7  # 1 for the shared finding + 6 distinct
    orders = wo.work_orders()
    assert len(orders) == 7 and len({o["id"] for o in orders}) == 7
    assert len(wo.orders_for(FINDING["finding_id"])) == 1

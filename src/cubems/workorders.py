"""Shared demo work-order store for the Operations Twin (manager app) and the technician app.

One local SQLite file (git-ignored) holds manager dispositions and demo inspection work orders.
Both Streamlit processes open it; every write runs in a `BEGIN IMMEDIATE` transaction, so status
changes and the duplicate check are atomic across processes. It is deliberately separate from
data/processed: nothing here is an analytical output, no one is notified, and no real maintenance
system is involved. An earlier JSON store is imported once and kept as a backup.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config

STATUS_CREATED = "Created"
STATUS_ASSIGNED = "Assigned for inspection"
STATUS_COMPLETED = "Completed (demo)"
STATUSES = (STATUS_CREATED, STATUS_ASSIGNED, STATUS_COMPLETED)
TECHNICIAN_STATUSES = (STATUS_ASSIGNED, STATUS_COMPLETED)  # a created-but-unassigned draft never reaches the queue
COMPLETED_LABEL = "Inspection marked complete (demo)"
DISPOSITIONS = ("Explained / no work", "Monitor", "Create inspection work order")
DEMO_TEAMS = ("HVAC Team A (demo)", "HVAC Team B (demo)", "HVAC contractor queue (demo)")
TECHNICIAN_ROLE = "Technician (demo role)"
CHECKLIST = (
    "Opened the work order and read the historical finding",
    "Checked the zone's AC schedule and setpoints (demo step)",
    "Checked the condition of the AC units serving the zone (demo step)",
    "Recorded what was found in the note below",
)
LEGACY_JSON = "operations_twin.json"


class DuplicateWorkOrder(Exception):
    """A work order already exists for this finding and a duplicate was not explicitly allowed."""

    def __init__(self, existing: dict):
        super().__init__(f"{existing['id']} already covers finding {existing['finding_id']}")
        self.existing = existing


class InvalidTransition(Exception):
    """The requested status change is not allowed from the work order's current status."""


def store_path() -> Path:
    return Path(os.environ.get("CUBEMS_DEMO_STORE", config.ROOT / "demo_state" / "operations_twin.sqlite3"))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    seq INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, finding_id TEXT NOT NULL, status TEXT NOT NULL,
    updated_at TEXT NOT NULL, data TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS work_orders_finding ON work_orders (finding_id);
CREATE TABLE IF NOT EXISTS dispositions (finding_id TEXT PRIMARY KEY, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


@contextmanager
def _db(write: bool = False):
    """A short-lived connection. Writers take the write lock up front (BEGIN IMMEDIATE) and wait up to 15 s."""
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=15, isolation_level=None)
    try:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=15000")
        con.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        if write or not _ready(con):
            if not write:  # first touch from a reader: upgrade to a write transaction to create the schema
                con.execute("COMMIT")
                con.execute("BEGIN IMMEDIATE")
            _init(con, path)
        yield con
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _ready(con: sqlite3.Connection) -> bool:
    row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='meta'").fetchone()
    return bool(row) and bool(con.execute("SELECT 1 FROM meta WHERE key='schema'").fetchone())


def _init(con: sqlite3.Connection, path: Path) -> None:
    if _ready(con):
        return
    for stmt in _SCHEMA.strip().split(";"):
        if stmt.strip():
            con.execute(stmt)
    con.execute("INSERT OR REPLACE INTO meta VALUES ('schema', '1')")
    con.execute("INSERT OR IGNORE INTO meta VALUES ('version', '0')")
    _migrate_json(con, path.parent / LEGACY_JSON)


def _migrate_json(con: sqlite3.Connection, legacy: Path) -> None:
    """Import the earlier JSON store once (inside the creating transaction), then keep it as a backup."""
    if not legacy.exists():
        return
    try:
        old = json.loads(legacy.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    for order in old.get("work_orders", []):
        order.setdefault("history", [])
        con.execute("INSERT OR IGNORE INTO work_orders (seq, id, finding_id, status, updated_at, data) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (int(order["id"].rsplit("-", 1)[1]), order["id"], order["finding_id"], order["status"],
                     order.get("assigned_at") or order.get("created_at") or _now(), json.dumps(order)))
    for fid, rec in old.get("dispositions", {}).items():
        con.execute("INSERT OR IGNORE INTO dispositions VALUES (?, ?)", (fid, json.dumps(rec)))
    con.execute("INSERT OR REPLACE INTO meta VALUES ('migrated_from', ?)", (legacy.name,))
    try:
        legacy.replace(legacy.with_name(legacy.stem + ".migrated.json"))
    except OSError:
        pass


def _bump(con: sqlite3.Connection) -> None:
    con.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='version'")


def _order(row) -> dict:
    return json.loads(row[0])


def _write(con: sqlite3.Connection, order: dict) -> None:
    con.execute("UPDATE work_orders SET status=?, updated_at=?, data=? WHERE id=?",
                (order["status"], _now(), json.dumps(order), order["id"]))
    _bump(con)


# ---------------------------------------------------------------- reads
def version() -> int:
    """Changes on every write by any process; apps poll this to notice the other app's updates."""
    with _db() as con:
        return int(con.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0])


def work_orders(statuses: tuple[str, ...] | None = None) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT data FROM work_orders ORDER BY seq").fetchall()
    orders = [_order(r) for r in rows]
    return orders if statuses is None else [o for o in orders if o["status"] in statuses]


def technician_queue(team: str | None = None) -> list[dict]:
    """Assigned (and already completed) work orders only. Unassigned drafts are never included."""
    orders = work_orders(TECHNICIAN_STATUSES)
    return orders if team is None else [o for o in orders if o["team"] == team]


def get(order_id: str) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT data FROM work_orders WHERE id=?", (order_id,)).fetchone()
    return _order(row) if row else None


def orders_for(finding_id: str) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT data FROM work_orders WHERE finding_id=? ORDER BY seq", (finding_id,)).fetchall()
    return [_order(r) for r in rows]


def disposition_for(finding_id: str) -> dict | None:
    with _db() as con:
        row = con.execute("SELECT data FROM dispositions WHERE finding_id=?", (finding_id,)).fetchone()
    return json.loads(row[0]) if row else None


# ---------------------------------------------------------------- writes
def record_disposition(finding_id: str, disposition: str, note: str = "") -> dict:
    """Record the manager's decision for a finding. This never creates a work record."""
    if disposition not in DISPOSITIONS:
        raise ValueError(f"Unknown disposition: {disposition}")
    rec = {"disposition": disposition, "note": note.strip(), "at": _now()}
    with _db(write=True) as con:
        con.execute("INSERT OR REPLACE INTO dispositions VALUES (?, ?)", (finding_id, json.dumps(rec)))
        _bump(con)
    return rec


def create_work_order(finding: dict, evidence_summary: str, request: str, allow_duplicate: bool = False) -> dict:
    """Create a demo inspection work order from a finding. Raises DuplicateWorkOrder unless allowed.

    The duplicate check and the insert share one write transaction, so two processes (or two clicks)
    cannot both create an order for the same finding.
    """
    with _db(write=True) as con:
        existing = con.execute("SELECT data FROM work_orders WHERE finding_id=? ORDER BY seq DESC LIMIT 1",
                               (finding["finding_id"],)).fetchone()
        if existing and not allow_duplicate:
            raise DuplicateWorkOrder(_order(existing))
        number = 1 + (con.execute("SELECT COALESCE(MAX(seq), 0) FROM work_orders").fetchone()[0])
        now = _now()
        order = {
            "id": f"DEMO-WO-{number:04d}",
            "finding_id": finding["finding_id"],
            "alert_date": finding["date"],
            "zone_id": finding["zone_id"],
            "floor": int(finding["floor"]),
            "zone": int(finding["zone"]),
            "actual_kwh": float(finding["y"]),
            "predicted_kwh": float(finding["pred"]),
            "range_low_kwh": float(finding["lo"]),
            "range_high_kwh": float(finding["hi"]),
            "departure_kwh": float(finding["departure_kwh"]),
            "direction": finding["direction"],
            "review_priority": int(finding["priority"]),
            "evidence_summary": evidence_summary,
            "request": request.strip(),
            "status": STATUS_CREATED,
            "team": None,
            "created_at": now,
            "assigned_at": None,
            "completed_at": None,
            "history": [{"status": STATUS_CREATED, "at": now}],
        }
        con.execute("INSERT INTO work_orders (seq, id, finding_id, status, updated_at, data) VALUES (?, ?, ?, ?, ?, ?)",
                    (number, order["id"], order["finding_id"], order["status"], now, json.dumps(order)))
        con.execute("INSERT OR REPLACE INTO dispositions VALUES (?, ?)",
                    (finding["finding_id"], json.dumps({"disposition": DISPOSITIONS[2], "note": "", "at": now})))
        _bump(con)
    return order


def _load_for_update(con: sqlite3.Connection, order_id: str) -> dict:
    row = con.execute("SELECT data FROM work_orders WHERE id=?", (order_id,)).fetchone()
    if not row:
        raise KeyError(order_id)
    return _order(row)


def assign(order_id: str, team: str) -> dict:
    """Assign a created demo work order to a demo team or queue. Sends nothing to anyone."""
    with _db(write=True) as con:
        order = _load_for_update(con, order_id)
        if order["status"] != STATUS_CREATED:
            raise InvalidTransition(f"{order_id} is '{order['status']}'; only a created work order can be assigned.")
        now = _now()
        order.update(status=STATUS_ASSIGNED, team=team, assigned_at=now)
        order["history"].append({"status": STATUS_ASSIGNED, "team": team, "at": now})
        _write(con, order)
    return order


def complete(order_id: str, checklist: dict[str, bool], note: str = "", role: str = TECHNICIAN_ROLE) -> dict:
    """Mark an assigned demo work order complete. Records a demo role and a timestamp, nothing else.

    This is a workflow status only: it does not record a site visit and changes no meter reading.
    """
    with _db(write=True) as con:
        order = _load_for_update(con, order_id)
        if order["status"] != STATUS_ASSIGNED:
            raise InvalidTransition(f"{order_id} is '{order['status']}'; only an assigned work order can be completed.")
        now = _now()
        order.update(status=STATUS_COMPLETED, completed_at=now, completed_by_role=role,
                     checklist={k: bool(v) for k, v in checklist.items()}, technician_note=note.strip())
        order["history"].append({"status": STATUS_COMPLETED, "role": role, "at": now})
        _write(con, order)
    return order


def reset() -> None:
    """Delete every demo record (rows, not the file, so the other app keeps a valid database)."""
    with _db(write=True) as con:
        con.execute("DELETE FROM work_orders")
        con.execute("DELETE FROM dispositions")
        _bump(con)

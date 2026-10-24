"""SQLite log of analyst actions (accept, correct, dismiss, feedback, approve_report, reject_report)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from src import config

CREATE_SQL = '''
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    report_id TEXT,
    claim_id TEXT,
    action TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    note TEXT,
    analyst TEXT
)
'''


def _path(db_path: Optional[str]) -> str:
    return db_path or config.AUDIT_DB


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = _path(db_path)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(CREATE_SQL)
    conn.commit()
    return conn


def log_action(
    report_id: str,
    claim_id: Optional[str],
    action: str,
    old_value,
    new_value,
    note: Optional[str],
    analyst: str,
    db_path: Optional[str] = None,
) -> int:
    """Append one analyst action and return its row id."""
    conn = _connect(db_path)
    try:
        cursor = conn.execute(
            '''
            INSERT INTO actions (
                timestamp, report_id, claim_id, action, old_value, new_value, note, analyst
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                datetime.now(timezone.utc).isoformat(),
                report_id,
                claim_id,
                action,
                None if old_value is None else str(old_value),
                None if new_value is None else str(new_value),
                note,
                analyst,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def get_log(report_id: Optional[str] = None, db_path: Optional[str] = None) -> list[dict]:
    """Return log rows, optionally filtered by report_id, oldest first."""
    conn = _connect(db_path)
    try:
        if report_id is None:
            rows = conn.execute('SELECT * FROM actions ORDER BY id ASC').fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM actions WHERE report_id = ? ORDER BY id ASC',
                (report_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

JSON_COLS = ("audio_meta", "transcript", "draft_note", "note", "risk_flags", "timings", "warnings", "signoff")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    language TEXT NOT NULL,
    filename TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    audio_meta TEXT,
    transcript TEXT,
    draft_note TEXT,
    note TEXT,
    risk_flags TEXT NOT NULL DEFAULT '[]',
    timings TEXT NOT NULL DEFAULT '{}',
    warnings TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    risk_alert INTEGER NOT NULL DEFAULT 0,
    signed_by TEXT,
    signed_at TEXT,
    signoff TEXT
);
CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    at TEXT NOT NULL,
    target TEXT NOT NULL,
    ref TEXT NOT NULL,
    before TEXT,
    after TEXT
);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ponytail: connection per call, last write wins on concurrent PATCHes; add row versions if that matters.
class Store:
    def __init__(self, path: Path):
        self.path = path
        with closing(self._connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def create(self, id: str, language: str, filename: str, audio_meta: dict) -> None:
        ts = now()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO sessions (id, status, language, filename, created_at, updated_at, audio_meta)"
                " VALUES (?, 'queued', ?, ?, ?, ?, ?)",
                (id, language, filename, ts, ts, json.dumps(audio_meta)),
            )

    def get(self, id: str) -> dict | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM sessions WHERE id = ?", (id,)).fetchone()
        return _decode(row) if row else None

    def list_all(self) -> list[dict]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY created_at DESC, rowid DESC").fetchall()
        return [_decode(r) for r in rows]

    def update(self, id: str, **fields) -> None:
        fields["updated_at"] = now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [json.dumps(v) if k in JSON_COLS else v for k, v in fields.items()]
        with closing(self._connect()) as db, db:
            db.execute(f"UPDATE sessions SET {cols} WHERE id = ?", (*values, id))

    def add_edit(self, session_id: str, target: str, ref: str, before, after) -> None:
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO edits (session_id, at, target, ref, before, after) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, now(), target, ref, json.dumps(before), json.dumps(after)),
            )

    def edit_count(self, session_id: str) -> int:
        with closing(self._connect()) as db:
            return db.execute("SELECT COUNT(*) FROM edits WHERE session_id = ?", (session_id,)).fetchone()[0]

    def fail_interrupted(self) -> int:
        with closing(self._connect()) as db, db:
            cur = db.execute(
                "UPDATE sessions SET status = 'failed', error = 'interrupted by restart', updated_at = ?"
                " WHERE status IN ('queued', 'processing')",
                (now(),),
            )
            return cur.rowcount


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in JSON_COLS:
        if d[k] is not None:
            d[k] = json.loads(d[k])
    d["risk_alert"] = bool(d["risk_alert"])
    return d

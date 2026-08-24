"""
db.py — SQLite database layer for the AI Candidate Screening Platform.

All iteration over candidates must use actual DB query results, never range(1, n+1).
s_no is the primary key and may not be contiguous in the real dataset.
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path

DB_PATH = Path("screening.db")

# C8: Column alias map lives here, applied at CSV ingestion boundary before validation.
COLUMN_ALIASES = {
    "github": "github_url",
    "resume": "resume_url",
    "s.no": "s_no",
    "sno": "s_no",
}

REQUIRED_COLUMNS = {
    "s_no", "name", "email", "college", "branch",
    "cgpa", "best_ai_project", "research_work",
    "github_url", "resume_url",
}

STATUS_FLOW = [
    "uploaded",
    "resume_parsed",
    "github_analyzed",
    "ai_scored",
    "ranked",
    "test_sent",
    "test_scored",
    "shortlisted",
    "interview_scheduled",
    "invited",
]

# ─── Schema ───────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    s_no            INTEGER PRIMARY KEY,
    name            TEXT,
    email           TEXT,
    college         TEXT,
    branch          TEXT,
    cgpa            REAL,
    best_ai_project TEXT,
    research_work   TEXT,
    github_url      TEXT,
    resume_url      TEXT,
    resume_text     TEXT,
    test_la         REAL,
    test_code       REAL,
    status          TEXT DEFAULT 'uploaded',
    error_notes     TEXT
);

CREATE TABLE IF NOT EXISTS scores (
    s_no              INTEGER PRIMARY KEY,
    jd_match          REAL,
    embedding_sim     REAL,
    project_quality   REAL,
    github_score      REAL,
    test_score        REAL,
    final_score       REAL,
    llm_reasoning     TEXT,
    github_breakdown  TEXT,
    rank              INTEGER,
    FOREIGN KEY (s_no) REFERENCES candidates(s_no)
);

CREATE TABLE IF NOT EXISTS interview_events (
    s_no               INTEGER PRIMARY KEY,
    calendar_event_id  TEXT,
    meet_link          TEXT,
    scheduled_time     TEXT,
    invite_sent        INTEGER DEFAULT 0,
    FOREIGN KEY (s_no) REFERENCES candidates(s_no)
);
"""


# ─── Connection helper ─────────────────────────────────────────────────────────

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def reset_db() -> None:
    """Drop and recreate all tables. Used for fresh demo runs."""
    with get_conn() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS interview_events;
            DROP TABLE IF EXISTS scores;
            DROP TABLE IF EXISTS candidates;
        """)
        conn.executescript(SCHEMA)


# ─── CSV normalization (C8) ────────────────────────────────────────────────────

def normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply COLUMN_ALIASES to incoming DataFrame headers before any schema check.
    Handles case-insensitive matching and strips whitespace.
    """
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    return df


def validate_csv(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns (empty = valid)."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    return sorted(missing)


# ─── Candidate operations ──────────────────────────────────────────────────────

def insert_candidates(df: pd.DataFrame) -> tuple[int, int]:
    """
    Normalize columns, validate schema, then bulk-insert.
    Skips rows whose s_no already exists (INSERT OR IGNORE).
    Returns (inserted_count, skipped_count).

    C1: Dedup key is s_no, NOT email. Real CSVs share a recruiter forwarding address.
    """
    df = normalize_csv_columns(df)
    missing = validate_csv(df)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Coerce types
    df["s_no"] = pd.to_numeric(df["s_no"], errors="coerce").astype("Int64")
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")

    # Drop rows with null s_no
    df = df.dropna(subset=["s_no"])

    # Strip test columns if present in CSV (demo flow: candidates.csv has no test cols)
    df = df.drop(columns=["test_la", "test_code"], errors="ignore")

    cols = [
        "s_no", "name", "email", "college", "branch",
        "cgpa", "best_ai_project", "research_work",
        "github_url", "resume_url",
    ]
    # Keep only known columns that exist in df
    cols = [c for c in cols if c in df.columns]

    inserted, skipped = 0, 0
    with get_conn() as conn:
        existing = {row["s_no"] for row in conn.execute("SELECT s_no FROM candidates")}
        for _, row in df.iterrows():
            sno = int(row["s_no"])
            if sno in existing:
                skipped += 1
                continue
            placeholders = ", ".join("?" * len(cols))
            col_names = ", ".join(cols)
            values = [row.get(c) for c in cols]
            conn.execute(
                f"INSERT OR IGNORE INTO candidates ({col_names}) VALUES ({placeholders})",
                values,
            )
            inserted += 1
    return inserted, skipped


def update_candidate(s_no: int, **kwargs) -> None:
    """Generic column updater. Pass column=value pairs."""
    if not kwargs:
        return
    set_clause = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [s_no]
    with get_conn() as conn:
        conn.execute(f"UPDATE candidates SET {set_clause} WHERE s_no = ?", values)


def update_status(s_no: int, status: str, error: str | None = None) -> None:
    """Advance state machine for a single candidate."""
    with get_conn() as conn:
        if error:
            conn.execute(
                "UPDATE candidates SET status = ?, error_notes = ? WHERE s_no = ?",
                (status, error, s_no),
            )
        else:
            conn.execute(
                "UPDATE candidates SET status = ? WHERE s_no = ?",
                (status, s_no),
            )


def get_by_status(status: str) -> list[sqlite3.Row]:
    """
    Return all candidates at a given status stage.
    C9: Callers MUST iterate over these rows — never assume contiguous s_no.
    """
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM candidates WHERE status = ?", (status,)
        ).fetchall()


def get_candidates_past_status(status: str) -> list[sqlite3.Row]:
    """Return all candidates at or past a given status in the flow."""
    idx = STATUS_FLOW.index(status) if status in STATUS_FLOW else 0
    statuses = STATUS_FLOW[idx:]
    placeholders = ", ".join("?" * len(statuses))
    with get_conn() as conn:
        return conn.execute(
            f"SELECT * FROM candidates WHERE status IN ({placeholders})", statuses
        ).fetchall()


def get_all_candidates() -> pd.DataFrame:
    """
    Full join: candidates + scores + interview_events.
    Used for the dashboard table — always fresh from DB, no cached state.
    """
    query = """
        SELECT
            c.s_no, c.name, c.email, c.college, c.branch, c.cgpa,
            c.best_ai_project, c.research_work,
            c.github_url, c.resume_url, c.resume_text,
            c.test_la, c.test_code, c.status, c.error_notes,
            s.jd_match, s.embedding_sim, s.project_quality,
            s.github_score, s.test_score, s.final_score,
            s.llm_reasoning, s.github_breakdown, s.rank,
            ie.meet_link, ie.scheduled_time, ie.invite_sent
        FROM candidates c
        LEFT JOIN scores s ON c.s_no = s.s_no
        LEFT JOIN interview_events ie ON c.s_no = ie.s_no
        ORDER BY COALESCE(s.rank, 9999), c.s_no
    """
    with get_conn() as conn:
        return pd.read_sql_query(query, conn)


def get_candidate(s_no: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM candidates WHERE s_no = ?", (s_no,)
        ).fetchone()


# ─── Score operations ──────────────────────────────────────────────────────────

def upsert_score(s_no: int, **kwargs) -> None:
    """Insert or replace a score row. Pass any subset of score columns."""
    if not kwargs:
        return
    kwargs["s_no"] = s_no
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    conflict_set = ", ".join(
        f"{k} = excluded.{k}" for k in kwargs if k != "s_no"
    )
    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO scores ({cols}) VALUES ({placeholders})
            ON CONFLICT(s_no) DO UPDATE SET {conflict_set}
            """,
            list(kwargs.values()),
        )


def get_score(s_no: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM scores WHERE s_no = ?", (s_no,)
        ).fetchone()


def get_all_scores() -> list[sqlite3.Row]:
    with get_conn() as conn:
        return conn.execute("SELECT * FROM scores").fetchall()


# ─── Test result operations ────────────────────────────────────────────────────

def merge_test_results(test_df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Merge test_la, test_code from a test-results CSV into candidates table.
    CSV must have columns: s_no, test_la, test_code.
    Returns (merged_count, warnings[]).
    C9: Joined on s_no. No sequential assumptions.
    """
    test_df = normalize_csv_columns(test_df)
    required = {"s_no", "test_la", "test_code"}
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Test results CSV missing columns: {missing}")

    test_df["s_no"] = pd.to_numeric(test_df["s_no"], errors="coerce").astype("Int64")
    test_df["test_la"] = pd.to_numeric(test_df["test_la"], errors="coerce")
    test_df["test_code"] = pd.to_numeric(test_df["test_code"], errors="coerce")
    test_df = test_df.dropna(subset=["s_no"])

    merged, warnings = 0, []
    with get_conn() as conn:
        existing_snos = {
            row["s_no"] for row in conn.execute("SELECT s_no FROM candidates")
        }
        for _, row in test_df.iterrows():
            sno = int(row["s_no"])
            if sno not in existing_snos:
                warnings.append(f"s_no {sno} not found in candidates — skipped")
                continue
            conn.execute(
                "UPDATE candidates SET test_la = ?, test_code = ? WHERE s_no = ?",
                (row["test_la"], row["test_code"], sno),
            )
            merged += 1
    return merged, warnings


# ─── Interview event operations ────────────────────────────────────────────────

def upsert_interview_event(s_no: int, **kwargs) -> None:
    """Insert or replace an interview event row."""
    kwargs["s_no"] = s_no
    cols = ", ".join(kwargs.keys())
    placeholders = ", ".join("?" * len(kwargs))
    conflict_set = ", ".join(
        f"{k} = excluded.{k}" for k in kwargs if k != "s_no"
    )
    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO interview_events ({cols}) VALUES ({placeholders})
            ON CONFLICT(s_no) DO UPDATE SET {conflict_set}
            """,
            list(kwargs.values()),
        )


def get_interview_event(s_no: int) -> sqlite3.Row | None:
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM interview_events WHERE s_no = ?", (s_no,)
        ).fetchone()


# ─── Quick self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, os
    DB_PATH = Path(tempfile.mktemp(suffix=".db"))
    init_db()
    print("Tables created.")

    sample = pd.DataFrame([
        {
            "s_no": 1, "name": "Alice", "email": "recruiter@example.com",
            "college": "IIT", "branch": "CSE", "cgpa": 9.1,
            "best_ai_project": "LLM chatbot", "research_work": "NLP paper",
            "github_url": "https://github.com/alice", "resume_url": "https://drive.google.com/..."
        },
        {
            "s_no": 3, "name": "Bob", "email": "recruiter@example.com",
            "college": "NIT", "branch": "ECE", "cgpa": 8.5,
            "best_ai_project": "CV model", "research_work": "",
            "github_url": "", "resume_url": "https://drive.google.com/..."
        },
    ])
    ins, skp = insert_candidates(sample)
    print(f"Inserted: {ins}, Skipped: {skp}")

    # Test alias normalization
    aliased = pd.DataFrame([{
        "S.No": 5, "name": "Carol", "email": "recruiter@example.com",
        "college": "VIT", "branch": "IT", "cgpa": 8.0,
        "best_ai_project": "RL agent", "research_work": "",
        "github": "https://github.com/carol", "resume": "https://drive.google.com/..."
    }])
    ins2, _ = insert_candidates(aliased)
    print(f"Alias test inserted: {ins2}")

    df = get_all_candidates()
    print(f"Candidates in DB: {len(df)}")
    print(df[["s_no", "name", "github_url"]].to_string())

    upsert_score(1, jd_match=82.0, project_quality=75.0, github_score=88.0, final_score=81.5, rank=1)
    upsert_score(3, jd_match=60.0, project_quality=55.0, github_score=None, final_score=57.5, rank=2)
    print("Scores upserted.")

    update_status(1, "ranked")
    update_status(3, "github_analyzed", error="no_github_url")
    rows = get_by_status("ranked")
    print(f"Ranked candidates: {[dict(r) for r in rows]}")

    os.unlink(DB_PATH)
    print("Self-test passed.")

"""Thread-safe SQLite persistence for recommendation governance and audit.

Every recommendation the system surfaces to a user can be committed to an
immutable ``audit_logs`` ledger. The ledger records not only the prediction but
the *evidence* behind it — the dominant SHAP driver and the SHAP/LIME Jaccard
agreement index — so that an agricultural extension officer can retrospectively
justify any advice that was acted upon in the field.

Concurrency model
-----------------
Streamlit dispatches each browser session on its own worker thread, and SQLite
connection objects are not safe to share across threads. This module therefore
keeps one connection **per thread** in a :class:`threading.local` store, and
serialises writes behind a module-level lock so that concurrent sessions cannot
interleave a transaction. Write-ahead logging is enabled so that readers are
never blocked by an in-flight writer.

All statements are parameterised; no value is ever interpolated into SQL text.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import pandas as pd

from src.core.config import DATABASE_PATH

logger = logging.getLogger(__name__)

#: Serialises write transactions across Streamlit worker threads.
_WRITE_LOCK = threading.Lock()

#: Per-thread connection registry: ``{db_path: sqlite3.Connection}``.
_THREAD_LOCAL = threading.local()

#: Schema version, bumped whenever :data:`_SCHEMA` changes shape. Persisted in
#: SQLite's ``user_version`` pragma and used to drive forward migrations.
SCHEMA_VERSION: int = 3

TABLE_NAME: str = "audit_logs"

#: Accounts. A row here is created only when someone chooses to sign up;
#: guest use writes ``user_id IS NULL`` and is the default path.
USERS_TABLE: str = "users"

#: Server-side sessions. Tokens are stored hashed, never in the clear, so a
#: dump of this table cannot be replayed as a login.
SESSIONS_TABLE: str = "sessions"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    district            TEXT    NOT NULL DEFAULT 'Unspecified',
    N                   REAL    NOT NULL CHECK (N  >= 0),
    P                   REAL    NOT NULL CHECK (P  >= 0),
    K                   REAL    NOT NULL CHECK (K  >= 0),
    pH                  REAL    NOT NULL CHECK (pH > 0),
    temp                REAL    NOT NULL,
    humidity            REAL    NOT NULL,
    rainfall            REAL    NOT NULL CHECK (rainfall >= 0),
    recommended_crop    TEXT    NOT NULL,
    confidence          REAL    NOT NULL CHECK (confidence BETWEEN 0 AND 100),
    primary_shap_driver TEXT,
    jaccard_index       REAL    CHECK (jaccard_index IS NULL
                                       OR jaccard_index BETWEEN 0 AND 1)
);
"""

_USERS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {USERS_TABLE} (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    phone         TEXT    NOT NULL UNIQUE,
    pin_hash      TEXT    NOT NULL,
    display_name  TEXT,
    village       TEXT,
    district      TEXT,
    acres         REAL    CHECK (acres IS NULL OR acres > 0),
    language      TEXT    NOT NULL DEFAULT 'en',
    recovery_hash TEXT,
    created_at    TEXT    NOT NULL,
    last_login_at TEXT,
    failed_count  INTEGER NOT NULL DEFAULT 0,
    locked_until  TEXT
);
"""

_SESSIONS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {SESSIONS_TABLE} (
    token_hash TEXT    PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    created_at TEXT    NOT NULL,
    expires_at TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES {USERS_TABLE}(id) ON DELETE CASCADE
);
"""

#: A farmer's saved fields. The reason an account is worth having: someone
#: with three plots should not retype seven readings for each one.
PLOTS_TABLE: str = "plots"

_PLOTS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {PLOTS_TABLE} (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    name       TEXT    NOT NULL,
    district   TEXT    NOT NULL,
    acres      REAL    NOT NULL CHECK (acres > 0),
    N          REAL    NOT NULL CHECK (N  >= 0),
    P          REAL    NOT NULL CHECK (P  >= 0),
    K          REAL    NOT NULL CHECK (K  >= 0),
    pH         REAL    NOT NULL CHECK (pH > 0),
    temp       REAL    NOT NULL,
    humidity   REAL    NOT NULL,
    rainfall   REAL    NOT NULL CHECK (rainfall >= 0),
    created_at TEXT    NOT NULL,
    UNIQUE (user_id, name),
    FOREIGN KEY (user_id) REFERENCES {USERS_TABLE}(id) ON DELETE CASCADE
);
"""

_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_timestamp "
    f"ON {TABLE_NAME} (timestamp DESC);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_district "
    f"ON {TABLE_NAME} (district);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_crop "
    f"ON {TABLE_NAME} (recommended_crop);",
)

#: Column order used by :func:`log_transaction`'s INSERT and by the audit view.
LOG_COLUMNS: Sequence[str] = (
    "timestamp",
    "district",
    "N",
    "P",
    "K",
    "pH",
    "temp",
    "humidity",
    "rainfall",
    "recommended_crop",
    "confidence",
    "primary_shap_driver",
    "jaccard_index",
    # NULL means guest. Every row written before accounts existed is a guest
    # row, which is the correct reading of it.
    "user_id",
)


# --------------------------------------------------------------------------- #
# Connection management
# --------------------------------------------------------------------------- #
def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a tuned SQLite connection for the calling thread."""
    conn = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    # WAL lets readers proceed during writes; NORMAL synchronous is the
    # standard durability/throughput compromise for WAL mode.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except sqlite3.DatabaseError:  # pragma: no cover - e.g. read-only volume
        logger.debug("Could not apply WAL pragmas to %s", db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _thread_connection(db_path: Path) -> sqlite3.Connection:
    """Return this thread's cached connection to ``db_path``, opening if needed."""
    registry = getattr(_THREAD_LOCAL, "connections", None)
    if registry is None:
        registry = {}
        _THREAD_LOCAL.connections = registry

    key = str(db_path)
    conn = registry.get(key)
    if conn is not None:
        try:  # Cheap liveness probe — the handle may have been closed elsewhere.
            conn.execute("SELECT 1;")
            return conn
        except sqlite3.ProgrammingError:
            registry.pop(key, None)

    conn = _connect(db_path)
    registry[key] = conn
    return conn


@contextmanager
def get_connection(
    db_path: Optional[Path] = None, *, write: bool = False
) -> Iterator[sqlite3.Connection]:
    """Yield a thread-local connection inside a managed transaction.

    Parameters
    ----------
    db_path:
        Target database file. Defaults to :data:`~src.core.config.DATABASE_PATH`.
    write:
        When ``True`` the block is serialised behind the module write lock and
        committed on clean exit. Any exception triggers a full rollback, so a
        partially-applied transaction can never be observed.

    Yields
    ------
    sqlite3.Connection
        A live connection owned by the calling thread.

    Raises
    ------
    sqlite3.Error
        Re-raised after rollback so callers can surface the failure.
    """
    path = Path(db_path) if db_path is not None else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = _thread_connection(path)

    if not write:
        yield conn
        return

    with _WRITE_LOCK:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            logger.exception("Transaction rolled back on %s", path)
            raise


def close_connections() -> None:
    """Close every connection owned by the calling thread.

    Chiefly used by the test-suite to release file handles between temporary
    databases; long-lived application threads simply reuse their connection.
    """
    registry = getattr(_THREAD_LOCAL, "connections", None) or {}
    for conn in registry.values():
        try:
            conn.close()
        except sqlite3.Error:  # pragma: no cover - defensive
            pass
    registry.clear()


# --------------------------------------------------------------------------- #
# Schema lifecycle
# --------------------------------------------------------------------------- #
def init_db(db_path: Optional[Path] = None) -> None:
    """Create the ``audit_logs`` table and indexes if they do not yet exist.

    Idempotent, and safe to call from every worker thread on startup. Applies
    any pending forward migration via :func:`_migrate`.
    """
    with get_connection(db_path, write=True) as conn:
        _create_all(conn)
        for statement in _INDEXES:
            conn.execute(statement)
        _migrate(conn)
    logger.debug("Schema initialised at version %d", SCHEMA_VERSION)


def _create_all(conn: sqlite3.Connection) -> None:
    """Create every table. Idempotent -- all use IF NOT EXISTS."""
    for ddl in (_SCHEMA, _USERS_SCHEMA, _SESSIONS_SCHEMA, _PLOTS_SCHEMA):
        conn.executescript(ddl)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply forward migrations from the stored ``user_version`` to current.

    Version 0 denotes a database created before versioning was introduced; it
    is structurally identical to version 1, so the upgrade is a stamp only.
    Future schema changes append an ``if current < n`` block here.
    """
    current = int(conn.execute("PRAGMA user_version;").fetchone()[0])
    if current == SCHEMA_VERSION:
        return
    if current > SCHEMA_VERSION:
        logger.warning(
            "Database schema v%d is newer than this build (v%d); "
            "leaving it untouched.",
            current,
            SCHEMA_VERSION,
        )
        return

    existing = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({TABLE_NAME});").fetchall()
    }
    # Additive column migration: bring legacy tables up to the current shape.
    #
    # v2 adds accounts. Every pre-existing row predates them and is therefore
    # guest-owned: the new column is nullable with no default, so ALTER TABLE
    # leaves those rows intact with user_id IS NULL. Rewriting or deleting
    # them to fit the new shape would destroy the ledger this system exists
    # to keep.
    for column, ddl in (
        ("primary_shap_driver", "TEXT"),
        ("jaccard_index", "REAL"),
        ("user_id", "INTEGER"),
    ):
        if existing and column not in existing:
            conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {column} {ddl};")
            logger.info("Migrated: added column %s.%s", TABLE_NAME, column)

    # v3: a one-time recovery code, so a forgotten PIN does not destroy the
    # account. Nullable, so accounts created under v2 keep working and can
    # mint a code from Settings whenever they next sign in.
    user_columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({USERS_TABLE});").fetchall()
    }
    if user_columns and "recovery_hash" not in user_columns:
        conn.execute(f"ALTER TABLE {USERS_TABLE} ADD COLUMN recovery_hash TEXT;")
        logger.info("Migrated: added column %s.recovery_hash", USERS_TABLE)

    # The v2 tables are created by _create_all, which runs before this and is
    # idempotent; nothing to backfill for them.
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_user "
        f"ON {TABLE_NAME} (user_id);"
    )
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION};")


def get_schema_version(db_path: Optional[Path] = None) -> int:
    """Return the ``user_version`` pragma of the target database."""
    with get_connection(db_path) as conn:
        return int(conn.execute("PRAGMA user_version;").fetchone()[0])


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #
def log_transaction(
    district: str,
    n: float,
    p: float,
    k: float,
    ph: float,
    temperature: float,
    humidity: float,
    rainfall: float,
    recommended_crop: str,
    confidence: float,
    primary_shap_driver: Optional[str] = None,
    jaccard_index: Optional[float] = None,
    *,
    user_id: Optional[int] = None,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Persist a single recommendation to the audit ledger.

    Parameters
    ----------
    district:
        Administrative unit the reading belongs to.
    n, p, k:
        Macronutrient concentrations in kg/ha.
    ph:
        Soil reaction.
    temperature, humidity, rainfall:
        Microclimate at the time of the recommendation (°C, %, mm).
    recommended_crop:
        Argmax class emitted by the stacking ensemble.
    confidence:
        Meta-learner posterior for that class, as a **percentage** in [0, 100].
    primary_shap_driver:
        Highest-magnitude TreeSHAP feature, when an XAI pass was run.
    jaccard_index:
        SHAP/LIME top-k agreement in [0, 1], when an XAI pass was run.
    timestamp:
        ISO-8601 override, primarily for deterministic tests. Defaults to now.
    db_path:
        Target database; defaults to the configured application database.

    Returns
    -------
    int
        The autoincrement primary key of the inserted row.

    Raises
    ------
    sqlite3.IntegrityError
        If a CHECK constraint rejects the row (e.g. negative nutrients or a
        confidence outside [0, 100]). The transaction is rolled back first, so
        the ledger is left exactly as it was.
    """
    row = (
        timestamp or datetime.now().isoformat(timespec="seconds"),
        str(district) if district else "Unspecified",
        float(n),
        float(p),
        float(k),
        float(ph),
        float(temperature),
        float(humidity),
        float(rainfall),
        str(recommended_crop),
        float(confidence),
        str(primary_shap_driver) if primary_shap_driver is not None else None,
        float(jaccard_index) if jaccard_index is not None else None,
        int(user_id) if user_id is not None else None,
    )
    placeholders = ", ".join("?" for _ in LOG_COLUMNS)
    sql = (
        f"INSERT INTO {TABLE_NAME} ({', '.join(LOG_COLUMNS)}) "
        f"VALUES ({placeholders});"
    )

    with get_connection(db_path, write=True) as conn:
        cursor = conn.execute(sql, row)
        record_id = int(cursor.lastrowid or 0)

    logger.info("Logged recommendation #%d (%s)", record_id, recommended_crop)
    return record_id


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #
def fetch_audit_history(
    limit: int = 100,
    *,
    district: Optional[str] = None,
    crop: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user_id: Optional[int] = None,
    guest_only: bool = False,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return the most recent audit records as a DataFrame, newest first.

    Parameters
    ----------
    limit:
        Maximum rows to return. Non-positive values are treated as unbounded.
    district, crop:
        Optional exact-match filters.
    start_date, end_date:
        Inclusive ISO-8601 date (or datetime) bounds on ``timestamp``. A bare
        ``YYYY-MM-DD`` end bound is widened to the end of that day so the
        lexicographic comparison covers the full date.
    db_path:
        Target database; defaults to the configured application database.

    Returns
    -------
    pandas.DataFrame
        Audit rows, or an empty frame with the correct columns if the table is
        empty or unreadable. Never raises for a missing table.
    """
    clauses: list[str] = []
    params: list[Any] = []

    if district:
        clauses.append("district = ?")
        params.append(district)
    if crop:
        clauses.append("recommended_crop = ?")
        params.append(crop)
    if start_date:
        clauses.append("timestamp >= ?")
        params.append(str(start_date))
    if end_date:
        bound = str(end_date)
        if len(bound) == 10:  # bare date -> inclusive of the whole day
            bound += "T23:59:59"
        clauses.append("timestamp <= ?")
        params.append(bound)

    sql = f"SELECT id, {', '.join(LOG_COLUMNS)} FROM {TABLE_NAME}"
    # Ownership scoping. This is the clause that stops one farmer reading
    # another's ledger, so it is applied here in SQL rather than by filtering
    # a DataFrame afterwards -- a filter that a caller can forget to apply is
    # not access control.
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(int(user_id))
    elif guest_only:
        clauses.append("user_id IS NULL")

    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY id DESC"
    if limit and limit > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    empty = pd.DataFrame(columns=["id", *LOG_COLUMNS])
    try:
        with get_connection(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        logger.warning("Audit history unavailable: %s", exc)
        return empty

    if not rows:
        return empty
    return pd.DataFrame([dict(row) for row in rows], columns=["id", *LOG_COLUMNS])


def count_records(db_path: Optional[Path] = None) -> int:
    """Return the total number of rows in the audit ledger (0 if absent)."""
    try:
        with get_connection(db_path) as conn:
            return int(conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0])
    except sqlite3.Error:
        return 0


def clear_history(db_path: Optional[Path] = None) -> int:
    """Delete every audit row and return how many were removed.

    Provided for test fixtures and operator-initiated resets; the dashboard
    deliberately does not expose this.
    """
    with get_connection(db_path, write=True) as conn:
        cursor = conn.execute(f"DELETE FROM {TABLE_NAME};")
        return int(cursor.rowcount or 0)


__all__ = [
    "SCHEMA_VERSION",
    "TABLE_NAME",
    "LOG_COLUMNS",
    "get_connection",
    "close_connections",
    "init_db",
    "get_schema_version",
    "log_transaction",
    "fetch_audit_history",
    "count_records",
    "clear_history",
]

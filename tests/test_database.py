"""Verification of the SQLite audit ledger.

Covers schema creation and migration, parameterised CRUD, constraint-driven
transaction rollback, filtering, SQL-injection resistance, and concurrent
writes from multiple threads.
"""

from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import pytest

from src.database import db_manager
from src.database.db_manager import (
    LOG_COLUMNS,
    SCHEMA_VERSION,
    TABLE_NAME,
    clear_history,
    count_records,
    fetch_audit_history,
    get_connection,
    get_schema_version,
    init_db,
    log_transaction,
)


def _record(**overrides: Any) -> Dict[str, Any]:
    """A valid audit record, with optional field overrides."""
    base: Dict[str, Any] = {
        "district": "Udupi",
        "n": 90.0,
        "p": 42.0,
        "k": 43.0,
        "ph": 6.5,
        "temperature": 20.88,
        "humidity": 82.0,
        "rainfall": 202.94,
        "recommended_crop": "rice",
        "confidence": 97.54,
        "primary_shap_driver": "rainfall",
        "jaccard_index": 1.0,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class TestSchemaLifecycle:
    """Schema creation must be idempotent and versioned."""

    def test_init_creates_the_database_file(self, tmp_path: Path) -> None:
        path = tmp_path / "fresh.db"
        init_db(path)
        assert path.exists()
        db_manager.close_connections()

    def test_init_creates_the_audit_table(self, temp_db: Path) -> None:
        with get_connection(temp_db) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
                (TABLE_NAME,),
            ).fetchone()
        assert row is not None

    def test_table_carries_every_declared_column(self, temp_db: Path) -> None:
        with get_connection(temp_db) as conn:
            columns = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({TABLE_NAME});").fetchall()
            }
        assert {"id", *LOG_COLUMNS}.issubset(columns)

    def test_init_is_idempotent(self, temp_db: Path) -> None:
        log_transaction(**_record(), db_path=temp_db)
        init_db(temp_db)  # Must not drop or duplicate anything.
        init_db(temp_db)
        assert count_records(temp_db) == 1

    def test_schema_version_is_stamped(self, temp_db: Path) -> None:
        assert get_schema_version(temp_db) == SCHEMA_VERSION

    def test_indexes_are_created(self, temp_db: Path) -> None:
        with get_connection(temp_db) as conn:
            indexes = {
                str(row["name"])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index';"
                ).fetchall()
            }
        assert any("timestamp" in name for name in indexes)

    def test_migration_adds_columns_to_a_legacy_table(self, tmp_path: Path) -> None:
        """A pre-versioning table must gain the XAI columns, keeping its rows."""
        path = tmp_path / "legacy.db"
        legacy = sqlite3.connect(str(path))
        legacy.execute(
            f"""CREATE TABLE {TABLE_NAME} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, district TEXT,
                    N REAL, P REAL, K REAL, pH REAL, temp REAL,
                    humidity REAL, rainfall REAL,
                    recommended_crop TEXT, confidence REAL);"""
        )
        legacy.execute(
            f"INSERT INTO {TABLE_NAME} (timestamp, district, N, P, K, pH, temp, "
            f"humidity, rainfall, recommended_crop, confidence) "
            f"VALUES (?,?,?,?,?,?,?,?,?,?,?);",
            ("2024-01-01T00:00:00", "Udupi", 90, 42, 43, 6.5, 21, 82, 203, "rice", 97.5),
        )
        legacy.commit()
        legacy.close()

        init_db(path)
        try:
            with get_connection(path) as conn:
                columns = {
                    str(row["name"])
                    for row in conn.execute(f"PRAGMA table_info({TABLE_NAME});").fetchall()
                }
            assert "primary_shap_driver" in columns
            assert "jaccard_index" in columns
            assert get_schema_version(path) == SCHEMA_VERSION
            assert count_records(path) == 1, "Migration must preserve existing rows"
        finally:
            db_manager.close_connections()


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
class TestWritePath:
    """Inserts must be parameterised, validated, and monotonic."""

    def test_insert_returns_a_positive_identifier(self, temp_db: Path) -> None:
        assert log_transaction(**_record(), db_path=temp_db) > 0

    def test_identifiers_increment(self, temp_db: Path) -> None:
        first = log_transaction(**_record(), db_path=temp_db)
        second = log_transaction(**_record(), db_path=temp_db)
        assert second > first

    def test_values_round_trip_faithfully(self, temp_db: Path) -> None:
        log_transaction(**_record(), db_path=temp_db)
        row = fetch_audit_history(db_path=temp_db).iloc[0]
        assert row["district"] == "Udupi"
        assert row["N"] == pytest.approx(90.0)
        assert row["recommended_crop"] == "rice"
        assert row["confidence"] == pytest.approx(97.54)
        assert row["primary_shap_driver"] == "rainfall"
        assert row["jaccard_index"] == pytest.approx(1.0)

    def test_optional_xai_fields_may_be_null(self, temp_db: Path) -> None:
        log_transaction(
            **_record(primary_shap_driver=None, jaccard_index=None), db_path=temp_db
        )
        row = fetch_audit_history(db_path=temp_db).iloc[0]
        assert pd.isna(row["primary_shap_driver"])
        assert pd.isna(row["jaccard_index"])

    def test_blank_district_defaults_to_unspecified(self, temp_db: Path) -> None:
        log_transaction(**_record(district=""), db_path=temp_db)
        assert fetch_audit_history(db_path=temp_db).iloc[0]["district"] == "Unspecified"

    def test_explicit_timestamp_is_honoured(self, temp_db: Path) -> None:
        log_transaction(**_record(), timestamp="2025-01-15T08:30:00", db_path=temp_db)
        assert (
            fetch_audit_history(db_path=temp_db).iloc[0]["timestamp"]
            == "2025-01-15T08:30:00"
        )

    def test_quotes_in_input_are_stored_verbatim(self, temp_db: Path) -> None:
        """Parameterised statements must neither execute nor mangle the text."""
        hostile = "O'Brien; DROP TABLE audit_logs;--"
        log_transaction(**_record(district=hostile), db_path=temp_db)
        frame = fetch_audit_history(db_path=temp_db)
        assert frame.iloc[0]["district"] == hostile
        assert count_records(temp_db) == 1, "The table must still exist"


# --------------------------------------------------------------------------- #
# Constraints and rollback
# --------------------------------------------------------------------------- #
class TestConstraintsAndRollback:
    """A rejected write must leave the ledger byte-for-byte unchanged."""

    @pytest.mark.parametrize("field", ["n", "p", "k"])
    def test_negative_nutrients_are_rejected(self, temp_db: Path, field: str) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(**{field: -1.0}), db_path=temp_db)

    def test_non_positive_ph_is_rejected(self, temp_db: Path) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(ph=0.0), db_path=temp_db)

    def test_negative_rainfall_is_rejected(self, temp_db: Path) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(rainfall=-5.0), db_path=temp_db)

    @pytest.mark.parametrize("confidence", [-0.1, 100.1, 1000.0])
    def test_confidence_outside_the_percentage_range_is_rejected(
        self, temp_db: Path, confidence: float
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(confidence=confidence), db_path=temp_db)

    @pytest.mark.parametrize("jaccard", [-0.01, 1.01])
    def test_jaccard_outside_the_unit_interval_is_rejected(
        self, temp_db: Path, jaccard: float
    ) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(jaccard_index=jaccard), db_path=temp_db)

    def test_rejected_write_does_not_change_the_row_count(self, temp_db: Path) -> None:
        log_transaction(**_record(), db_path=temp_db)
        before = count_records(temp_db)
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(n=-10.0), db_path=temp_db)
        assert count_records(temp_db) == before

    def test_rollback_leaves_earlier_rows_intact(self, temp_db: Path) -> None:
        log_transaction(**_record(recommended_crop="rice"), db_path=temp_db)
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(confidence=500.0), db_path=temp_db)
        frame = fetch_audit_history(db_path=temp_db)
        assert len(frame) == 1
        assert frame.iloc[0]["recommended_crop"] == "rice"

    def test_connection_is_usable_after_a_rollback(self, temp_db: Path) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            log_transaction(**_record(n=-1.0), db_path=temp_db)
        assert log_transaction(**_record(), db_path=temp_db) > 0

    def test_explicit_transaction_rolls_back_on_error(self, temp_db: Path) -> None:
        """A failure mid-block must discard every statement in that block."""
        log_transaction(**_record(), db_path=temp_db)
        before = count_records(temp_db)

        with pytest.raises(RuntimeError):
            with get_connection(temp_db, write=True) as conn:
                conn.execute(
                    f"INSERT INTO {TABLE_NAME} ({', '.join(LOG_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in LOG_COLUMNS)});",
                    (
                        "2025-01-01T00:00:00", "Udupi", 1.0, 1.0, 1.0, 6.0,
                        20.0, 50.0, 100.0, "rice", 50.0, "N", 0.5,
                    ),
                )
                raise RuntimeError("simulated mid-transaction failure")

        assert count_records(temp_db) == before


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
class TestReadPath:
    """Queries must be ordered, filterable, and safe when empty."""

    def test_empty_ledger_returns_a_typed_empty_frame(self, temp_db: Path) -> None:
        frame = fetch_audit_history(db_path=temp_db)
        assert isinstance(frame, pd.DataFrame)
        assert frame.empty
        assert list(frame.columns) == ["id", *LOG_COLUMNS]

    def test_missing_table_returns_empty_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        frame = fetch_audit_history(db_path=tmp_path / "never_initialised.db")
        assert frame.empty
        db_manager.close_connections()

    def test_results_are_newest_first(self, temp_db: Path) -> None:
        for crop in ("rice", "maize", "cotton"):
            log_transaction(**_record(recommended_crop=crop), db_path=temp_db)
        frame = fetch_audit_history(db_path=temp_db)
        assert frame.iloc[0]["recommended_crop"] == "cotton"
        assert list(frame["id"]) == sorted(frame["id"], reverse=True)

    def test_limit_is_respected(self, temp_db: Path) -> None:
        for _ in range(10):
            log_transaction(**_record(), db_path=temp_db)
        assert len(fetch_audit_history(limit=4, db_path=temp_db)) == 4

    def test_non_positive_limit_returns_everything(self, temp_db: Path) -> None:
        for _ in range(5):
            log_transaction(**_record(), db_path=temp_db)
        assert len(fetch_audit_history(limit=0, db_path=temp_db)) == 5

    def test_district_filter(self, temp_db: Path) -> None:
        log_transaction(**_record(district="Udupi"), db_path=temp_db)
        log_transaction(**_record(district="Mysuru"), db_path=temp_db)
        frame = fetch_audit_history(district="Mysuru", db_path=temp_db)
        assert len(frame) == 1
        assert frame.iloc[0]["district"] == "Mysuru"

    def test_crop_filter(self, temp_db: Path) -> None:
        log_transaction(**_record(recommended_crop="rice"), db_path=temp_db)
        log_transaction(**_record(recommended_crop="maize"), db_path=temp_db)
        assert len(fetch_audit_history(crop="rice", db_path=temp_db)) == 1

    def test_date_range_filter(self, temp_db: Path) -> None:
        log_transaction(**_record(), timestamp="2025-01-10T10:00:00", db_path=temp_db)
        log_transaction(**_record(), timestamp="2025-06-15T10:00:00", db_path=temp_db)
        log_transaction(**_record(), timestamp="2025-12-20T10:00:00", db_path=temp_db)
        frame = fetch_audit_history(
            start_date="2025-02-01", end_date="2025-11-30", db_path=temp_db
        )
        assert len(frame) == 1
        assert frame.iloc[0]["timestamp"].startswith("2025-06")

    def test_bare_end_date_covers_the_whole_day(self, temp_db: Path) -> None:
        """An end bound of '2025-06-15' must include 23:59 on that date."""
        log_transaction(**_record(), timestamp="2025-06-15T23:30:00", db_path=temp_db)
        frame = fetch_audit_history(
            start_date="2025-06-15", end_date="2025-06-15", db_path=temp_db
        )
        assert len(frame) == 1

    def test_filters_compose(self, temp_db: Path) -> None:
        log_transaction(
            **_record(district="Udupi", recommended_crop="rice"),
            timestamp="2025-03-01T10:00:00",
            db_path=temp_db,
        )
        log_transaction(
            **_record(district="Udupi", recommended_crop="maize"),
            timestamp="2025-03-02T10:00:00",
            db_path=temp_db,
        )
        frame = fetch_audit_history(
            district="Udupi", crop="rice", start_date="2025-01-01", db_path=temp_db
        )
        assert len(frame) == 1

    def test_hostile_filter_value_is_treated_as_data(self, temp_db: Path) -> None:
        log_transaction(**_record(), db_path=temp_db)
        frame = fetch_audit_history(
            district="'; DROP TABLE audit_logs;--", db_path=temp_db
        )
        assert frame.empty
        assert count_records(temp_db) == 1, "The table must survive the query"

    def test_count_and_clear(self, temp_db: Path) -> None:
        for _ in range(3):
            log_transaction(**_record(), db_path=temp_db)
        assert count_records(temp_db) == 3
        assert clear_history(temp_db) == 3
        assert count_records(temp_db) == 0

    def test_count_on_an_uninitialised_database_is_zero(self, tmp_path: Path) -> None:
        assert count_records(tmp_path / "absent.db") == 0
        db_manager.close_connections()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
class TestThreadSafety:
    """Streamlit dispatches sessions across threads; the ledger must cope."""

    def test_concurrent_writes_all_commit(self, temp_db: Path) -> None:
        writers = 12

        def write(index: int) -> int:
            return log_transaction(
                **_record(district=f"District-{index}"), db_path=temp_db
            )

        with ThreadPoolExecutor(max_workers=6) as pool:
            ids = list(pool.map(write, range(writers)))

        assert len(set(ids)) == writers, "Identifiers must be unique"
        assert count_records(temp_db) == writers
        db_manager.close_connections()

    def test_each_thread_gets_its_own_connection(self, temp_db: Path) -> None:
        """Each thread must receive a distinct connection object.

        The connections are held in a list for the duration of the assertion.
        Comparing ``id()`` of released objects would be unsound: CPython reuses
        memory addresses, so a closed connection's address can be handed to the
        next thread's connection and the test fails at random.
        """
        connections: List[sqlite3.Connection] = []
        lock = threading.Lock()

        def capture() -> None:
            with get_connection(temp_db) as conn:
                with lock:
                    connections.append(conn)  # strong ref: no address reuse

        threads = [threading.Thread(target=capture) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(connections) == 4
        for index, conn in enumerate(connections):
            for other in connections[index + 1:]:
                assert conn is not other, "Connections must not be shared"

        for conn in connections:
            conn.close()

    def test_reads_interleave_with_writes(self, temp_db: Path) -> None:
        log_transaction(**_record(), db_path=temp_db)

        def read() -> int:
            return len(fetch_audit_history(db_path=temp_db))

        def write(_: int) -> int:
            return log_transaction(**_record(), db_path=temp_db)

        with ThreadPoolExecutor(max_workers=4) as pool:
            writes = pool.map(write, range(6))
            reads = [pool.submit(read) for _ in range(6)]
            list(writes)
            for future in reads:
                assert future.result() >= 1

        assert count_records(temp_db) == 7
        db_manager.close_connections()

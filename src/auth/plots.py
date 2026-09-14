"""A farmer's saved fields.

This is the reason to have an account at all. Someone working three plots has
three different soils, and making them retype seven readings every time they
switch is exactly the friction that stops people using the thing. Save each
field once, then pick it from a list.

Every function takes ``user_id`` and filters on it in SQL. Plots are private
to their owner and there is no unscoped read here at all -- unlike the audit
ledger, which the examiner view reads whole on purpose, a plot has no
legitimate cross-user reader.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import FEATURE_NAMES
from src.database.db_manager import PLOTS_TABLE, get_connection

logger = logging.getLogger(__name__)

#: Enough for any smallholder, and a bound on what one account can store.
MAX_PLOTS_PER_USER: int = 20

#: Column order used when a plot is written or read.
_READING_COLUMNS = ("N", "P", "K", "pH", "temp", "humidity", "rainfall")

#: Storage column -> model feature name. The ledger already uses ``temp`` and
#: ``pH``; the model uses ``temperature`` and ``ph``. Keeping the storage
#: names consistent with the ledger matters more than matching the model, so
#: the translation lives here.
_TO_FEATURE = {
    "N": "N", "P": "P", "K": "K", "pH": "ph",
    "temp": "temperature", "humidity": "humidity", "rainfall": "rainfall",
}


class PlotError(Exception):
    """A plot could not be saved."""


@dataclass(frozen=True)
class Plot:
    """One saved field."""

    id: int
    user_id: int
    name: str
    district: str
    acres: float
    readings: Dict[str, float]
    created_at: str

    @property
    def label(self) -> str:
        """``North field · Udupi · 2 acres`` -- one line for a picker."""
        unit = "acre" if self.acres == 1 else "acres"
        return f"{self.name} · {self.district} · {self.acres:g} {unit}"

    def summary(self) -> str:
        r = self.readings
        return (
            f"N {r['N']:.0f} · P {r['P']:.0f} · K {r['K']:.0f}"
            f" · pH {r['ph']:.1f} · {r['temperature']:.0f}°C"
            f" · {r['rainfall']:.0f} mm"
        )


def _row_to_plot(row: sqlite3.Row) -> Plot:
    return Plot(
        id=int(row["id"]),
        user_id=int(row["user_id"]),
        name=str(row["name"]),
        district=str(row["district"]),
        acres=float(row["acres"]),
        readings={
            feature: float(row[column]) for column, feature in _TO_FEATURE.items()
        },
        created_at=str(row["created_at"]),
    )


def save_plot(
    user_id: int,
    name: str,
    district: str,
    acres: float,
    features: Dict[str, float],
    *,
    db_path: Optional[Path] = None,
) -> Plot:
    """Create or overwrite a named plot for this user.

    Re-saving an existing name updates it rather than failing, because that
    is what a farmer correcting last season's readings means by it.
    """
    label = " ".join(str(name).strip().split())
    if not label:
        raise PlotError("Give the field a name, so you can tell them apart.")
    if len(label) > 40:
        raise PlotError("That name is too long. Keep it under 40 characters.")

    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise PlotError(f"Missing readings: {', '.join(missing)}")
    if float(acres) <= 0:
        raise PlotError("The plot size must be more than zero.")

    values = {
        column: float(features[feature])
        for column, feature in _TO_FEATURE.items()
    }
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with get_connection(db_path, write=True) as conn:
        existing = conn.execute(
            f"SELECT id FROM {PLOTS_TABLE} WHERE user_id = ? AND name = ?;",
            (user_id, label),
        ).fetchone()

        if existing is None:
            count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {PLOTS_TABLE} WHERE user_id = ?;",
                    (user_id,),
                ).fetchone()[0]
            )
            if count >= MAX_PLOTS_PER_USER:
                raise PlotError(
                    f"You already have {MAX_PLOTS_PER_USER} saved fields. "
                    f"Delete one before adding another."
                )
            assignments = ", ".join(_READING_COLUMNS)
            placeholders = ", ".join("?" for _ in _READING_COLUMNS)
            cursor = conn.execute(
                f"INSERT INTO {PLOTS_TABLE} "
                f"(user_id, name, district, acres, {assignments}, created_at) "
                f"VALUES (?, ?, ?, ?, {placeholders}, ?);",
                (
                    user_id, label, str(district), float(acres),
                    *(values[c] for c in _READING_COLUMNS), now,
                ),
            )
            plot_id = int(cursor.lastrowid or 0)
        else:
            plot_id = int(existing["id"])
            sets = ", ".join(f"{c} = ?" for c in _READING_COLUMNS)
            conn.execute(
                f"UPDATE {PLOTS_TABLE} SET district = ?, acres = ?, {sets} "
                f"WHERE id = ? AND user_id = ?;",
                (
                    str(district), float(acres),
                    *(values[c] for c in _READING_COLUMNS),
                    plot_id, user_id,
                ),
            )

        row = conn.execute(
            f"SELECT * FROM {PLOTS_TABLE} WHERE id = ? AND user_id = ?;",
            (plot_id, user_id),
        ).fetchone()
    return _row_to_plot(row)


def list_plots(user_id: int, *, db_path: Optional[Path] = None) -> List[Plot]:
    """Every plot this user owns, newest first."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM {PLOTS_TABLE} WHERE user_id = ? "
            f"ORDER BY created_at DESC, id DESC;",
            (user_id,),
        ).fetchall()
    return [_row_to_plot(row) for row in rows]


def get_plot(
    user_id: int, plot_id: int, *, db_path: Optional[Path] = None
) -> Optional[Plot]:
    """One plot, or ``None``.

    Scoped by ``user_id`` as well as ``plot_id``: a caller that passes
    someone else's id gets nothing rather than their soil readings.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {PLOTS_TABLE} WHERE id = ? AND user_id = ?;",
            (plot_id, user_id),
        ).fetchone()
    return _row_to_plot(row) if row else None


def delete_plot(
    user_id: int, plot_id: int, *, db_path: Optional[Path] = None
) -> bool:
    """Remove a plot. Returns whether anything was deleted."""
    with get_connection(db_path, write=True) as conn:
        cursor = conn.execute(
            f"DELETE FROM {PLOTS_TABLE} WHERE id = ? AND user_id = ?;",
            (plot_id, user_id),
        )
        return bool(cursor.rowcount)


def count_plots(user_id: int, *, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM {PLOTS_TABLE} WHERE user_id = ?;",
                (user_id,),
            ).fetchone()[0]
        )

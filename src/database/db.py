"""Dual-read façade over the recommendation ledger.

Two people read this table and they need different things from it.

A **farmer** wants to know what they were told and when: a date they can say
out loud, a score out of a hundred, and column headings in words. An
**examiner** wants the row exactly as stored, including the explainer-agreement
index and the dominant SHAP driver, so it can be matched against the database
and quoted in a report.

Serving one of them the other's view is the failure mode this module exists to
prevent. The farmer view had been leaking ``primary_shap_driver`` and raw ISO
timestamps into a screen that otherwise addresses the reader in plain words.

The storage engine itself -- connection pooling, WAL, migrations, the
parameterised writes -- lives in :mod:`src.database.db_manager` and is not
reimplemented here. This module is the presentation seam on top of it.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.database import db_manager
from src.database.db_manager import (  # re-exported: one import site for callers
    clear_history,
    close_connections,
    count_records,
    fetch_audit_history,
    get_connection,
    init_db,
    log_transaction,
)

#: Column order of the farmer-facing view.
FARMER_COLUMNS = [
    "Saved on", "Place", "Crop", "Match",
    "Nitrogen", "Phosphorus", "Potassium", "Soil pH", "Rain (mm)",
]

#: Stored columns that must never reach a farmer-facing table. Named rather
#: than filtered by heuristic so that adding an audit column is a deliberate
#: decision about which side of the seam it falls on.
TECHNICAL_ONLY = ("primary_shap_driver", "jaccard_index")


def record_date(stamp: object, simple: bool) -> str:
    """A saved record's date, in the register the reader is addressed in.

    Technical mode keeps the ISO form, which is what the ledger stores and
    what an examiner will want to match against the database.
    """
    parsed = pd.to_datetime(stamp, errors="coerce")
    if pd.isna(parsed):
        return "unknown date" if simple else str(stamp)[:10]
    return f"{parsed:%d %b %Y}" if simple else f"{parsed:%Y-%m-%d}"


def farmer_view(frame: pd.DataFrame) -> pd.DataFrame:
    """The ledger as a farmer can read it.

    Localised dates, an integer score out of 100, plain column names, and
    none of :data:`TECHNICAL_ONLY`.
    """
    if frame.empty:
        return pd.DataFrame(columns=FARMER_COLUMNS)

    view = pd.DataFrame(
        {
            "Saved on": pd.to_datetime(
                frame["timestamp"], errors="coerce"
            ).dt.strftime("%d %b %Y"),
            "Place": frame["district"],
            "Crop": frame["recommended_crop"].astype(str).str.title(),
            "Match": frame["confidence"].round().astype("Int64").astype(str)
            + " / 100",
            "Nitrogen": frame["N"].round().astype("Int64"),
            "Phosphorus": frame["P"].round().astype("Int64"),
            "Potassium": frame["K"].round().astype("Int64"),
            "Soil pH": frame["pH"].round(1),
            "Rain (mm)": frame["rainfall"].round().astype("Int64"),
        }
    )
    # An unparseable timestamp would otherwise render as the string "NaT".
    return view.assign(**{"Saved on": view["Saved on"].fillna("unknown")})


def examiner_view(frame: pd.DataFrame) -> pd.DataFrame:
    """The ledger exactly as stored. Deliberately a pass-through.

    Present as a named counterpart to :func:`farmer_view` so call sites read
    as a choice between two views rather than "the formatted one or the raw
    variable".
    """
    return frame


def read_ledger(
    *,
    simple: bool,
    limit: int = 200,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch the ledger already rendered for the persona asking for it."""
    frame = fetch_audit_history(
        limit=int(limit), start_date=start_date, end_date=end_date
    )
    return farmer_view(frame) if simple else examiner_view(frame)


def table_height(rows: int, cap: int = 380) -> int:
    """Fit a dataframe widget to its contents.

    Streamlit pads a fixed-height dataframe with blank rows, so a one-record
    ledger renders nine empty lines and reads as a loading failure.
    """
    return int(min(cap, 38 + 36 * max(rows, 1)))


__all__ = [
    "FARMER_COLUMNS", "TECHNICAL_ONLY",
    "record_date", "farmer_view", "examiner_view", "read_ledger",
    "table_height",
    # Re-exported storage engine.
    "init_db", "log_transaction", "fetch_audit_history", "count_records",
    "clear_history", "close_connections", "get_connection", "db_manager",
]

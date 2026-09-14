"""Server-side sessions.

Streamlit's ``session_state`` is **not** authentication. It is per-browser-tab
memory that is lost on refresh and cannot be revoked, so treating "there is a
user in session_state" as proof of sign-in gives you a login that survives
nothing and that a logout cannot reliably end.

So the session lives in the database instead. A token is issued at sign-in,
stored **hashed**, and checked on every use. ``session_state`` holds only the
opaque token, which is worth nothing without the matching row.

Known limit, stated plainly: because the token lives in ``session_state`` and
not a cookie, a browser refresh still ends the session and the farmer signs in
again. Fixing that needs a cookie component, which this deployment does not
have. The token, expiry and revocation machinery here is what makes adding one
a small change rather than a rewrite.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.auth.accounts import User, _row_to_user
from src.database.db_manager import SESSIONS_TABLE, USERS_TABLE, get_connection

logger = logging.getLogger(__name__)

#: How long a session lasts. Long enough to span a working day in a field.
SESSION_TTL_HOURS: int = 12

#: Token entropy in bytes. 32 bytes is 256 bits -- not guessable.
_TOKEN_BYTES: int = 32


def _hash_token(token: str) -> str:
    """Tokens are stored hashed.

    A plain SHA-256 is right here and bcrypt would be wrong: the input is 256
    bits of cryptographic randomness, not a guessable human secret, so there
    is nothing for a slow hash to defend against -- and this runs on every
    single request.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def issue_token(user_id: int, *, db_path: Optional[Path] = None) -> str:
    """Start a session and return its token. Store only what comes back."""
    token = secrets.token_urlsafe(_TOKEN_BYTES)
    now = _now()
    with get_connection(db_path, write=True) as conn:
        _purge_expired(conn, now)
        conn.execute(
            f"INSERT INTO {SESSIONS_TABLE} "
            f"(token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?);",
            (
                _hash_token(token),
                user_id,
                now.isoformat(timespec="seconds"),
                (now + timedelta(hours=SESSION_TTL_HOURS)).isoformat(
                    timespec="seconds"
                ),
            ),
        )
    return token


def user_for_token(
    token: Optional[str], *, db_path: Optional[Path] = None
) -> Optional[User]:
    """Resolve a token to its user, or ``None``.

    ``None`` covers every failure equally -- absent, unknown, expired, or
    belonging to a deleted account -- because the caller's only correct
    response to any of them is to treat the visitor as a guest.
    """
    if not token:
        return None
    now = _now()
    with get_connection(db_path, write=True) as conn:
        _purge_expired(conn, now)
        row = conn.execute(
            f"SELECT u.* FROM {SESSIONS_TABLE} s "
            f"JOIN {USERS_TABLE} u ON u.id = s.user_id "
            f"WHERE s.token_hash = ? AND s.expires_at > ?;",
            (_hash_token(token), now.isoformat(timespec="seconds")),
        ).fetchone()
    return _row_to_user(row) if row else None


def revoke_token(token: Optional[str], *, db_path: Optional[Path] = None) -> None:
    """End one session. Idempotent: revoking an unknown token is a no-op."""
    if not token:
        return
    with get_connection(db_path, write=True) as conn:
        conn.execute(
            f"DELETE FROM {SESSIONS_TABLE} WHERE token_hash = ?;",
            (_hash_token(token),),
        )


def revoke_all(user_id: int, *, db_path: Optional[Path] = None) -> int:
    """Sign a user out everywhere. Returns how many sessions ended."""
    with get_connection(db_path, write=True) as conn:
        cursor = conn.execute(
            f"DELETE FROM {SESSIONS_TABLE} WHERE user_id = ?;", (user_id,)
        )
        return int(cursor.rowcount or 0)


def active_sessions(user_id: int, *, db_path: Optional[Path] = None) -> int:
    now = _now().isoformat(timespec="seconds")
    with get_connection(db_path) as conn:
        return int(
            conn.execute(
                f"SELECT COUNT(*) FROM {SESSIONS_TABLE} "
                f"WHERE user_id = ? AND expires_at > ?;",
                (user_id, now),
            ).fetchone()[0]
        )


def _purge_expired(conn, now: datetime) -> None:
    """Drop dead sessions. Cheap, and keeps the table from growing forever."""
    conn.execute(
        f"DELETE FROM {SESSIONS_TABLE} WHERE expires_at <= ?;",
        (now.isoformat(timespec="seconds"),),
    )

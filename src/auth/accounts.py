"""Phone-number accounts for GREENROOT.

Sign-in is by **phone number and a 4-digit PIN**, not email and a password.
Most of the intended users do not have an email address, and a long password
typed on a phone keypad outdoors is a barrier that stops people using the
thing at all.

On the honest weakness of a 4-digit PIN
---------------------------------------
Ten thousand combinations is not much. This is a deliberate trade: the
alternative that people would actually use is no account at all. It is made
survivable rather than strong, by three things working together:

1. **bcrypt**, so each guess costs real time even with the database stolen.
2. **A hard lockout** after :data:`MAX_FAILED_ATTEMPTS` wrong PINs, which
   makes online guessing useless -- an attacker gets five tries, not ten
   thousand.
3. **Nothing valuable behind it.** An account holds crop advice and soil
   readings, not money and not identity documents. The blast radius of a
   compromised account is that someone sees which crop a stranger was
   advised to plant.

If this ever gates anything that matters, replace the PIN with an SMS OTP:
:func:`sign_in` is already shaped for it -- swap the PIN comparison for a
one-time code and nothing else in the codebase changes.

Guest use is the default and needs none of this. An account exists so that a
farmer's saved advice and plots follow them to a new phone.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt

from src.database.db_manager import PLOTS_TABLE, USERS_TABLE, get_connection

logger = logging.getLogger(__name__)

#: Wrong PINs before the account is frozen.
MAX_FAILED_ATTEMPTS: int = 5

#: How long a frozen account stays frozen.
LOCKOUT_MINUTES: int = 15

#: PIN length. Four digits is what a farmer will actually remember.
PIN_LENGTH: int = 4

#: Refused outright: these are the first PINs anyone tries.
_BANNED_PINS = frozenset(
    {"0000", "1111", "2222", "3333", "4444", "5555", "6666", "7777",
     "8888", "9999", "1234", "4321", "1122", "2580", "0123"}
)

#: bcrypt work factor. 12 is ~0.3s per verification on commodity hardware --
#: unnoticeable to a person, ruinous to a brute-force loop.
BCRYPT_ROUNDS: int = 12

#: Recovery-code alphabet. No I, L, O, U, 0 or 1: this gets written on paper
#: and read back by someone who did not write it, and those are the
#: characters people confuse.
_CODE_ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXYZ"

#: Characters per group, and groups per code. 30**8 is about 6.5e11 -- far
#: beyond guessing, and still short enough to write on the back of a card.
_CODE_GROUP: int = 4
_CODE_GROUPS: int = 2


def generate_recovery_code() -> str:
    """A fresh code, formatted as ``ABCD-EFGH`` for reading aloud."""
    groups = [
        "".join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_GROUP))
        for _ in range(_CODE_GROUPS)
    ]
    return "-".join(groups)


def _normalise_code(code: str) -> str:
    """Accept a code however it was written down.

    Lower case, missing dashes and stray spaces are all the same code. The
    two most common transcription slips are folded in as well: someone who
    writes O for zero or l for one is corrected rather than refused, since
    neither character is in the alphabet.
    """
    text = re.sub(r"[^0-9A-Za-z]", "", str(code)).upper()
    return text.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))


class AuthError(Exception):
    """Any refusal to sign in or register."""


class LockedOut(AuthError):
    """Too many wrong PINs; the account is frozen for a while."""


class WeakPin(AuthError):
    """The PIN is not four digits, or is one of the obvious ones."""


@dataclass(frozen=True)
class User:
    """A signed-in account. Never carries the PIN hash."""

    id: int
    phone: str
    display_name: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    acres: Optional[float] = None
    language: str = "en"

    @property
    def masked_phone(self) -> str:
        """``98765 43210`` -> ``•••••43210``. For display next to a logout."""
        return f"•••••{self.phone[-5:]}" if len(self.phone) >= 5 else self.phone

    @property
    def greeting(self) -> str:
        return self.display_name or self.masked_phone


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _now().isoformat(timespec="seconds")


def normalise_phone(raw: str) -> str:
    """Reduce a typed Indian mobile number to its ten digits.

    ``+91 98765 43210``, ``098765-43210`` and ``9876543210`` are the same
    person, and a farmer who types it differently on a new phone must not end
    up with a second account holding none of their history.
    """
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) > 10:
        # Drop a country code or a trunk prefix; the last ten are the number.
        digits = digits[-10:]
    if len(digits) != 10 or digits[0] not in "6789":
        raise AuthError(
            "That does not look like a mobile number. Enter the 10 digits, "
            "for example 9876543210."
        )
    return digits


def _validate_pin(pin: str) -> str:
    text = re.sub(r"\D", "", str(pin))
    if len(text) != PIN_LENGTH:
        raise WeakPin(f"The PIN must be {PIN_LENGTH} digits.")
    if text in _BANNED_PINS:
        raise WeakPin(
            "That PIN is too easy to guess. Pick something that is not a "
            "repeat or a run of digits."
        )
    return text


def _hash_pin(pin: str) -> str:
    return bcrypt.hashpw(
        pin.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)
    ).decode("ascii")


def _pin_matches(pin: str, stored: str) -> bool:
    try:
        return bcrypt.checkpw(pin.encode("utf-8"), stored.encode("ascii"))
    except (ValueError, TypeError):
        # A corrupted or truncated hash must read as "wrong PIN", never as a
        # crash and never as a pass.
        logger.warning("Unreadable PIN hash encountered; treating as a refusal.")
        return False


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        phone=str(row["phone"]),
        display_name=row["display_name"],
        village=row["village"],
        district=row["district"],
        acres=row["acres"],
        language=str(row["language"] or "en"),
    )


def register(
    phone: str,
    pin: str,
    *,
    display_name: Optional[str] = None,
    district: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> User:
    """Create an account.

    Returns the user **and a one-time recovery code**. Show that code once,
    tell them to write it down, and never display it again: only its hash is
    kept, so it cannot be looked up or re-shown.
    """
    number = normalise_phone(phone)
    checked = _validate_pin(pin)
    code = generate_recovery_code()

    with get_connection(db_path, write=True) as conn:
        existing = conn.execute(
            f"SELECT id FROM {USERS_TABLE} WHERE phone = ?;", (number,)
        ).fetchone()
        if existing:
            raise AuthError(
                "This number already has an account. Sign in with your PIN "
                "instead."
            )
        cursor = conn.execute(
            f"INSERT INTO {USERS_TABLE} "
            f"(phone, pin_hash, display_name, district, created_at, "
            f"recovery_hash) VALUES (?, ?, ?, ?, ?, ?);",
            (
                number, _hash_pin(checked), display_name, district, _stamp(),
                _hash_pin(_normalise_code(code)),
            ),
        )
        row = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE id = ?;", (cursor.lastrowid,)
        ).fetchone()
    logger.info("Account created for %s", User(0, number).masked_phone)
    # The plaintext code is returned exactly once, here. It is never stored
    # and cannot be recovered -- which is the point of it.
    return _row_to_user(row), code


def sign_in(
    phone: str, pin: str, *, db_path: Optional[Path] = None
) -> User:
    """Verify a phone and PIN.

    Raises :class:`LockedOut` when the account is frozen, and a generic
    :class:`AuthError` otherwise -- deliberately the same message whether the
    number is unknown or the PIN is wrong, so the form cannot be used to
    discover which numbers are registered.

    Note the transaction shape. ``get_connection(write=True)`` rolls back when
    an exception leaves its block, so incrementing the failure counter and
    then raising inside the same transaction discards the increment -- the
    lockout counts to one forever and the rate limiting silently does nothing.
    The counter is therefore committed in its own transaction, which is
    closed before the refusal is raised.
    """
    number = normalise_phone(phone)
    typed = re.sub(r"\D", "", str(pin))
    refusal = "Wrong number or PIN. Please try again."

    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE phone = ?;", (number,)
        ).fetchone()

    if row is None:
        raise AuthError(refusal)

    locked_until = row["locked_until"]
    if locked_until:
        until = datetime.fromisoformat(locked_until)
        if until > _now():
            minutes = max(1, int((until - _now()).total_seconds() // 60) + 1)
            raise LockedOut(
                f"Too many wrong PINs. Try again in {minutes} minute"
                + ("" if minutes == 1 else "s")
                + "."
            )

    if not _pin_matches(typed, str(row["pin_hash"])):
        failed = int(row["failed_count"]) + 1
        now_locked = failed >= MAX_FAILED_ATTEMPTS
        lock = (
            (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(
                timespec="seconds"
            )
            if now_locked
            else None
        )
        # Committed and closed before raising, or the rollback eats it.
        with get_connection(db_path, write=True) as conn:
            conn.execute(
                f"UPDATE {USERS_TABLE} SET failed_count = ?, locked_until = ? "
                f"WHERE id = ?;",
                (failed, lock, row["id"]),
            )
        if now_locked:
            raise LockedOut(
                f"Too many wrong PINs. This account is locked for "
                f"{LOCKOUT_MINUTES} minutes."
            )
        raise AuthError(refusal)

    # Success clears the counter; a lockout must not outlive a correct PIN.
    with get_connection(db_path, write=True) as conn:
        conn.execute(
            f"UPDATE {USERS_TABLE} SET failed_count = 0, locked_until = NULL, "
            f"last_login_at = ? WHERE id = ?;",
            (_stamp(), row["id"]),
        )
        fresh = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE id = ?;", (row["id"],)
        ).fetchone()
    return _row_to_user(fresh)


def reset_pin_with_code(
    phone: str, code: str, new_pin: str, *, db_path: Optional[Path] = None
) -> User:
    """Set a new PIN using the recovery code, for someone who forgot theirs.

    Without this the account is simply lost, which is what the first version
    of this module did -- it told people to start again and described that as
    policy. For a farmer who signed in once and comes back a season later,
    that is their history gone.

    The same lockout guards this path. Recovery that is not rate-limited is
    not recovery, it is a second, unguarded door into the account: an
    attacker would simply brute-force the code instead of the PIN. Using the
    code also burns it, and issues a fresh one.
    """
    number = normalise_phone(phone)
    checked = _validate_pin(new_pin)
    refusal = "Wrong number or recovery code."

    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE phone = ?;", (number,)
        ).fetchone()
    if row is None:
        raise AuthError(refusal)

    locked_until = row["locked_until"]
    if locked_until and datetime.fromisoformat(locked_until) > _now():
        raise LockedOut(
            "Too many failed attempts. Wait for the lock to clear before "
            "using your recovery code."
        )

    stored = row["recovery_hash"]
    if not stored:
        raise AuthError(
            "This account has no recovery code. It was created before "
            "recovery codes existed — sign in with your PIN and make one in "
            "Settings."
        )

    if not _pin_matches(_normalise_code(code), str(stored)):
        failed = int(row["failed_count"]) + 1
        now_locked = failed >= MAX_FAILED_ATTEMPTS
        lock = (
            (_now() + timedelta(minutes=LOCKOUT_MINUTES)).isoformat(
                timespec="seconds")
            if now_locked
            else None
        )
        # Committed before raising; see the note in sign_in().
        with get_connection(db_path, write=True) as conn:
            conn.execute(
                f"UPDATE {USERS_TABLE} SET failed_count = ?, locked_until = ? "
                f"WHERE id = ?;",
                (failed, lock, row["id"]),
            )
        if now_locked:
            raise LockedOut(
                f"Too many failed attempts. Locked for {LOCKOUT_MINUTES} "
                f"minutes."
            )
        raise AuthError(refusal)

    # Correct. Set the new PIN, burn the used code, mint a replacement, and
    # end every existing session -- a PIN reset must log out anyone already
    # holding one.
    fresh_code = generate_recovery_code()
    from src.database.db_manager import SESSIONS_TABLE

    with get_connection(db_path, write=True) as conn:
        conn.execute(
            f"UPDATE {USERS_TABLE} SET pin_hash = ?, recovery_hash = ?, "
            f"failed_count = 0, locked_until = NULL WHERE id = ?;",
            (_hash_pin(checked), _hash_pin(_normalise_code(fresh_code)),
             row["id"]),
        )
        conn.execute(
            f"DELETE FROM {SESSIONS_TABLE} WHERE user_id = ?;", (row["id"],)
        )
        updated = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE id = ?;", (row["id"],)
        ).fetchone()
    logger.info("PIN reset by recovery code for account %s", row["id"])
    return _row_to_user(updated), fresh_code


def regenerate_recovery_code(
    user_id: int, pin: str, *, db_path: Optional[Path] = None
) -> str:
    """Mint a new recovery code, replacing any existing one.

    Requires the current PIN: otherwise anyone with a borrowed unlocked phone
    could mint themselves a permanent key to the account.
    """
    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT pin_hash FROM {USERS_TABLE} WHERE id = ?;", (user_id,)
        ).fetchone()
    if row is None or not _pin_matches(
        re.sub(r"\D", "", str(pin)), str(row["pin_hash"])
    ):
        raise AuthError("The PIN is wrong.")

    code = generate_recovery_code()
    with get_connection(db_path, write=True) as conn:
        conn.execute(
            f"UPDATE {USERS_TABLE} SET recovery_hash = ? WHERE id = ?;",
            (_hash_pin(_normalise_code(code)), user_id),
        )
    return code


def has_recovery_code(user_id: int, *, db_path: Optional[Path] = None) -> bool:
    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT recovery_hash FROM {USERS_TABLE} WHERE id = ?;", (user_id,)
        ).fetchone()
    return bool(row and row["recovery_hash"])


def change_pin(
    user_id: int, old_pin: str, new_pin: str, *, db_path: Optional[Path] = None
) -> None:
    """Replace a PIN, verifying the old one first."""
    checked = _validate_pin(new_pin)
    with get_connection(db_path, write=True) as conn:
        row = conn.execute(
            f"SELECT phone, pin_hash FROM {USERS_TABLE} WHERE id = ?;",
            (user_id,),
        ).fetchone()
        if row is None or not _pin_matches(
            re.sub(r"\D", "", str(old_pin)), str(row["pin_hash"])
        ):
            raise AuthError("The current PIN is wrong.")
        conn.execute(
            f"UPDATE {USERS_TABLE} SET pin_hash = ? WHERE id = ?;",
            (_hash_pin(checked), user_id),
        )


def update_profile(
    user_id: int,
    *,
    display_name: Optional[str] = None,
    village: Optional[str] = None,
    district: Optional[str] = None,
    acres: Optional[float] = None,
    language: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> User:
    """Update the fields given; leave the rest alone."""
    fields = {
        "display_name": display_name,
        "village": village,
        "district": district,
        "acres": acres,
        "language": language,
    }
    changes = {k: v for k, v in fields.items() if v is not None}
    with get_connection(db_path, write=True) as conn:
        if changes:
            assignments = ", ".join(f"{k} = ?" for k in changes)
            conn.execute(
                f"UPDATE {USERS_TABLE} SET {assignments} WHERE id = ?;",
                (*changes.values(), user_id),
            )
        row = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE id = ?;", (user_id,)
        ).fetchone()
    if row is None:
        raise AuthError("That account no longer exists.")
    return _row_to_user(row)


def get_user(user_id: int, *, db_path: Optional[Path] = None) -> Optional[User]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {USERS_TABLE} WHERE id = ?;", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def delete_account(
    user_id: int, *, keep_ledger: bool = True, db_path: Optional[Path] = None
) -> None:
    """Delete an account, and say honestly what that does and does not remove.

    The account row, its sessions and its saved plots go. Past recommendations
    are **detached rather than destroyed** by default: they become
    guest-owned, which severs the link to the person while preserving the
    agronomic audit trail the system exists to keep.

    Pass ``keep_ledger=False`` for a true erasure when someone asks for one.
    Whatever this does, the UI must say which of the two it did.
    """
    from src.database.db_manager import SESSIONS_TABLE, TABLE_NAME

    with get_connection(db_path, write=True) as conn:
        if keep_ledger:
            conn.execute(
                f"UPDATE {TABLE_NAME} SET user_id = NULL WHERE user_id = ?;",
                (user_id,),
            )
        else:
            conn.execute(
                f"DELETE FROM {TABLE_NAME} WHERE user_id = ?;", (user_id,)
            )
        conn.execute(f"DELETE FROM {PLOTS_TABLE} WHERE user_id = ?;", (user_id,))
        conn.execute(
            f"DELETE FROM {SESSIONS_TABLE} WHERE user_id = ?;", (user_id,)
        )
        conn.execute(f"DELETE FROM {USERS_TABLE} WHERE id = ?;", (user_id,))
    logger.info("Account %s deleted (ledger kept: %s)", user_id, keep_ledger)


def count_users(*, db_path: Optional[Path] = None) -> int:
    with get_connection(db_path) as conn:
        return int(
            conn.execute(f"SELECT COUNT(*) FROM {USERS_TABLE};").fetchone()[0]
        )

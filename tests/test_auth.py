"""Accounts, sessions, and ledger isolation between users."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

from src import auth
from src.auth import accounts, sessions
from src.database import db_manager as dbm


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "auth.db"
    dbm.init_db(path)
    yield path
    dbm.close_connections()


@pytest.fixture
def alice(db):
    user, _code = auth.register("9876543210", "4729", display_name="Alice",
                                district="Udupi", db_path=db)
    return user


LOG = dict(n=60, p=40, k=45, ph=6.2, temperature=26, humidity=78,
           rainfall=190, confidence=91.0)


# --------------------------------------------------------------------------- #
# Phone normalisation
# --------------------------------------------------------------------------- #
class TestNormalisePhone:
    @pytest.mark.parametrize("typed", [
        "9876543210", "+91 98765 43210", "098765-43210", "+919876543210",
        " 91 9876543210 ", "98765 43210",
    ])
    def test_every_spelling_of_one_number_is_the_same_account(self, typed):
        """A farmer retyping their number on a new phone must not get a
        second, empty account."""
        assert auth.normalise_phone(typed) == "9876543210"

    @pytest.mark.parametrize("bad", ["123", "", "abcdefghij", "1234567890",
                                     "5876543210"])
    def test_rejects_what_is_not_an_indian_mobile(self, bad):
        # Indian mobiles start 6-9; 1234567890 and 5... are not numbers.
        with pytest.raises(auth.AuthError):
            auth.normalise_phone(bad)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
class TestRegister:
    def test_creates_an_account(self, alice):
        assert alice.id > 0 and alice.phone == "9876543210"

    def test_pin_is_never_stored_in_the_clear(self, db, alice):
        conn = sqlite3.connect(db)
        stored = conn.execute(
            f"SELECT pin_hash FROM {dbm.USERS_TABLE} WHERE id=?;", (alice.id,)
        ).fetchone()[0]
        conn.close()
        assert "4729" not in stored
        assert stored.startswith("$2")          # a bcrypt hash

    def test_user_object_carries_no_hash(self, alice):
        assert not any("hash" in f for f in vars(alice))

    def test_duplicate_number_is_refused(self, db, alice):
        with pytest.raises(auth.AuthError, match="already has an account"):
            auth.register("+91 98765 43210", "1357", db_path=db)

    @pytest.mark.parametrize("weak", ["1234", "0000", "1111", "4321"])
    def test_obvious_pins_are_refused(self, db, weak):
        with pytest.raises(auth.WeakPin):
            auth.register("9000000001", weak, db_path=db)

    @pytest.mark.parametrize("bad", ["123", "12345", "", "abcd"])
    def test_pin_must_be_four_digits(self, db, bad):
        with pytest.raises(auth.WeakPin):
            auth.register("9000000001", bad, db_path=db)

    def test_masked_phone_hides_most_of_it(self, alice):
        assert alice.masked_phone == "•••••43210"


# --------------------------------------------------------------------------- #
# Sign-in and the lockout
# --------------------------------------------------------------------------- #
class TestSignIn:
    def test_correct_pin_signs_in(self, db, alice):
        assert auth.sign_in("9876543210", "4729", db_path=db).id == alice.id

    def test_wrong_pin_refused(self, db, alice):
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "1357", db_path=db)

    def test_unknown_number_and_wrong_pin_give_the_same_message(self, db, alice):
        """Otherwise the form is an oracle for which numbers are registered."""
        with pytest.raises(auth.AuthError) as unknown:
            auth.sign_in("9000000009", "4729", db_path=db)
        with pytest.raises(auth.AuthError) as wrong:
            auth.sign_in("9876543210", "1357", db_path=db)
        assert str(unknown.value) == str(wrong.value)

    def test_lockout_actually_engages(self, db, alice):
        """Regression test for a real defect.

        The failure counter was incremented inside the same write transaction
        that then raised the refusal -- and get_connection(write=True) rolls
        back on exception, so the increment was discarded every time. The
        counter never passed 1, the lockout never fired, and a 4-digit PIN was
        therefore brute-forceable at full speed.
        """
        for _ in range(accounts.MAX_FAILED_ATTEMPTS - 1):
            with pytest.raises(auth.AuthError):
                auth.sign_in("9876543210", "1357", db_path=db)
        with pytest.raises(auth.LockedOut):
            auth.sign_in("9876543210", "1357", db_path=db)

    def test_failure_counter_is_actually_persisted(self, db, alice):
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "1357", db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute(
            f"SELECT failed_count FROM {dbm.USERS_TABLE} WHERE id=?;",
            (alice.id,),
        ).fetchone()[0]
        conn.close()
        assert count == 1, "the increment was rolled back"

    def test_correct_pin_does_not_bypass_an_active_lock(self, db, alice):
        for _ in range(accounts.MAX_FAILED_ATTEMPTS):
            with pytest.raises(auth.AuthError):
                auth.sign_in("9876543210", "1357", db_path=db)
        with pytest.raises(auth.LockedOut):
            auth.sign_in("9876543210", "4729", db_path=db)

    def test_a_correct_pin_clears_the_counter(self, db, alice):
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "1357", db_path=db)
        auth.sign_in("9876543210", "4729", db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute(
            f"SELECT failed_count FROM {dbm.USERS_TABLE} WHERE id=?;",
            (alice.id,),
        ).fetchone()[0]
        conn.close()
        assert count == 0

    def test_lock_expires(self, db, alice):
        for _ in range(accounts.MAX_FAILED_ATTEMPTS):
            with pytest.raises(auth.AuthError):
                auth.sign_in("9876543210", "1357", db_path=db)
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(
            timespec="seconds")
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE {dbm.USERS_TABLE} SET locked_until=? WHERE id=?;",
                     (past, alice.id))
        conn.commit(); conn.close()
        assert auth.sign_in("9876543210", "4729", db_path=db).id == alice.id

    def test_corrupt_hash_reads_as_refusal_not_a_crash(self, db, alice):
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE {dbm.USERS_TABLE} SET pin_hash='garbage' "
                     f"WHERE id=?;", (alice.id,))
        conn.commit(); conn.close()
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "4729", db_path=db)


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
class TestSessions:
    def test_token_resolves_to_its_user(self, db, alice):
        token = auth.issue_token(alice.id, db_path=db)
        assert auth.user_for_token(token, db_path=db).id == alice.id

    def test_token_is_stored_hashed(self, db, alice):
        token = auth.issue_token(alice.id, db_path=db)
        conn = sqlite3.connect(db)
        stored = conn.execute(
            f"SELECT token_hash FROM {dbm.SESSIONS_TABLE};").fetchone()[0]
        conn.close()
        assert token not in stored

    def test_logout_revokes(self, db, alice):
        token = auth.issue_token(alice.id, db_path=db)
        auth.revoke_token(token, db_path=db)
        assert auth.user_for_token(token, db_path=db) is None

    def test_revoking_an_unknown_token_is_a_no_op(self, db):
        auth.revoke_token("not-a-real-token", db_path=db)   # must not raise

    def test_expired_token_is_refused(self, db, alice):
        token = auth.issue_token(alice.id, db_path=db)
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(
            timespec="seconds")
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE {dbm.SESSIONS_TABLE} SET expires_at=?;", (past,))
        conn.commit(); conn.close()
        assert auth.user_for_token(token, db_path=db) is None

    @pytest.mark.parametrize("junk", [None, "", "garbage", "a" * 64])
    def test_junk_tokens_resolve_to_nobody(self, db, junk):
        assert auth.user_for_token(junk, db_path=db) is None

    def test_tokens_are_unique(self, db, alice):
        issued = {auth.issue_token(alice.id, db_path=db) for _ in range(20)}
        assert len(issued) == 20

    def test_revoke_all_ends_every_session(self, db, alice):
        tokens = [auth.issue_token(alice.id, db_path=db) for _ in range(3)]
        auth.revoke_all(alice.id, db_path=db)
        assert all(auth.user_for_token(t, db_path=db) is None for t in tokens)


# --------------------------------------------------------------------------- #
# Ledger isolation — the rule that matters most
# --------------------------------------------------------------------------- #
class TestLedgerIsolation:
    @pytest.fixture
    def two_users(self, db):
        a, _ = auth.register("9876543210", "4729", db_path=db)
        b, _ = auth.register("9000000002", "8351", db_path=db)
        dbm.log_transaction(district="Udupi", recommended_crop="jute",
                            user_id=a.id, db_path=db, **LOG)
        dbm.log_transaction(district="Mysuru", recommended_crop="rice",
                            user_id=b.id, db_path=db, **LOG)
        dbm.log_transaction(district="Dharwad", recommended_crop="cotton",
                            db_path=db, **LOG)
        return a, b

    def test_a_user_sees_only_their_own_rows(self, db, two_users):
        a, b = two_users
        assert list(dbm.fetch_audit_history(user_id=a.id, db_path=db)
                    .recommended_crop) == ["jute"]
        assert list(dbm.fetch_audit_history(user_id=b.id, db_path=db)
                    .recommended_crop) == ["rice"]

    def test_guest_rows_are_not_visible_to_a_signed_in_user(self, db, two_users):
        a, _ = two_users
        assert "cotton" not in list(
            dbm.fetch_audit_history(user_id=a.id, db_path=db).recommended_crop)

    def test_guest_only_sees_guest_rows(self, db, two_users):
        assert list(dbm.fetch_audit_history(guest_only=True, db_path=db)
                    .recommended_crop) == ["cotton"]

    def test_unscoped_read_sees_everything(self, db, two_users):
        """The examiner ledger is deliberately unscoped."""
        assert len(dbm.fetch_audit_history(db_path=db)) == 3

    def test_a_guest_write_is_null_not_zero(self, db):
        dbm.log_transaction(district="Udupi", recommended_crop="jute",
                            db_path=db, **LOG)
        conn = sqlite3.connect(db)
        value = conn.execute(
            f"SELECT user_id FROM {dbm.TABLE_NAME};").fetchone()[0]
        conn.close()
        assert value is None


# --------------------------------------------------------------------------- #
# Profile, PIN change, deletion
# --------------------------------------------------------------------------- #
class TestProfile:
    def test_updates_only_what_is_given(self, db, alice):
        updated = auth.update_profile(alice.id, village="Brahmavar", db_path=db)
        assert updated.village == "Brahmavar"
        assert updated.district == "Udupi"      # untouched
        assert updated.display_name == "Alice"

    def test_greeting_falls_back_to_a_masked_number(self, db):
        anon, _ = auth.register("9000000003", "8351", db_path=db)
        assert anon.greeting == "•••••00003"


class TestChangePin:
    def test_requires_the_old_pin(self, db, alice):
        with pytest.raises(auth.AuthError):
            auth.change_pin(alice.id, "0000", "8351", db_path=db)

    def test_changes_it(self, db, alice):
        auth.change_pin(alice.id, "4729", "8351", db_path=db)
        assert auth.sign_in("9876543210", "8351", db_path=db).id == alice.id
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "4729", db_path=db)

    def test_new_pin_must_also_be_strong(self, db, alice):
        with pytest.raises(auth.WeakPin):
            auth.change_pin(alice.id, "4729", "1234", db_path=db)


class TestDeleteAccount:
    def test_removes_the_account_and_its_sessions(self, db, alice):
        token = auth.issue_token(alice.id, db_path=db)
        auth.delete_account(alice.id, db_path=db)
        assert auth.get_user(alice.id, db_path=db) is None
        assert auth.user_for_token(token, db_path=db) is None

    def test_by_default_the_ledger_survives_but_is_detached(self, db, alice):
        dbm.log_transaction(district="Udupi", recommended_crop="jute",
                            user_id=alice.id, db_path=db, **LOG)
        auth.delete_account(alice.id, db_path=db)
        rows = dbm.fetch_audit_history(db_path=db)
        assert len(rows) == 1, "the agronomic record should survive"
        assert list(dbm.fetch_audit_history(guest_only=True, db_path=db)
                    .recommended_crop) == ["jute"]

    def test_full_erasure_removes_the_rows_too(self, db, alice):
        dbm.log_transaction(district="Udupi", recommended_crop="jute",
                            user_id=alice.id, db_path=db, **LOG)
        auth.delete_account(alice.id, keep_ledger=False, db_path=db)
        assert len(dbm.fetch_audit_history(db_path=db)) == 0

    def test_the_number_can_register_again_afterwards(self, db, alice):
        auth.delete_account(alice.id, db_path=db)
        fresh, _ = auth.register("9876543210", "8351", db_path=db)
        assert fresh.id != alice.id


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #
class TestMigration:
    def test_a_v1_database_upgrades_without_losing_rows(self, tmp_path):
        path = tmp_path / "legacy.db"
        conn = sqlite3.connect(path)
        conn.executescript(f"""
        CREATE TABLE {dbm.TABLE_NAME} (
          id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
          district TEXT NOT NULL DEFAULT 'Unspecified',
          N REAL NOT NULL, P REAL NOT NULL, K REAL NOT NULL, pH REAL NOT NULL,
          temp REAL NOT NULL, humidity REAL NOT NULL, rainfall REAL NOT NULL,
          recommended_crop TEXT NOT NULL, confidence REAL NOT NULL,
          primary_shap_driver TEXT, jaccard_index REAL);
        PRAGMA user_version=1;
        """)
        conn.execute(
            f"INSERT INTO {dbm.TABLE_NAME} (timestamp,district,N,P,K,pH,temp,"
            f"humidity,rainfall,recommended_crop,confidence) VALUES "
            f"('2026-01-01T00:00:00','Udupi',60,40,45,6.2,26,78,190,'jute',91);")
        conn.commit(); conn.close()

        dbm.init_db(path)

        assert dbm.get_schema_version(path) == dbm.SCHEMA_VERSION
        rows = dbm.fetch_audit_history(db_path=path)
        assert len(rows) == 1
        assert rows.recommended_crop.iloc[0] == "jute"
        # Pre-account rows are guest rows; that is the correct reading.
        assert rows.user_id.isna().all()
        dbm.close_connections()

    def test_the_new_tables_exist_after_migration(self, db):
        conn = sqlite3.connect(db)
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';")}
        conn.close()
        assert {dbm.USERS_TABLE, dbm.SESSIONS_TABLE,
                dbm.PLOTS_TABLE} <= names


# --------------------------------------------------------------------------- #
# PIN recovery
# --------------------------------------------------------------------------- #
class TestRecovery:
    @pytest.fixture
    def account(self, db):
        user, code = auth.register("9876543210", "4729", db_path=db)
        return user, code

    def test_registration_returns_a_code(self, account):
        _, code = account
        assert len(code.replace("-", "")) == 8
        assert "-" in code

    def test_code_is_stored_hashed_not_in_the_clear(self, db, account):
        user, code = account
        conn = sqlite3.connect(db)
        stored = conn.execute(
            f"SELECT recovery_hash FROM {dbm.USERS_TABLE} WHERE id=?;",
            (user.id,)).fetchone()[0]
        conn.close()
        assert code.replace("-", "") not in stored
        assert stored.startswith("$2")

    def test_a_forgotten_pin_can_be_reset(self, db, account):
        user, code = account
        auth.reset_pin_with_code("9876543210", code, "8351", db_path=db)
        assert auth.sign_in("9876543210", "8351", db_path=db).id == user.id

    def test_the_old_pin_stops_working(self, db, account):
        _, code = account
        auth.reset_pin_with_code("9876543210", code, "8351", db_path=db)
        with pytest.raises(auth.AuthError):
            auth.sign_in("9876543210", "4729", db_path=db)

    @pytest.mark.parametrize("style", [
        str.lower, lambda c: c.replace("-", ""),
        lambda c: f"  {c}  ", lambda c: c.replace("-", " "),
    ])
    def test_code_is_accepted_however_it_was_written_down(self, db, account, style):
        _, code = account
        auth.reset_pin_with_code("9876543210", style(code), "8351", db_path=db)
        assert auth.sign_in("9876543210", "8351", db_path=db)

    def test_wrong_code_is_refused(self, db, account):
        with pytest.raises(auth.AuthError):
            auth.reset_pin_with_code("9876543210", "AAAA-BBBB", "8351",
                                     db_path=db)

    def test_recovery_is_rate_limited_too(self, db, account):
        """Otherwise it is a second, unguarded door into the account.

        An attacker would simply brute-force the recovery code instead of the
        PIN, and the PIN lockout would have bought nothing.
        """
        for _ in range(accounts.MAX_FAILED_ATTEMPTS - 1):
            with pytest.raises(auth.AuthError):
                auth.reset_pin_with_code("9876543210", "AAAA-BBBB", "8351",
                                         db_path=db)
        with pytest.raises(auth.LockedOut):
            auth.reset_pin_with_code("9876543210", "AAAA-BBBB", "8351",
                                     db_path=db)

    def test_a_used_code_is_burned(self, db, account):
        _, code = account
        auth.reset_pin_with_code("9876543210", code, "8351", db_path=db)
        with pytest.raises(auth.AuthError):
            auth.reset_pin_with_code("9876543210", code, "2468", db_path=db)

    def test_reset_issues_a_fresh_code(self, db, account):
        _, code = account
        _, new_code = auth.reset_pin_with_code("9876543210", code, "8351",
                                               db_path=db)
        assert new_code != code
        auth.reset_pin_with_code("9876543210", new_code, "2468", db_path=db)

    def test_reset_ends_every_existing_session(self, db, account):
        user, code = account
        token = auth.issue_token(user.id, db_path=db)
        auth.reset_pin_with_code("9876543210", code, "8351", db_path=db)
        assert auth.user_for_token(token, db_path=db) is None

    def test_new_pin_must_still_be_strong(self, db, account):
        _, code = account
        with pytest.raises(auth.WeakPin):
            auth.reset_pin_with_code("9876543210", code, "1234", db_path=db)

    def test_regenerating_requires_the_pin(self, db, account):
        user, _ = account
        with pytest.raises(auth.AuthError):
            auth.regenerate_recovery_code(user.id, "0000", db_path=db)
        fresh = auth.regenerate_recovery_code(user.id, "4729", db_path=db)
        assert len(fresh.replace("-", "")) == 8

    def test_regenerating_invalidates_the_old_code(self, db, account):
        user, code = account
        auth.regenerate_recovery_code(user.id, "4729", db_path=db)
        with pytest.raises(auth.AuthError):
            auth.reset_pin_with_code("9876543210", code, "8351", db_path=db)

    def test_codes_avoid_ambiguous_characters(self):
        """These get written on paper and read back by someone else."""
        codes = "".join(auth.generate_recovery_code() for _ in range(200))
        assert not (set("OIL01U") & set(codes))

    def test_a_v2_account_without_a_code_is_still_reportable_internally(
            self, db, account):
        """``has_recovery_code`` is how the app knows to offer one. It takes
        a user id, so only a signed-in user can ask it about themselves."""
        user, _ = account
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE {dbm.USERS_TABLE} SET recovery_hash=NULL "
                     f"WHERE id=?;", (user.id,))
        conn.commit(); conn.close()
        assert auth.has_recovery_code(user.id, db_path=db) is False

    def test_but_the_reset_form_does_not_say_so_out_loud(self, db, account):
        """This test used to assert the opposite -- that the refusal named
        the reason. It did, and in doing so it told anyone who asked that the
        number was registered, before any failure counter had been touched.
        The explanation now sits on the form, where everyone sees it."""
        user, _ = account
        conn = sqlite3.connect(db)
        conn.execute(f"UPDATE {dbm.USERS_TABLE} SET recovery_hash=NULL "
                     f"WHERE id=?;", (user.id,))
        conn.commit(); conn.close()
        with pytest.raises(auth.AuthError) as raised:
            auth.reset_pin_with_code("9876543210", "AAAA-BBBB", "8351",
                                     db_path=db)
        assert "recovery code" not in str(raised.value).replace(
            "Wrong number or recovery code.", "")


# --------------------------------------------------------------------------- #
# Saved plots
# --------------------------------------------------------------------------- #
FEATURES = {"N": 60.0, "P": 40.0, "K": 45.0, "ph": 6.2,
            "temperature": 26.0, "humidity": 78.0, "rainfall": 190.0}


class TestPlots:
    @pytest.fixture
    def alice_id(self, db):
        user, _ = auth.register("9876543210", "4729", db_path=db)
        return user.id

    def test_save_and_read_back(self, db, alice_id):
        saved = auth.save_plot(alice_id, "North field", "Udupi", 2.0,
                               FEATURES, db_path=db)
        assert saved.name == "North field"
        assert saved.readings["ph"] == 6.2
        assert saved.readings["temperature"] == 26.0

    def test_listed_for_its_owner(self, db, alice_id):
        auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES, db_path=db)
        auth.save_plot(alice_id, "South", "Udupi", 1.0, FEATURES, db_path=db)
        assert {p.name for p in auth.list_plots(alice_id, db_path=db)} == {
            "North", "South"}

    def test_one_user_cannot_read_anothers_plot(self, db, alice_id):
        other, _ = auth.register("9000000002", "8351", db_path=db)
        mine = auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES,
                              db_path=db)
        assert auth.get_plot(other.id, mine.id, db_path=db) is None
        assert auth.list_plots(other.id, db_path=db) == []

    def test_one_user_cannot_delete_anothers_plot(self, db, alice_id):
        other, _ = auth.register("9000000002", "8351", db_path=db)
        mine = auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES,
                              db_path=db)
        assert auth.delete_plot(other.id, mine.id, db_path=db) is False
        assert auth.get_plot(alice_id, mine.id, db_path=db) is not None

    def test_resaving_a_name_updates_rather_than_duplicates(self, db, alice_id):
        auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES, db_path=db)
        changed = {**FEATURES, "N": 99.0}
        auth.save_plot(alice_id, "North", "Udupi", 3.0, changed, db_path=db)
        plots = auth.list_plots(alice_id, db_path=db)
        assert len(plots) == 1
        assert plots[0].readings["N"] == 99.0
        assert plots[0].acres == 3.0

    def test_name_is_whitespace_normalised(self, db, alice_id):
        p = auth.save_plot(alice_id, "  North   field  ", "Udupi", 1.0,
                           FEATURES, db_path=db)
        assert p.name == "North field"

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_a_nameless_plot_is_refused(self, db, alice_id, bad):
        with pytest.raises(auth.PlotError):
            auth.save_plot(alice_id, bad, "Udupi", 1.0, FEATURES, db_path=db)

    def test_missing_readings_are_refused(self, db, alice_id):
        with pytest.raises(auth.PlotError, match="Missing readings"):
            auth.save_plot(alice_id, "North", "Udupi", 1.0, {"N": 1.0},
                           db_path=db)

    def test_zero_acres_is_refused(self, db, alice_id):
        with pytest.raises(auth.PlotError):
            auth.save_plot(alice_id, "North", "Udupi", 0.0, FEATURES,
                           db_path=db)

    def test_the_cap_is_enforced(self, db, alice_id):
        for i in range(auth.MAX_PLOTS_PER_USER):
            auth.save_plot(alice_id, f"Field {i}", "Udupi", 1.0, FEATURES,
                           db_path=db)
        with pytest.raises(auth.PlotError, match="Delete one"):
            auth.save_plot(alice_id, "One more", "Udupi", 1.0, FEATURES,
                           db_path=db)

    def test_deleting_an_account_removes_its_plots(self, db, alice_id):
        auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES, db_path=db)
        auth.delete_account(alice_id, db_path=db)
        assert auth.count_plots(alice_id, db_path=db) == 0

    def test_readings_round_trip_into_model_feature_names(self, db, alice_id):
        from src.core.config import FEATURE_NAMES
        saved = auth.save_plot(alice_id, "North", "Udupi", 2.0, FEATURES,
                               db_path=db)
        assert set(saved.readings) == set(FEATURE_NAMES)

    def test_label_reads_as_one_line(self, db, alice_id):
        p = auth.save_plot(alice_id, "North", "Udupi", 1.0, FEATURES,
                           db_path=db)
        assert p.label == "North · Udupi · 1 acre"


class TestAccountEnumeration:
    """Returning the same words for "no such number" and "wrong PIN" hides
    nothing if the two answers arrive at different speeds, or if some other
    branch answers the question outright."""

    def test_an_unknown_number_costs_what_a_known_one_costs(self, db):
        """bcrypt at cost 12 takes ~0.3s. Skipping it for an unknown number
        made the two paths differ by a factor of three thousand, which is
        readable over any network."""
        accounts.register("9876543210", "5837", db_path=db)

        def elapsed(phone: str) -> float:
            start = time.perf_counter()
            with pytest.raises(accounts.AuthError):
                accounts.sign_in(phone, "1357", db_path=db)
            return time.perf_counter() - start

        known = min(elapsed("9876543210") for _ in range(3))
        unknown = min(elapsed("9000000001") for _ in range(3))
        # Generous: the point is orders of magnitude, not milliseconds.
        assert 0.25 < unknown / known < 4.0, (
            f"unknown number took {unknown:.4f}s, known took {known:.4f}s"
        )

    def test_the_refusal_reads_the_same_either_way(self, db):
        accounts.register("9876543210", "5837", db_path=db)
        said = []
        for phone in ("9876543210", "9000000001"):
            with pytest.raises(accounts.AuthError) as raised:
                accounts.sign_in(phone, "1357", db_path=db)
            said.append(str(raised.value))
        assert said[0] == said[1]

    def test_recovery_says_the_same_for_a_legacy_account(self, db):
        """An account with no recovery code must refuse exactly as an unknown
        number does. A distinct message there confirmed the number existed,
        before any failure counter had been touched -- so it could be asked
        as often as you liked."""
        user, _ = accounts.register("9876543210", "5837", db_path=db)
        with dbm.get_connection(db, write=True) as conn:
            conn.execute(
                f"UPDATE {dbm.USERS_TABLE} SET recovery_hash = NULL "
                f"WHERE id = ?;", (user.id,))

        said = []
        for phone in ("9876543210", "9000000001"):
            with pytest.raises(accounts.AuthError) as raised:
                accounts.reset_pin_with_code(phone, "ABCD-EFGH", "5837",
                                             db_path=db)
            said.append(str(raised.value))
        assert said[0] == said[1]

    def test_recovery_on_an_unknown_number_also_pays_the_cost(self, db):
        start = time.perf_counter()
        with pytest.raises(accounts.AuthError):
            accounts.reset_pin_with_code("9000000001", "ABCD-EFGH", "5837",
                                         db_path=db)
        assert time.perf_counter() - start > 0.05


class TestCodeNormalisationIsBackwardCompatible:
    """The O->0 and I/L->1 mapping was dropped as a no-op. These pin that it
    really was one, so a hash written under the old code still verifies."""

    def test_a_generated_code_is_untouched_apart_from_its_dash(self):
        for _ in range(200):
            code = auth.generate_recovery_code()
            assert accounts._normalise_code(code) == code.replace("-", "")

    def test_case_and_punctuation_are_still_forgiven(self):
        assert (accounts._normalise_code(" abcd-efgh ")
                == accounts._normalise_code("ABCDEFGH"))

    def test_a_code_issued_before_the_change_still_works(self, db):
        user, code = auth.register("9876543210", "4729", db_path=db)
        restored, _ = auth.reset_pin_with_code("9876543210", code.lower(),
                                               "8351", db_path=db)
        assert restored.id == user.id

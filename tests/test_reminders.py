"""Field reminders computed from saved advice."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.utils.agronomy import roadmap
from src.utils.reminders import (
    HORIZON_DAYS,
    STALE_AFTER_DAYS,
    Reminder,
    reminders_for,
    summarise,
    upcoming,
)

TODAY = date(2026, 9, 14)


def saved(days_ago: int) -> str:
    return (TODAY - timedelta(days=days_ago)).isoformat()


class TestRemindersForOneRecord:
    def test_finds_the_stage_that_is_due(self):
        """Rice tops-dresses around day 34; advice saved 34 days ago is due."""
        found = reminders_for("rice", saved(34), today=TODAY)
        assert found and found[0].days_away == 0
        assert found[0].is_today

    def test_a_past_stage_reads_as_overdue(self):
        found = reminders_for("rice", saved(40), today=TODAY)
        assert found[0].is_overdue
        assert found[0].urgency == "overdue"

    def test_sowing_is_never_reminded(self):
        """By the time a record exists, day 0 has happened or the advice was
        not acted on. Either way it is not a useful thing to say."""
        found = reminders_for("rice", saved(1), today=TODAY)
        assert all(r.stage != roadmap("rice")[0].name for r in found)

    def test_nothing_beyond_the_horizon(self):
        found = reminders_for("rice", saved(0), today=TODAY)
        assert all(r.days_away <= HORIZON_DAYS for r in found)

    def test_long_past_stages_are_dropped_not_nagged(self):
        found = reminders_for("rice", saved(34 + STALE_AFTER_DAYS + 5),
                              today=TODAY)
        assert all(r.stage != "Early growth" for r in found)

    def test_sorted_soonest_first(self):
        found = reminders_for("rice", saved(30), today=TODAY)
        assert [r.days_away for r in found] == sorted(r.days_away for r in found)

    @pytest.mark.parametrize("bad", ["not a date", "", None, 12345])
    def test_an_unparseable_date_yields_nothing_rather_than_raising(self, bad):
        assert reminders_for("rice", bad, today=TODAY) == []

    def test_an_unknown_crop_yields_nothing(self):
        assert reminders_for("dragonfruit", saved(10), today=TODAY) == []

    def test_accepts_a_date_object_as_well_as_a_string(self):
        as_text = reminders_for("rice", saved(34), today=TODAY)
        as_date = reminders_for("rice", TODAY - timedelta(days=34), today=TODAY)
        assert [r.stage for r in as_text] == [r.stage for r in as_date]

    def test_every_reminder_is_flagged_as_an_estimate(self):
        """The anchor is when advice was saved, not when the crop went in."""
        found = reminders_for("rice", saved(34), today=TODAY)
        assert all(r.is_estimate for r in found)

    def test_district_is_carried_through(self):
        found = reminders_for("rice", saved(34), district="Udupi", today=TODAY)
        assert found[0].district == "Udupi"


class TestWording:
    @pytest.mark.parametrize("days_ago,expected", [
        (34, "today"),
        (35, "about 1 day ago"),
        (40, "about 6 days ago"),
    ])
    def test_past_and_present_read_naturally(self, days_ago, expected):
        found = reminders_for("rice", saved(days_ago), today=TODAY)
        assert found[0].when_words() == expected

    def test_future_reads_naturally(self):
        found = reminders_for("rice", saved(33), today=TODAY)
        assert found[0].when_words() == "in about 1 day"

    def test_singular_and_plural_are_both_right(self):
        one = reminders_for("rice", saved(33), today=TODAY)[0]
        many = reminders_for("rice", saved(30), today=TODAY)[0]
        assert "1 day" in one.when_words() and "days" not in one.when_words()
        assert "days" in many.when_words()

    def test_headline_names_the_crop_and_stage(self):
        found = reminders_for("rice", saved(34), today=TODAY)[0]
        assert found.headline() == "Rice — Early growth"

    @pytest.mark.parametrize("days_ago,urgency", [
        (40, "overdue"), (34, "now"), (30, "soon"), (10, "later"),
    ])
    def test_urgency_bands(self, days_ago, urgency):
        found = reminders_for("rice", saved(days_ago), today=TODAY)
        assert found[0].urgency == urgency


class TestUpcomingAcrossRecords:
    ROWS = [
        {"recommended_crop": "rice", "timestamp": saved(40), "district": "Udupi"},
        {"recommended_crop": "maize", "timestamp": saved(25), "district": "Mysuru"},
    ]

    def test_gathers_from_every_record(self):
        crops = {r.crop for r in upcoming(self.ROWS, today=TODAY)}
        assert crops == {"rice", "maize"}

    def test_most_urgent_first(self):
        found = upcoming(self.ROWS, today=TODAY)
        assert found[0].crop == "rice"          # overdue beats upcoming
        assert found[0].is_overdue

    def test_a_repeated_crop_and_stage_appears_once(self):
        """Saving the same advice twice must not double the reminder."""
        rows = self.ROWS + [dict(self.ROWS[0])]
        found = upcoming(rows, today=TODAY)
        keys = [(r.crop, r.stage) for r in found]
        assert len(keys) == len(set(keys))

    def test_respects_the_limit(self):
        assert len(upcoming(self.ROWS * 5, today=TODAY, limit=1)) == 1

    def test_empty_input_is_empty_output(self):
        assert upcoming([], today=TODAY) == []

    def test_rows_missing_fields_are_skipped_not_fatal(self):
        assert upcoming([{}, {"recommended_crop": "rice"}], today=TODAY) == []


class TestSummarise:
    def test_says_nothing_when_there_is_nothing(self):
        assert summarise([]) == ""

    def test_leads_with_overdue_work(self):
        found = upcoming(
            [{"recommended_crop": "rice", "timestamp": saved(40)}], today=TODAY)
        assert summarise(found) == "1 thing to do now"

    def test_otherwise_names_the_next_thing(self):
        found = upcoming(
            [{"recommended_crop": "rice", "timestamp": saved(30)}], today=TODAY)
        assert summarise(found).startswith("Next: Rice")

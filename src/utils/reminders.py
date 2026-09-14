"""What a farmer needs to do next, worked out from advice they already saved.

Push notifications need a scheduler, and the Streamlit deployment has none.
But the *value* of a notification -- "your crop is 34 days old, it is time for
the first top dressing" -- does not actually need one. Every ingredient is
already stored: a saved recommendation carries its crop and the date it was
saved, and :func:`src.utils.agronomy.roadmap` gives the day offset of each
stage. The due dates follow by arithmetic.

So this computes them on the spot and the dashboard shows them when the
farmer next opens the app. That is a reminder they will actually see, rather
than a notification the hosting cannot send.

What it deliberately does not do
--------------------------------
It does not claim a crop was sown. The saved date is when the *advice* was
saved, which is usually within a few days of sowing but is not the same
thing. Every date is therefore presented as "about", and
:attr:`Reminder.is_estimate` is true for all of them, so the UI can say so
rather than implying a precision the data does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, List, Optional, Sequence

from src.utils.agronomy import RoadmapStage, profile, roadmap

#: A stage more than this many days past is treated as gone, not overdue.
#: Nagging somebody about a top dressing from two months ago is noise.
STALE_AFTER_DAYS: int = 21

#: Only look this far ahead. Beyond it, nothing is actionable yet.
HORIZON_DAYS: int = 30


@dataclass(frozen=True)
class Reminder:
    """One field action, with when it falls."""

    crop: str
    district: str
    stage: str
    action: str
    due_on: date
    days_away: int
    saved_on: date

    #: Always true. The anchor is when advice was saved, not when the crop
    #: went in, so every date here is approximate by construction.
    is_estimate: bool = True

    @property
    def is_overdue(self) -> bool:
        return self.days_away < 0

    @property
    def is_today(self) -> bool:
        return self.days_away == 0

    @property
    def urgency(self) -> str:
        """``overdue`` / ``now`` / ``soon`` / ``later``."""
        if self.days_away < 0:
            return "overdue"
        if self.days_away == 0:
            return "now"
        if self.days_away <= 7:
            return "soon"
        return "later"

    def when_words(self) -> str:
        """``in about 5 days`` / ``about 3 days ago`` / ``today``."""
        if self.days_away == 0:
            return "today"
        if self.days_away < 0:
            days = abs(self.days_away)
            return f"about {days} day{'' if days == 1 else 's'} ago"
        return f"in about {self.days_away} day{'' if self.days_away == 1 else 's'}"

    def headline(self) -> str:
        return f"{self.crop.title()} — {self.stage}"


def _as_date(value: object) -> Optional[date]:
    """Parse whatever the ledger holds into a date, or ``None``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)[:19]).date()
    except (TypeError, ValueError):
        return None


def reminders_for(
    crop: str,
    saved_on: object,
    *,
    district: str = "",
    today: Optional[date] = None,
    horizon_days: int = HORIZON_DAYS,
    stale_after_days: int = STALE_AFTER_DAYS,
) -> List[Reminder]:
    """Actions falling due for one saved recommendation.

    Returns them soonest first, overdue included, with anything long past or
    far in the future dropped. The sowing stage at day 0 is skipped: by the
    time there is a saved record, it has happened or the advice was not acted
    on, and either way telling someone to sow a crop they saved two weeks ago
    is not useful.
    """
    anchor = _as_date(saved_on)
    if anchor is None or profile(crop) is None:
        return []
    current = today or date.today()

    found: List[Reminder] = []
    for stage in roadmap(crop):
        if stage.day <= 0:
            continue
        due = anchor + timedelta(days=int(stage.day))
        away = (due - current).days
        if away < -stale_after_days or away > horizon_days:
            continue
        found.append(
            Reminder(
                crop=str(crop),
                district=str(district),
                stage=stage.name,
                action=stage.action,
                due_on=due,
                days_away=away,
                saved_on=anchor,
            )
        )
    return sorted(found, key=lambda r: r.days_away)


def upcoming(
    records: Iterable[dict],
    *,
    today: Optional[date] = None,
    limit: int = 5,
) -> List[Reminder]:
    """Everything due across a farmer's saved records, soonest first.

    ``records`` are mappings with at least ``recommended_crop`` and
    ``timestamp`` -- the shape the audit ledger already returns.
    """
    out: List[Reminder] = []
    for row in records:
        out.extend(
            reminders_for(
                str(row.get("recommended_crop", "")),
                row.get("timestamp"),
                district=str(row.get("district", "")),
                today=today,
            )
        )
    # Overdue first, then soonest. Two records of the same crop can produce
    # the same stage twice; keep the more urgent one.
    seen = set()
    unique: List[Reminder] = []
    for reminder in sorted(out, key=lambda r: r.days_away):
        key = (reminder.crop, reminder.stage)
        if key in seen:
            continue
        seen.add(key)
        unique.append(reminder)
    return unique[:limit]


def summarise(found: Sequence[Reminder]) -> str:
    """One line for a badge, or an empty string when there is nothing due."""
    if not found:
        return ""
    overdue = [r for r in found if r.is_overdue]
    if overdue:
        count = len(overdue)
        return f"{count} thing{'' if count == 1 else 's'} to do now"
    soonest = found[0]
    return f"Next: {soonest.headline()} {soonest.when_words()}"


__all__ = [
    "Reminder", "reminders_for", "upcoming", "summarise",
    "STALE_AFTER_DAYS", "HORIZON_DAYS",
]

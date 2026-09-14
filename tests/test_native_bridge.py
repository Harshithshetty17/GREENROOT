"""The Android handoff.

The Kotlin half cannot be exercised from here, so the contract these tests
pin is the payload and the injected script: shape, escaping, and the rule
that a browser session is never affected.
"""

from __future__ import annotations

import json
import re

import pytest

from src.core import native


def _bridge_argument(script: str) -> dict:
    """Pull the argument back out of the emitted script, as the app would."""
    match = re.search(r"bridge\.saveCard\((.*)\);", script)
    assert match, "no saveCard call in the script"
    # JavaScript reads "<\/" as "</"; undo that before parsing.
    outer = match.group(1).replace("<\\/", "</")
    return json.loads(json.loads(outer))


CARD = dict(
    card_id="12",
    crop="Jute",
    confidence=91.37,
    district="Udupi",
    readings="N 60 · P 40",
    advice=["Add urea", "Do not add potash"],
)


class TestBuildCard:
    def test_carries_every_field_the_kotlin_side_reads(self):
        card = native.build_card(**CARD)
        # Mirrors SavedCard.fromJson.
        assert set(card) == {
            "id", "crop", "confidence", "district", "readings", "advice"
        }

    def test_confidence_is_a_number_not_a_string(self):
        # optDouble on a string returns the fallback, silently zeroing it.
        assert isinstance(native.build_card(**CARD)["confidence"], float)

    def test_advice_is_capped(self):
        card = native.build_card(**{**CARD, "advice": [f"line {i}" for i in range(20)]})
        assert len(card["advice"]) == 6

    def test_cap_keeps_the_first_items(self):
        # advice_lines sorts worst-first, so the cap must not drop the
        # critical ones.
        card = native.build_card(**{**CARD, "advice": list("abcdefgh")})
        assert card["advice"] == list("abcdef")

    def test_id_is_a_string(self):
        # optString on a number returns "" on some Android versions.
        assert native.build_card(**{**CARD, "card_id": 12})["id"] == "12"


class TestSaveScript:
    def test_round_trips(self):
        assert _bridge_argument(native.save_script(native.build_card(**CARD))) == \
            native.build_card(**CARD)

    @pytest.mark.parametrize(
        "hostile",
        [
            "</script><img src=x onerror=alert(1)>",
            '"; alert(1); //',
            "O'Brien",
            "back\\slash",
            "ಉಡುಪಿ",
            "line\nbreak",
        ],
    )
    def test_hostile_district_survives_intact(self, hostile):
        script = native.save_script(native.build_card(**{**CARD, "district": hostile}))
        # The script block must not be closed early by the payload.
        assert script.count("</script>") == 1
        assert _bridge_argument(script)["district"] == hostile

    def test_guards_on_a_missing_bridge(self):
        # In a browser the name is absent; the script must return, not throw.
        script = native.save_script(native.build_card(**CARD))
        assert "typeof bridge.saveCard !== 'function'" in script
        assert "return" in script

    def test_names_the_bridge_the_kotlin_side_binds(self):
        assert native.BRIDGE == "GreenRootNative"
        assert native.BRIDGE in native.save_script(native.build_card(**CARD))


class TestPushCard:
    def test_returns_false_rather_than_raising_without_streamlit(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def refuse(name, *a, **kw):
            if name.startswith("streamlit"):
                raise ImportError("no streamlit here")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert native.push_card(None, native.build_card(**CARD)) is False


class TestReadingsSummary:
    def test_matches_the_order_the_hero_shows(self):
        line = native.readings_summary(
            {"N": 60, "P": 40, "K": 45, "ph": 6.2, "temperature": 26.4,
             "humidity": 80, "rainfall": 190.6}
        )
        assert line == "N 60 · P 40 · K 45 · pH 6.2 · 26°C · 191 mm"


class TestAdviceLines:
    def test_orders_worst_first(self):
        class Item:
            def __init__(self, severity, text):
                self.severity = severity
                self._text = text

            def say(self, simple):
                return self._text

        class Advisory:
            items = [Item("info", "i"), Item("critical", "c"), Item("warning", "w")]

        assert native.advice_lines(Advisory(), simple=True) == ["c", "w", "i"]

"""The Kannada interface layer."""

from __future__ import annotations

from datetime import date

import pytest

from src.utils import i18n
from src.utils.agronomy import roadmap
from src.utils.reminders import reminders_for
from src.utils.plain_language import COPY, text


class TestLanguageCodes:
    def test_both_languages_are_offered(self):
        assert dict(i18n.available_languages()) == {"en": "English", "kn": "ಕನ್ನಡ"}

    def test_each_language_names_itself_in_its_own_script(self):
        """The only form a reader who cannot read the other one recognises."""
        assert any("ಀ" <= ch <= "೿" for ch in i18n.LANGUAGES["kn"])

    @pytest.mark.parametrize("junk", [None, "", "fr", "KN ", "xx"])
    def test_anything_unknown_resolves_to_english(self, junk):
        assert i18n.normalise(junk) == "en"

    def test_known_codes_survive(self):
        assert i18n.normalise("kn") == "kn"
        assert i18n.normalise("en") == "en"


class TestTranslate:
    def test_returns_kannada_when_there_is_some(self):
        out = i18n.translate("run", "Show me the best crop", "kn")
        assert out != "Show me the best crop"
        assert any("ಀ" <= ch <= "೿" for ch in out)

    def test_falls_back_to_english_rather_than_blanking(self):
        """A partial translation is useful; a hole in the UI is not."""
        assert i18n.translate("no_such_key", "Fallback", "kn") == "Fallback"

    def test_english_is_returned_untouched(self):
        assert i18n.translate("run", "Show me", "en") == "Show me"

    def test_technical_register_stays_english_by_design(self):
        """Posterior probability and Jaccard index are English terminology."""
        assert i18n.translate(
            "run", "Generate recommendation", "kn", simple=False
        ) == "Generate recommendation"

    def test_an_empty_translation_does_not_win_over_english(self):
        """`or` rather than a membership test, so a blank never renders."""
        i18n.KANNADA_EXTRA["_blank_probe"] = ""
        try:
            assert i18n.translate("_blank_probe", "English", "kn") == "English"
        finally:
            del i18n.KANNADA_EXTRA["_blank_probe"]


class TestCatalogue:
    @pytest.mark.parametrize("key", sorted(i18n.KANNADA_UI))
    def test_every_key_maps_onto_a_real_copy_key(self, key):
        """A translation for a key that no longer exists is dead weight and
        hides the fact that its screen is untranslated."""
        assert key in COPY

    @pytest.mark.parametrize(
        "key,value", sorted(i18n.KANNADA_UI.items()) + sorted(
            i18n.KANNADA_EXTRA.items())
    )
    def test_every_value_actually_contains_kannada(self, key, value):
        """Guards against an English string left in a Kannada slot."""
        assert any("ಀ" <= ch <= "೿" for ch in value), f"{key} is not Kannada"

    @pytest.mark.parametrize("key", sorted(i18n.KANNADA_UI))
    def test_emoji_prefixes_are_preserved(self, key):
        """Tab labels lead with an emoji; losing it in translation would
        change the layout and the visual anchor a low-literacy user uses."""
        english = COPY[key][0]
        leading = english.split(" ")[0]
        if leading and not leading.isascii():
            assert i18n.KANNADA_UI[key].startswith(leading), key

    def test_the_lines_that_matter_most_are_translated(self):
        """A farmer must get the disclaimer and the warnings in their own
        language, whatever else is still English."""
        for key in ("disclaimer", "low_confidence", "unusual_input",
                    "season_clash", "run", "primary_label"):
            assert key in i18n.KANNADA_UI, f"{key} must be translated"

    def test_coverage_is_reported_honestly(self):
        numbers = i18n.coverage()
        assert numbers["copy_keys"] == len(COPY)
        assert numbers["copy_translated"] == sum(
            1 for k in COPY if k in i18n.KANNADA_UI)
        assert numbers["copy_translated"] <= numbers["copy_keys"]


class TestThroughTextApi:
    """The seam the whole dashboard actually calls."""

    def test_default_is_english(self):
        assert text("run", True) == COPY["run"][0]

    def test_kannada_flows_through(self):
        assert text("run", True, "kn") == i18n.KANNADA_UI["run"]

    def test_technical_register_ignores_language(self):
        assert text("run", False, "kn") == COPY["run"][1]

    def test_unknown_key_still_returns_the_key(self):
        assert text("nope", True, "kn") == "nope"

    @pytest.mark.parametrize("key", sorted(COPY))
    def test_no_key_ever_renders_empty_in_kannada(self, key):
        assert text(key, True, "kn").strip()


class TestReviewHonesty:
    def test_the_warning_exists_in_both_languages(self):
        assert i18n.review_note("en") == i18n.REVIEW_STATUS
        assert i18n.review_note("kn") == i18n.REVIEW_STATUS_KN

    def test_the_kannada_warning_is_in_kannada(self):
        """Told to the person who chose Kannada, so it must be readable
        by them."""
        assert any("ಀ" <= ch <= "೿" for ch in i18n.REVIEW_STATUS_KN)

    def test_it_says_the_translation_is_unchecked(self):
        assert "not been checked" in i18n.REVIEW_STATUS


# --------------------------------------------------------------------------- #
# Season sentences
# --------------------------------------------------------------------------- #
from src.utils import seasons  # noqa: E402


class TestSeasonMessages:
    @pytest.mark.parametrize("crop,season", [
        ("rice", "rabi"), ("rice", "kharif"), ("coconut", "rabi"),
    ])
    def test_all_three_shapes_render_in_kannada(self, crop, season):
        fit = seasons.assess(crop, season)
        out = fit.message(True, "kn")
        assert any("ಀ" <= ch <= "೿" for ch in out)
        assert out != fit.message(True, "en")

    def test_the_crop_is_named_in_kannada_not_english(self, ):
        """A Kannada sentence with an English crop name in the middle reads
        as broken."""
        out = seasons.assess("rice", "rabi").message(True, "kn")
        assert "ಭತ್ತ" in out
        assert "Rice" not in out

    def test_the_season_is_named_in_kannada(self):
        out = seasons.assess("rice", "rabi").message(True, "kn")
        assert "ಹಿಂಗಾರು" in out          # Rabi
        assert "Rabi" not in out

    def test_no_leftover_template_placeholders(self):
        for crop, season in [("rice", "rabi"), ("rice", "kharif"),
                             ("coconut", "rabi")]:
            out = seasons.assess(crop, season).message(True, "kn")
            assert "{" not in out and "}" not in out

    def test_english_is_unchanged(self):
        fit = seasons.assess("rice", "rabi")
        assert fit.message(True) == fit.message(True, "en")

    def test_technical_register_stays_english(self):
        fit = seasons.assess("rice", "rabi")
        assert fit.message(False, "kn") == fit.message(False, "en")

    def test_multiple_seasons_are_joined_in_kannada(self):
        """English joins with 'or'; Kannada must not."""
        out = seasons.assess("rice", "kharif").sowable_labels_in("kn")
        assert " or " not in out
        assert "ಅಥವಾ" in out

    @pytest.mark.parametrize("crop", sorted(seasons.CROP_SEASONS))
    def test_every_crop_produces_a_kannada_sentence(self, crop):
        out = seasons.assess(crop, "kharif").message(True, "kn")
        assert out.strip() and "{" not in out


class TestFieldCalendarInKannada:
    """The reminder panel is the first thing a returning farmer reads, and
    until now every word of it was English on a Kannada screen."""

    def test_stage_name_and_action_both_translate(self):
        name, action = i18n.stage_words("Early growth",
                                        "First top dressing: one third...",
                                        i18n.KANNADA)
        assert name == "ಆರಂಭಿಕ ಬೆಳವಣಿಗೆ"
        assert "ಯೂರಿಯಾ" in action

    def test_english_is_returned_unchanged(self):
        pair = ("Flowering", "Last third of the urea.")
        assert i18n.stage_words(*pair, i18n.ENGLISH) == pair

    def test_every_roadmap_stage_has_a_translation(self):
        """Both roadmaps -- annual and perennial -- end to end, so a stage
        added to agronomy.py without a translation fails here rather than
        showing up as English on a farmer's screen."""
        for crop in ("rice", "coconut"):
            for stage in roadmap(crop):
                assert stage.name in i18n.KANNADA_STAGES, stage.name
                assert stage.name in i18n.KANNADA_STAGE_ACTIONS, stage.name

    def test_the_harvest_line_carries_the_day_count(self):
        _, action = i18n.stage_words("Harvest", "Ready around day 120.",
                                     i18n.KANNADA, day=120)
        assert "120" in action
        assert "{days}" not in action

    def test_an_unknown_stage_falls_back_rather_than_raising(self):
        assert i18n.stage_words("Ratooning", "Cut low.", i18n.KANNADA) == (
            "Ratooning", "Cut low.")


class TestWhenWords:
    @pytest.mark.parametrize("days,fragment", [
        (0, "ಇಂದು"), (1, "ದಿನದಲ್ಲಿ"), (5, "ದಿನಗಳಲ್ಲಿ"),
        (-1, "ದಿನದ ಹಿಂದೆ"), (-6, "ದಿನಗಳ ಹಿಂದೆ"),
    ])
    def test_singular_plural_past_and_future(self, days, fragment):
        assert fragment in i18n.when_words(days, "ignored", i18n.KANNADA)

    def test_the_number_survives(self):
        assert "5" in i18n.when_words(5, "in about 5 days", i18n.KANNADA)

    def test_english_passes_straight_through(self):
        assert i18n.when_words(5, "in about 5 days") == "in about 5 days"

    def test_matches_what_the_reminder_itself_says_in_english(self):
        """The English is passed in rather than rebuilt, so the two wordings
        cannot drift apart."""
        found = reminders_for("rice", "2026-08-11", today=date(2026, 9, 14))
        assert found
        for item in found:
            assert i18n.when_words(item.days_away, item.when_words()) == \
                item.when_words()


class TestInterpolatedPhrases:
    def test_a_district_and_a_count_are_filled_in(self):
        said = i18n.phrase("soil_from_survey", "english", i18n.KANNADA,
                           district="Udupi", count="1,204")
        assert "Udupi" in said and "1,204" in said and "english" not in said

    def test_english_is_untouched(self):
        assert i18n.phrase("soil_from_survey", "english", i18n.ENGLISH,
                           district="Udupi", count="1") == "english"

    def test_an_unknown_key_falls_back_instead_of_raising(self):
        assert i18n.phrase("no_such_key", "english", i18n.KANNADA,
                           district="Udupi") == "english"

    def test_every_phrase_accepts_the_district_the_app_passes(self):
        for key in i18n.KANNADA_PHRASES:
            assert i18n.phrase(key, "", i18n.KANNADA,
                               district="Udupi", count="10")


class TestNothingIsTranslatedIntoTheVoid:
    """A translated string nothing reads is worse than a missing one: it
    reads as covered, it costs a reviewer time in the review sheet, and the
    screen it was written for is still in English. Eight keys had drifted
    into that state before this test existed."""

    @staticmethod
    def _keys_app_reads() -> set:
        import ast
        import pathlib

        source = pathlib.Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        used = set()
        for node in ast.walk(tree):
            # t_extra("key", "English fallback")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "t_extra" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                used.add(node.args[0].value)
            # i18n.KANNADA_EXTRA["key"]
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and ast.unparse(node.value).endswith("KANNADA_EXTRA")):
                used.add(node.slice.value)
        return used

    def test_every_kannada_extra_is_reachable_from_the_app(self):
        unused = set(i18n.KANNADA_EXTRA) - self._keys_app_reads()
        assert not unused, (
            "translated but never shown: " + ", ".join(sorted(unused))
        )

    def test_the_review_sheet_can_show_an_english_line_for_every_key(self):
        """A reviewer given a Kannada string and nothing to compare it
        against cannot review it. Every extra key must resolve to English
        somewhere -- a t_extra fallback, or the named exceptions."""
        from export_translations import DIRECT_ENGLISH, english_fallbacks

        known = set(english_fallbacks()) | set(DIRECT_ENGLISH)
        assert set(i18n.KANNADA_EXTRA) <= known, (
            "no English to review against: "
            + ", ".join(sorted(set(i18n.KANNADA_EXTRA) - known))
        )

    def test_every_key_the_app_asks_for_has_a_translation(self):
        """The other direction: a call site with no entry silently renders
        its English fallback on a Kannada screen."""
        missing = self._keys_app_reads() - set(i18n.KANNADA_EXTRA)
        assert not missing, (
            "asked for but untranslated: " + ", ".join(sorted(missing))
        )


class TestShortDate:
    """Dates a farmer reads: on the saved-records table and beside a saved
    reading in the reload picker."""

    def test_english_is_unchanged(self):
        assert i18n.short_date("2026-09-14T10:00:00") == "14 Sep 2026"

    def test_kannada_names_the_month(self):
        assert i18n.short_date("2026-09-14", i18n.KANNADA) == "14 ಸೆಪ್ಟೆಂ 2026"

    def test_the_digits_stay_western(self):
        """They are read off a phone keypad and sit beside figures that are
        Western everywhere else in the app."""
        said = i18n.short_date("2026-09-14", i18n.KANNADA)
        assert "14" in said and "2026" in said

    def test_every_month_has_a_name(self):
        for month in range(1, 13):
            said = i18n.short_date(date(2026, month, 1), i18n.KANNADA)
            assert said and not any(c.isascii() and c.isalpha() for c in said)

    @pytest.mark.parametrize("bad", [None, "", "not a date", 12345, object()])
    def test_unparseable_input_is_none_rather_than_a_crash(self, bad):
        assert i18n.short_date(bad, i18n.KANNADA) is None

    def test_accepts_a_date_as_well_as_a_string(self):
        assert (i18n.short_date(date(2026, 9, 14), i18n.KANNADA)
                == i18n.short_date("2026-09-14", i18n.KANNADA))

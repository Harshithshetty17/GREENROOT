"""The Kannada interface layer."""

from __future__ import annotations

import pytest

from src.utils import i18n
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

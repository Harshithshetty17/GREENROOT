#!/usr/bin/env python3
"""Export every Kannada string next to its English, for review.

The translations in :mod:`src.utils.i18n` were written by someone who does
not speak Kannada, and the app says so. Telling a project "get a native
speaker to check this" is easy advice and hard to act on if the strings are
scattered through Python source -- so this turns the whole surface into one
file a Kannada speaker can read through in a sitting and mark up.

Two formats:

* ``--format md`` (default) -- a table to read on a phone or print.
* ``--format csv`` -- a column to type corrections into and hand back.

Nothing here reads or writes the model artifacts.

Usage
-----
    python export_translations.py                     # to stdout, Markdown
    python export_translations.py -o review.md
    python export_translations.py --format csv -o review.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"
sys.path.insert(0, str(ROOT))

from src.utils import i18n  # noqa: E402
from src.utils.agronomy import CROPS, SURVEY_CORROBORATED, roadmap  # noqa: E402
from src.utils.plain_language import COPY  # noqa: E402


class Row(NamedTuple):
    """One string to check."""

    group: str
    key: str
    english: str
    kannada: str
    #: Where the reader will see it, so they can judge tone and length.
    where: str
    #: True when something other than the author's judgement backs it.
    evidenced: bool = False


def _english_action(stage_name: str) -> Tuple[str, int]:
    """The English an action was translated from, and the day it falls on.

    Read back out of :func:`~src.utils.agronomy.roadmap` rather than copied,
    so the sheet cannot show a reviewer an English line the app no longer
    says. ``rice`` covers the annual stages and ``coconut`` the perennial
    ones; between them they produce all eight.

    The day comes back too because the harvest line interpolates it. Filling
    the Kannada with the same number the English shows means a reviewer
    compares two finished sentences rather than a sentence against a
    ``{days}`` placeholder.
    """
    for crop in ("rice", "coconut"):
        for stage in roadmap(crop):
            if stage.name == stage_name:
                return stage.action, stage.day
    return "", 0


#: Three keys the app reads by subscript rather than through ``t_extra`` --
#: the unit words, which are chosen by a helper, and the picker label, which
#: is deliberately bilingual. They carry no fallback argument to read back,
#: so their English is named here.
DIRECT_ENGLISH: Dict[str, str] = {
    "acre": "acre / acres",
    "bags": "bag / bags",
    "language": "Language",
}


def english_fallbacks(source: Path | None = None) -> Dict[str, str]:
    """``key -> the English the app would have shown`` for the extra strings.

    :data:`~src.utils.i18n.KANNADA_EXTRA` holds Kannada only: the English for
    those keys lives at the call site, as the fallback argument to
    ``t_extra(key, english)``. Without it the review sheet shows a reviewer
    47 Kannada lines and nothing to check them against.

    So it is read back out of ``app.py`` rather than kept as a second
    catalogue here -- the English in the sheet is then literally the English
    the app would render, and cannot drift from it.
    """
    tree = ast.parse((source or APP).read_text(encoding="utf-8"))
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "t_extra" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[1], ast.Constant)):
            # First call site wins; a key used twice with different wording
            # would be a bug in app.py, not something to paper over here.
            found.setdefault(node.args[0].value, node.args[1].value)
    return found


def _english_band_details() -> Dict[str, str]:
    """``band label -> the sentence under the crop card``.

    Swept out of :func:`~src.utils.plain_language.confidence_band` at scores
    that select each band, rather than copied, so the sheet cannot show a
    reviewer a sentence the app has stopped saying.
    """
    from src.utils.plain_language import confidence_band

    return {band.label: band.detail
            for band in (confidence_band(score) for score in (95, 70, 45, 20))}


def phrase_fallbacks(source: Path | None = None) -> Dict[str, str]:
    """``key -> English`` for the interpolated sentences.

    Same idea as :func:`english_fallbacks`, for ``i18n.phrase(key, english,
    ...)``. The English there is built with an f-string, so the literal parts
    are joined and the interpolations become the placeholder names the
    Kannada uses -- which is what a reviewer needs to see anyway.
    """
    tree = ast.parse((source or APP).read_text(encoding="utf-8"))
    found: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "phrase" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)):
            continue
        english = node.args[1]
        if isinstance(english, ast.Constant):
            found.setdefault(node.args[0].value, str(english.value))
        elif isinstance(english, ast.JoinedStr):
            found.setdefault(node.args[0].value, "".join(
                part.value if isinstance(part, ast.Constant)
                else "{" + ast.unparse(part.value).split(".")[-1] + "}"
                for part in english.values))
    return found


def collect() -> List[Row]:
    """Every translated string in the app, grouped for a human reader."""
    rows: List[Row] = []

    for key, kannada in sorted(i18n.KANNADA_UI.items()):
        english = COPY[key][0] if key in COPY else ""
        rows.append(Row("Interface", key, english, kannada,
                        "buttons, headings and labels"))

    fallbacks = {**DIRECT_ENGLISH, **english_fallbacks()}
    for key, kannada in sorted(i18n.KANNADA_EXTRA.items()):
        rows.append(Row("Interface (extra)", key, fallbacks.get(key, ""),
                        kannada, "controls and section titles"))

    details = _english_band_details()
    for english, (label_kn, detail_kn) in i18n.KANNADA_BANDS.items():
        rows.append(Row("Confidence", f"band:{english}", english, label_kn,
                        "beside the crop name -- the headline judgement"))
        rows.append(Row("Confidence", f"band-detail:{english}",
                        details.get(english, ""), detail_kn,
                        "the sentence under the crop card"))

    # Field instructions first among the untranslated-by-evidence groups:
    # these are the lines somebody acts on with a bag of urea in their hand,
    # so they are the ones worth the reviewer's freshest attention.
    for stage_name in i18n.KANNADA_STAGE_ACTIONS:
        english, day = _english_action(stage_name)
        _, kannada = i18n.stage_words(stage_name, english, i18n.KANNADA, day=day)
        rows.append(Row(
            "Field instructions", f"action:{stage_name}", english, kannada,
            "the What-to-do-next panel and the season plan"))

    for english, kannada in i18n.KANNADA_STAGES.items():
        rows.append(Row("Field stages", f"stage:{english}", english, kannada,
                        "the heading of each reminder"))

    for key, kannada in sorted(i18n.KANNADA_WHEN.items()):
        rows.append(Row("Timing", f"when:{key}", key.replace("_", " "), kannada,
                        "beside each reminder -- how soon it falls"))

    for english, kannada in i18n.KANNADA_MONTHS.items():
        rows.append(Row("Months", f"month:{english}", english, kannada,
                        "a saved record's date, inside a table column"))

    for english, kannada in i18n.KANNADA_LEDGER_COLUMNS.items():
        rows.append(Row("Table headers", f"column:{english}", english, kannada,
                        "the saved-records table -- a narrow phone column"))

    said = phrase_fallbacks()
    for key, kannada in sorted(i18n.KANNADA_PHRASES.items()):
        rows.append(Row("Sentences", key, said.get(key, ""), kannada,
                        "under the soil card and the weather box"))

    rows.append(Row("Warning", "review_status", i18n.REVIEW_STATUS,
                    i18n.REVIEW_STATUS_KN,
                    "shown to anyone who switches to Kannada"))

    for crop, profile in sorted(CROPS.items()):
        rows.append(Row(
            "Crop names", crop, profile.english, profile.kannada,
            "the crop card and the soil health card",
            evidenced=crop in SURVEY_CORROBORATED,
        ))

    return rows


def as_markdown(rows: List[Row]) -> str:
    out = io.StringIO()
    numbers = i18n.coverage()
    out.write("# GREENROOT — Kannada review sheet\n\n")
    out.write(
        "Please read each Kannada line against its English and mark anything "
        "that is wrong, unclear, or that a farmer would not say. **Where you "
        "are unsure, say so** — leaving a line flagged is more useful than "
        "guessing.\n\n"
        "This is agronomic advice people act on in a field, so a wrong word "
        "has a real cost. The English is the source of truth; the Kannada was "
        "written by someone who does not speak the language.\n\n"
    )
    out.write(
        f"- Interface strings translated: "
        f"**{numbers['copy_translated']} of {numbers['copy_keys']}**\n"
        f"- Extra strings: **{numbers['extra_translated']}**\n"
        f"- Crop names: **{len(CROPS)}**, of which "
        f"**{len(SURVEY_CORROBORATED)}** are already corroborated against the "
        f"government soil survey shipped with the project (marked ✅ below — "
        f"these need the least attention)\n\n"
    )
    out.write(
        "**If you only have time for one section, read _Field instructions_.** "
        "Those are the lines a farmer acts on with a bag of fertiliser in "
        "hand; everything else is navigation, and a clumsy button is "
        "recoverable in a way a wrong dose is not.\n\n"
    )

    for group in dict.fromkeys(r.group for r in rows):
        members = [r for r in rows if r.group == group]
        out.write(f"\n## {group}\n\n")
        out.write("| | English | Kannada | Correction |\n")
        out.write("|---|---|---|---|\n")
        for row in members:
            mark = "✅" if row.evidenced else ""
            english = (row.english or "_(no English equivalent)_").replace(
                "|", "\\|").replace("\n", " ")
            kannada = row.kannada.replace("|", "\\|").replace("\n", " ")
            out.write(f"| {mark} | {english} | {kannada} | |\n")
    return out.getvalue()


def as_csv(rows: List[Row]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["group", "key", "english", "kannada_draft",
                     "your_correction", "ok?", "already_evidenced", "shown_in"])
    for row in rows:
        writer.writerow([row.group, row.key, row.english, row.kannada, "", "",
                         "yes" if row.evidenced else "", row.where])
    return out.getvalue()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--format", choices=("md", "csv"), default="md")
    parser.add_argument("-o", "--output", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = collect()
    body = as_markdown(rows) if args.format == "md" else as_csv(rows)

    if args.output:
        args.output.write_text(body, encoding="utf-8")
        print(f"{len(rows)} strings written to {args.output}")
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

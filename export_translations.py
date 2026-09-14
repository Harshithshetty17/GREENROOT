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
import csv
import io
import sys
from pathlib import Path
from typing import List, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import i18n  # noqa: E402
from src.utils.agronomy import CROPS, SURVEY_CORROBORATED  # noqa: E402
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


def collect() -> List[Row]:
    """Every translated string in the app, grouped for a human reader."""
    rows: List[Row] = []

    for key, kannada in sorted(i18n.KANNADA_UI.items()):
        english = COPY[key][0] if key in COPY else ""
        rows.append(Row("Interface", key, english, kannada,
                        "buttons, headings and labels"))

    for key, kannada in sorted(i18n.KANNADA_EXTRA.items()):
        rows.append(Row("Interface (extra)", key, "", kannada,
                        "controls and section titles"))

    for english, (label_kn, detail_kn) in i18n.KANNADA_BANDS.items():
        rows.append(Row("Confidence", f"band:{english}", english, label_kn,
                        "beside the crop name -- the headline judgement"))
        rows.append(Row("Confidence", f"band-detail:{english}", "", detail_kn,
                        "the sentence under the crop card"))

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

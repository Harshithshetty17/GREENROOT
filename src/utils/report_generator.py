"""Soil Health Card rendering for agricultural extension officers.

The same recommendation payload is emitted in three formats for three
audiences: plain text for SMS and low-bandwidth field terminals, Markdown for
the record and for version-controlled archives, and print-ready HTML for the
physical card handed to the cultivator.

All three are produced from one :class:`SoilHealthCard` so the numbers can
never drift between formats. Every user-supplied string reaching the HTML
renderer is escaped, since district names and crop labels can originate from
free-text form input.
"""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.core.config import FEATURE_LABELS, FEATURE_NAMES, FEATURE_UNITS
from src.utils.agronomy_advisory import AdvisoryReport
from src.utils.localisation import (
    bilingual_crop,
    bilingual_feature,
    bilingual_label,
    label as tr_label,
)

logger = logging.getLogger(__name__)

#: Feature name -> (display label, unit), assembled from the config contract.
_FEATURE_META: Dict[str, Tuple[str, str]] = {
    name: (FEATURE_LABELS[i], FEATURE_UNITS[i]) for i, name in enumerate(FEATURE_NAMES)
}

_SOIL_FEATURES: Tuple[str, ...] = ("N", "P", "K", "ph")
_CLIMATE_FEATURES: Tuple[str, ...] = ("temperature", "humidity", "rainfall")


@dataclass(frozen=True)
class SoilHealthCard:
    """The complete data payload behind a printable advisory card.

    Attributes
    ----------
    district:
        Administrative unit the sample belongs to.
    features:
        Observed raw values keyed by :data:`~src.core.config.FEATURE_NAMES`.
    primary_crop:
        The recommended crop.
    confidence:
        Posterior for :attr:`primary_crop`, as a percentage.
    alternatives:
        ``(crop, confidence_pct)`` runner-up pairs, highest first.
    advisory:
        Agronomic advisory package, if one was generated.
    jaccard:
        SHAP/LIME agreement index, if an XAI audit was run.
    consensus_verdict:
        Human-readable fidelity verdict from the XAI audit.
    shap_drivers:
        Top TreeSHAP drivers behind the recommendation.
    generated_at:
        ISO-8601 timestamp; defaults to render time.
    bilingual:
        Render farmer-facing labels and crop names in Kannada alongside
        English. The English text always remains, so a mistranslation cannot
        silently change the advice.
    """

    district: str
    features: Dict[str, float]
    primary_crop: str
    confidence: float
    alternatives: Sequence[Tuple[str, float]] = field(default_factory=tuple)
    advisory: Optional[AdvisoryReport] = None
    jaccard: Optional[float] = None
    consensus_verdict: str = ""
    shap_drivers: Sequence[str] = field(default_factory=tuple)
    generated_at: str = ""
    bilingual: bool = False

    @property
    def timestamp(self) -> str:
        """Render timestamp, defaulting to now."""
        return self.generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")

    def text(self, key: str) -> str:
        """Return a card label, bilingual when the card is in bilingual mode."""
        return bilingual_label(key) if self.bilingual else tr_label(key)

    def crop_name(self, crop: str) -> str:
        """Return a crop name, bilingual when the card is in bilingual mode."""
        return bilingual_crop(crop) if self.bilingual else crop.capitalize()

    def _rows(self, keys: Sequence[str]) -> List[Tuple[str, str, str]]:
        """Return ``(label, formatted value, unit)`` rows for ``keys``."""
        rows: List[Tuple[str, str, str]] = []
        for key in keys:
            if key not in self.features:
                continue
            label, unit = _FEATURE_META.get(key, (key, ""))
            if self.bilingual:
                label = bilingual_feature(key)
            value = float(self.features[key])
            rows.append((label, f"{value:.1f}" if key == "ph" else f"{value:.0f}", unit))
        return rows

    def soil_rows(self) -> List[Tuple[str, str, str]]:
        """Tested soil chemistry rows."""
        return self._rows(_SOIL_FEATURES)

    def climate_rows(self) -> List[Tuple[str, str, str]]:
        """Microclimate rows."""
        return self._rows(_CLIMATE_FEATURES)


# --------------------------------------------------------------------------- #
# Plain text
# --------------------------------------------------------------------------- #
def render_text(card: SoilHealthCard, width: int = 68) -> str:
    """Render the card as fixed-width plain text.

    Suitable for SMS delivery, thermal receipt printers, and terminals without
    a rendering stack.
    """
    rule = "=" * width
    thin = "-" * width
    lines: List[str] = [
        rule,
        "GREENROOT — FARMER SOIL HEALTH CARD".center(width),
        "Precision Agriculture Decision Support System".center(width),
        rule,
        f"District : {card.district}",
        f"Issued   : {card.timestamp}",
        "",
        thin,
        "TESTED SOIL CHEMISTRY",
        thin,
    ]
    for label, value, unit in card.soil_rows():
        lines.append(f"  {label:<26} {value:>10} {unit}")

    lines += ["", thin, "MICROCLIMATE AT ASSESSMENT", thin]
    for label, value, unit in card.climate_rows():
        lines.append(f"  {label:<26} {value:>10} {unit}")

    lines += [
        "",
        thin,
        "RECOMMENDATION",
        thin,
        f"  Primary crop     : {card.primary_crop.upper()}",
        f"  Model confidence : {card.confidence:.1f}%",
    ]
    runners = [(c, p) for c, p in card.alternatives if c != card.primary_crop]
    if runners:
        lines.append("  Secondary options:")
        for rank, (crop, pct) in enumerate(runners, start=2):
            lines.append(f"    {rank}. {crop.capitalize():<18} {pct:>6.2f}%")

    if card.jaccard is not None:
        lines += [
            "",
            thin,
            "EXPLAINABILITY AUDIT",
            thin,
            f"  SHAP/LIME Jaccard index : {card.jaccard:.2f}",
            f"  Consensus verdict       : {card.consensus_verdict or 'n/a'}",
        ]
        if card.shap_drivers:
            lines.append(f"  Dominant drivers        : {', '.join(card.shap_drivers)}")

    if card.advisory and card.advisory.items:
        lines += ["", thin, "AGRONOMIC ADVISORY", thin]
        for item in card.advisory.items:
            marker = {"critical": "[!]", "warning": "[*]", "info": "[-]"}.get(
                item.severity, "[-]"
            )
            lines += _wrap(f"{marker} {item.category}: {item.message}", width - 2, "      ")

    if card.advisory and card.advisory.fertiliser_plan:
        lines += ["", thin, "FERTILISER PRESCRIPTION (per hectare)", thin]
        for product, quantity in card.advisory.fertiliser_plan.items():
            lines.append(f"  {product:<38} {quantity:>8.1f} kg")

    lines += [
        "",
        rule,
        "Advisory only. Corroborate with a certified laboratory soil test".center(width),
        "before committing the season.".center(width),
        rule,
    ]
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> List[str]:
    """Greedily wrap ``text`` to ``width``, indenting continuation lines."""
    words = text.split()
    if not words:
        return []
    lines: List[str] = []
    current = "  " + words[0]
    for word in words[1:]:
        if len(current) + len(word) + 1 <= width:
            current += " " + word
        else:
            lines.append(current)
            current = indent + word
    lines.append(current)
    return lines


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def render_markdown(card: SoilHealthCard) -> str:
    """Render the card as Markdown for archival and version control."""
    lines: List[str] = [
        "# GREENROOT — Farmer Soil Health Card",
        "",
        f"**District:** {card.district}  ",
        f"**Issued:** {card.timestamp}",
        "",
        "## 1. Tested Soil Chemistry",
        "",
        "| Parameter | Value | Unit |",
        "|---|---:|---|",
    ]
    lines += [f"| {label} | {value} | {unit} |" for label, value, unit in card.soil_rows()]

    lines += [
        "",
        "## 2. Microclimate at Assessment",
        "",
        "| Parameter | Value | Unit |",
        "|---|---:|---|",
    ]
    lines += [
        f"| {label} | {value} | {unit} |" for label, value, unit in card.climate_rows()
    ]

    lines += [
        "",
        "## 3. Recommendation",
        "",
        f"**Primary crop:** `{card.primary_crop.upper()}` "
        f"— {card.confidence:.1f}% model confidence",
        "",
        "| Rank | Crop | Confidence |",
        "|---:|---|---:|",
    ]
    for rank, (crop, pct) in enumerate(card.alternatives, start=1):
        lines.append(f"| {rank} | {crop.capitalize()} | {pct:.2f}% |")

    if card.jaccard is not None:
        lines += [
            "",
            "## 4. Explainability Audit",
            "",
            f"- **SHAP ∩ LIME Jaccard index (k=3):** {card.jaccard:.2f}",
            f"- **Consensus verdict:** {card.consensus_verdict or 'n/a'}",
        ]
        if card.shap_drivers:
            lines.append(f"- **Dominant drivers:** {', '.join(card.shap_drivers)}")

    if card.advisory and card.advisory.items:
        lines += ["", "## 5. Agronomic Advisory", ""]
        for item in card.advisory.items:
            lines.append(f"- {item.icon} **{item.category}** — {item.message}")

    if card.advisory and card.advisory.fertiliser_plan:
        lines += [
            "",
            "## 6. Fertiliser Prescription",
            "",
            "| Product | Quantity (kg/ha) |",
            "|---|---:|",
        ]
        for product, quantity in card.advisory.fertiliser_plan.items():
            lines.append(f"| {product} | {quantity:.1f} |")

    lines += [
        "",
        "---",
        "",
        "_Advisory only. Corroborate with a certified laboratory soil test "
        "before committing the season._",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_HTML_STYLE = """
:root { --ink:#14281d; --muted:#5c6f63; --line:#d5e0d8; --accent:#1f7a4d;
        --bg:#ffffff; --panel:#f4f8f5; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--ink);
       font-family:"Segoe UI", Roboto, "Noto Sans Kannada", "Tunga",
                   "Nirmala UI", Helvetica, Arial, sans-serif;
       font-size:14px; line-height:1.55; }
.card { max-width:820px; margin:0 auto; border:1px solid var(--line);
        border-radius:10px; overflow:hidden; }
header { background:var(--accent); color:#fff; padding:20px 24px; }
header h1 { margin:0 0 4px; font-size:20px; letter-spacing:.3px; }
header p { margin:0; opacity:.9; font-size:12.5px; }
.meta { display:flex; flex-wrap:wrap; gap:24px; padding:14px 24px;
        background:var(--panel); border-bottom:1px solid var(--line);
        font-size:13px; }
.meta span { color:var(--muted); }
.meta strong { color:var(--ink); }
section { padding:18px 24px; border-bottom:1px solid var(--line); }
section:last-of-type { border-bottom:none; }
h2 { font-size:13px; text-transform:uppercase; letter-spacing:.8px;
     color:var(--muted); margin:0 0 12px; font-weight:600; }
table { width:100%; border-collapse:collapse; }
th, td { padding:7px 10px; text-align:left; border-bottom:1px solid var(--line);
         font-size:13px; }
th { color:var(--muted); font-weight:600; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
.headline { display:flex; flex-wrap:wrap; align-items:baseline; gap:12px; }
.crop { font-size:26px; font-weight:700; color:var(--accent);
        text-transform:uppercase; letter-spacing:.5px; }
.conf { font-size:14px; color:var(--muted); }
ul.advisory { list-style:none; margin:0; padding:0; }
ul.advisory li { padding:8px 0 8px 12px; border-left:3px solid var(--line);
                 margin-bottom:8px; }
li.critical { border-left-color:#c0392b; background:#fdf3f2; }
li.warning  { border-left-color:#d98b0e; background:#fdf8ee; }
li.info     { border-left-color:var(--accent); background:var(--panel); }
li .cat { font-weight:600; }
footer { padding:14px 24px; background:var(--panel); color:var(--muted);
         font-size:12px; text-align:center; }
@media print { body { padding:0; } .card { border:none; } header { -webkit-print-color-adjust:exact; print-color-adjust:exact; } }
"""


def _esc(value: object) -> str:
    """HTML-escape an arbitrary value for safe interpolation."""
    return html.escape(str(value), quote=True)


def _html_rows(rows: Sequence[Tuple[str, str, str]]) -> str:
    """Render parameter rows as HTML table rows."""
    return "\n".join(
        f"<tr><td>{_esc(label)}</td><td class='num'>{_esc(value)}</td>"
        f"<td>{_esc(unit)}</td></tr>"
        for label, value, unit in rows
    )


def render_html(card: SoilHealthCard) -> str:
    """Render the card as a self-contained, print-ready HTML document."""
    alternatives = "\n".join(
        f"<tr><td class='num'>{rank}</td><td>{_esc(card.crop_name(crop))}</td>"
        f"<td class='num'>{pct:.2f}%</td></tr>"
        for rank, (crop, pct) in enumerate(card.alternatives, start=1)
    )

    xai_section = ""
    if card.jaccard is not None:
        drivers = (
            f"<tr><td>Dominant drivers</td><td class='num'>"
            f"{_esc(', '.join(card.shap_drivers))}</td></tr>"
            if card.shap_drivers
            else ""
        )
        xai_section = f"""
    <section>
      <h2>Explainability Audit</h2>
      <table>
        <tr><td>SHAP &cap; LIME Jaccard index (k=3)</td>
            <td class='num'>{card.jaccard:.2f}</td></tr>
        <tr><td>Consensus verdict</td>
            <td class='num'>{_esc(card.consensus_verdict or 'n/a')}</td></tr>
        {drivers}
      </table>
    </section>"""

    advisory_section = ""
    if card.advisory and card.advisory.items:
        items = "\n".join(
            f"<li class='{_esc(item.severity)}'>"
            f"<span class='cat'>{_esc(item.category)}</span> — {_esc(item.message)}</li>"
            for item in card.advisory.items
        )
        advisory_section = f"""
    <section>
      <h2>{_esc(card.text("advisory"))}</h2>
      <ul class='advisory'>
{items}
      </ul>
    </section>"""

    fertiliser_section = ""
    if card.advisory and card.advisory.fertiliser_plan:
        rows = "\n".join(
            f"<tr><td>{_esc(product)}</td><td class='num'>{quantity:.1f}</td></tr>"
            for product, quantity in card.advisory.fertiliser_plan.items()
        )
        fertiliser_section = f"""
    <section>
      <h2>{_esc(card.text("fertiliser"))}</h2>
      <table>
        <tr><th>Product</th><th class='num'>Quantity (kg)</th></tr>
{rows}
      </table>
    </section>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Kannada:wght@400;600&display=swap" rel="stylesheet">
<title>GREENROOT Soil Health Card — {_esc(card.district)}</title>
<style>{_HTML_STYLE}</style>
</head>
<body>
  <div class="card">
    <header>
      <h1>GREENROOT — {_esc(card.text("card_title"))}</h1>
      <p>{_esc(card.text("subtitle"))}</p>
    </header>
    <div class="meta">
      <div><span>{_esc(card.text("district"))}</span><br><strong>{_esc(card.district)}</strong></div>
      <div><span>{_esc(card.text("issued"))}</span><br><strong>{_esc(card.timestamp)}</strong></div>
      <div><span>Assessment</span><br><strong>Stacking ensemble + XAI consensus</strong></div>
    </div>
    <section>
      <h2>{_esc(card.text("recommendation"))}</h2>
      <div class="headline">
        <span class="crop">{_esc(card.crop_name(card.primary_crop))}</span>
        <span class="conf">{card.confidence:.1f}% — {_esc(card.text("confidence"))}</span>
      </div>
    </section>
    <section>
      <h2>{_esc(card.text("soil_chemistry"))}</h2>
      <table>
        <tr><th>{_esc(card.text('parameter'))}</th><th class='num'>{_esc(card.text('value'))}</th><th>{_esc(card.text('unit'))}</th></tr>
{_html_rows(card.soil_rows())}
      </table>
    </section>
    <section>
      <h2>{_esc(card.text("microclimate"))}</h2>
      <table>
        <tr><th>{_esc(card.text('parameter'))}</th><th class='num'>{_esc(card.text('value'))}</th><th>{_esc(card.text('unit'))}</th></tr>
{_html_rows(card.climate_rows())}
      </table>
    </section>
    <section>
      <h2>{_esc(card.text("alternatives"))}</h2>
      <table>
        <tr><th class='num'>{_esc(card.text('rank'))}</th><th>{_esc(card.text('crop'))}</th><th class='num'>{_esc(card.text('confidence'))}</th></tr>
{alternatives}
      </table>
    </section>{xai_section}{advisory_section}{fertiliser_section}
    <footer>
      {_esc(card.text('disclaimer'))}
    </footer>
  </div>
</body>
</html>"""



# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
#: Brand palette as RGB triples, mirroring the HTML stylesheet.
_PDF_ACCENT = (31, 122, 77)
_PDF_INK = (20, 40, 29)
_PDF_MUTED = (92, 111, 99)
_PDF_LINE = (213, 224, 216)
_PDF_PANEL = (244, 248, 245)

#: Severity marker colours for advisory items.
_PDF_SEVERITY = {
    "critical": (192, 57, 43),
    "warning": (217, 139, 14),
    "info": _PDF_ACCENT,
}


class PDFUnavailableError(RuntimeError):
    """Raised when the optional PDF dependency is not installed."""


def _latin(text: str) -> str:
    """Reduce text to characters the built-in PDF font can encode.

    fpdf2's core fonts are Latin-1 only. The card's Kannada text therefore
    cannot be rendered without embedding a Unicode font with Kannada coverage,
    which is not shipped with this project — so the PDF is English-only and
    this helper strips anything unencodable rather than raising mid-render.
    Typographic punctuation is transliterated first so the output stays clean
    rather than merely legal.
    """
    # Order matters: multi-character sequences must precede their prefixes,
    # so "\u00b0C" is handled before a bare "\u00b0".
    replacements = {
        "\u00b0C": "C",
        "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b0": " deg",
        "\u2264": "<=", "\u2265": ">=", "\u00d7": "x", "\u2192": "->",
        "\u2229": "n", "\u222a": "u", "\u03c3": "sd",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("latin-1", "ignore").decode("latin-1")


def render_pdf(card: SoilHealthCard) -> bytes:
    """Render the card as a print-ready single-page A4 PDF.

    The PDF is what an extension officer actually hands over, so it is laid out
    programmatically rather than converted from HTML — no browser or system
    library is required, and the result is byte-identical across platforms.

    Parameters
    ----------
    card:
        The payload to render. Bilingual cards fall back to English here; see
        :func:`_latin`.

    Returns
    -------
    bytes
        The PDF document.

    Raises
    ------
    PDFUnavailableError
        If ``fpdf2`` is not installed.
    """
    try:
        from fpdf import FPDF
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise PDFUnavailableError(
            "PDF export needs fpdf2. Install it with: pip install fpdf2"
        ) from exc

    # The core PDF fonts are Latin-1 only, so a bilingual card is rendered from
    # its English view: stripping Kannada from "Soil pH / ..." in place would
    # leave a dangling separator.
    if card.bilingual:
        card = dataclasses_replace(card, bilingual=False)

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    width = pdf.w - 2 * pdf.l_margin

    def heading(text: str) -> None:
        """Section heading in the muted small-caps style of the HTML card."""
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*_PDF_MUTED)
        pdf.cell(0, 5, _latin(text.upper()), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*_PDF_LINE)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + width, pdf.get_y())
        pdf.ln(2)

    def parameter_table(rows: List[Tuple[str, str, str]]) -> None:
        """Three-column parameter table with alternating row shading."""
        pdf.set_font("Helvetica", "", 9.5)
        for index, (name, value, unit) in enumerate(rows):
            if index % 2 == 0:
                pdf.set_fill_color(*_PDF_PANEL)
                fill = True
            else:
                fill = False
            pdf.set_text_color(*_PDF_INK)
            pdf.cell(width * 0.55, 6.5, _latin(f"  {name}"), fill=fill)
            pdf.cell(width * 0.25, 6.5, _latin(value), align="R", fill=fill)
            pdf.cell(
                width * 0.20, 6.5, _latin(f" {unit}"), fill=fill,
                new_x="LMARGIN", new_y="NEXT",
            )

    # ---- Header banner ---------------------------------------------------- #
    pdf.set_fill_color(*_PDF_ACCENT)
    pdf.rect(0, 0, pdf.w, 26, style="F")
    pdf.set_xy(pdf.l_margin, 7)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 7, _latin("GREENROOT - Farmer Soil Health Card"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, _latin("Precision Agriculture Decision Support System"),
             new_x="LMARGIN", new_y="NEXT")

    # ---- Meta strip ------------------------------------------------------- #
    pdf.set_y(30)
    pdf.set_fill_color(*_PDF_PANEL)
    pdf.rect(pdf.l_margin, pdf.get_y(), width, 12, style="F")
    pdf.set_xy(pdf.l_margin + 3, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*_PDF_MUTED)
    pdf.cell(width * 0.5, 4, _latin("DISTRICT"))
    pdf.cell(width * 0.5, 4, _latin("ISSUED"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 3)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*_PDF_INK)
    pdf.cell(width * 0.5, 5, _latin(card.district))
    pdf.cell(width * 0.5, 5, _latin(card.timestamp), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # ---- Recommendation --------------------------------------------------- #
    heading("Recommendation")
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*_PDF_ACCENT)
    pdf.cell(0, 10, _latin(card.primary_crop.upper()), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(*_PDF_MUTED)
    pdf.cell(0, 5, _latin(f"{card.confidence:.1f}% model confidence"),
             new_x="LMARGIN", new_y="NEXT")

    # ---- Measurements ----------------------------------------------------- #
    heading("Tested Soil Chemistry")
    parameter_table(card.soil_rows())
    heading("Microclimate at Assessment")
    parameter_table(card.climate_rows())

    # ---- Ranked suitability ----------------------------------------------- #
    heading("Ranked Crop Suitability")
    pdf.set_font("Helvetica", "", 9.5)
    for rank, (crop, percentage) in enumerate(card.alternatives, start=1):
        fill = rank % 2 == 1
        if fill:
            pdf.set_fill_color(*_PDF_PANEL)
        pdf.set_text_color(*_PDF_INK)
        pdf.cell(width * 0.12, 6.5, _latin(f"  {rank}."), fill=fill)
        pdf.cell(width * 0.58, 6.5, _latin(crop.capitalize()), fill=fill)
        pdf.cell(width * 0.30, 6.5, _latin(f"{percentage:.2f}%"), align="R",
                 fill=fill, new_x="LMARGIN", new_y="NEXT")

    # ---- Explainability --------------------------------------------------- #
    if card.jaccard is not None:
        heading("Explainability Audit")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(*_PDF_INK)
        rows = [
            ("SHAP n LIME Jaccard index (k=3)", f"{card.jaccard:.2f}", ""),
            ("Consensus verdict", card.consensus_verdict or "n/a", ""),
        ]
        if card.shap_drivers:
            rows.append(("Dominant drivers", ", ".join(card.shap_drivers), ""))
        parameter_table(rows)

    # ---- Advisory --------------------------------------------------------- #
    if card.advisory and card.advisory.items:
        heading("Agronomic Advisory")
        for item in card.advisory.items:
            colour = _PDF_SEVERITY.get(item.severity, _PDF_ACCENT)
            top = pdf.get_y()
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*colour)
            pdf.set_x(pdf.l_margin + 3)
            pdf.cell(0, 4.6, _latin(item.category), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*_PDF_INK)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(width - 6, 4.4, _latin(item.message))
            # Severity bar drawn after the text, so its height matches the item.
            pdf.set_fill_color(*colour)
            pdf.rect(pdf.l_margin, top, 1.2, pdf.get_y() - top, style="F")
            pdf.ln(1.6)

    # ---- Fertiliser ------------------------------------------------------- #
    if card.advisory and card.advisory.fertiliser_plan:
        heading("Fertiliser Prescription (per hectare)")
        parameter_table(
            [(product, f"{quantity:.1f}", "kg")
             for product, quantity in card.advisory.fertiliser_plan.items()]
        )

    # ---- Footer ----------------------------------------------------------- #
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(*_PDF_MUTED)
    pdf.multi_cell(
        width, 4,
        _latin(
            "Advisory only. Corroborate with a certified laboratory soil test "
            "before committing the season."
        ),
        align="C",
    )

    output = pdf.output()
    return bytes(output)


def save_report(
    card: SoilHealthCard, path: Path, fmt: str = "html", encoding: str = "utf-8"
) -> Path:
    """Render ``card`` in ``fmt`` and write it to ``path``.

    Parameters
    ----------
    card:
        The payload to render.
    path:
        Destination file; parent directories are created as needed.
    fmt:
        One of ``'text'``, ``'markdown'``, or ``'html'``.

    Returns
    -------
    pathlib.Path
        The path written.

    Raises
    ------
    ValueError
        If ``fmt`` is not a supported format.
    """
    renderers = {
        "text": render_text,
        "txt": render_text,
        "markdown": render_markdown,
        "md": render_markdown,
        "html": render_html,
        "pdf": render_pdf,
    }
    renderer = renderers.get(fmt.lower())
    if renderer is None:
        raise ValueError(
            f"Unsupported format {fmt!r}; expected one of {sorted(set(renderers))}."
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = renderer(card)
    if isinstance(rendered, bytes):
        destination.write_bytes(rendered)
    else:
        destination.write_text(rendered, encoding=encoding)
    logger.info("Wrote %s report to %s", fmt, destination)
    return destination


def build_card(
    district: str,
    prediction,
    advisory: Optional[AdvisoryReport] = None,
    consensus=None,
) -> SoilHealthCard:
    """Assemble a :class:`SoilHealthCard` from the pipeline's own objects.

    Parameters
    ----------
    district:
        Administrative unit for the header.
    prediction:
        A :class:`~src.models.inference.PredictionResult`.
    advisory:
        An optional :class:`~src.utils.agronomy_advisory.AdvisoryReport`.
    consensus:
        An optional :class:`~src.models.xai_engine.ConsensusReport`.
    """
    return SoilHealthCard(
        district=district,
        features=dict(prediction.raw_features),
        primary_crop=prediction.crop,
        confidence=float(prediction.confidence),
        alternatives=[(c.crop, c.confidence_pct) for c in prediction.top_k],
        advisory=advisory,
        jaccard=float(consensus.jaccard) if consensus is not None else None,
        consensus_verdict=consensus.verdict if consensus is not None else "",
        shap_drivers=tuple(consensus.shap_top_k) if consensus is not None else (),
    )


__all__ = [
    "SoilHealthCard",
    "build_card",
    "render_text",
    "render_markdown",
    "render_html",
    "render_pdf",
    "save_report",
    "PDFUnavailableError",
]

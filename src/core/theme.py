"""Design tokens and chart styling for the GREENROOT dashboard.

One module owns every colour and every matplotlib default, so the dashboard,
the Soil Health Card and the evaluation figures cannot drift apart.

Colour policy
-------------
Colour is assigned by the *job the data does*, not by taste:

* **Categorical** (crop identity, in the what-if sweep) — a fixed eight-slot
  order, assigned by position and never cycled. The order below is the
  validated reference ordering: its worst adjacent pair scores ΔE 9.1 under
  simulated protanopia (OKLab ×100, ≥8 required) and 19.6 under normal vision
  (≥15 required), in both light and dark modes. Candidate orderings led by the
  brand green were tested and rejected — they pass in light but fall into the
  6–8 warning band or fail outright in dark — so the brand green is used for
  chrome and single-series marks instead, and identity encoding keeps the
  validated order.
* **Sequential / magnitude** (crop counts) — one series, one colour. A
  value-ramp across nominal categories would double-encode bar length as hue.
* **Diverging** (Z-scores, SHAP/LIME attributions) — a warm/cool pair either
  side of a neutral grey zero. These quantities are signed, not good/bad, so
  the status palette would be wrong: high nitrogen is neither.
* **Status** (advisory severity) — reserved for good/warning/critical and
  never reused as a series colour.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import matplotlib as mpl

# --------------------------------------------------------------------------- #
# Brand
# --------------------------------------------------------------------------- #
BRAND = "#1f7a4d"
BRAND_DARK = "#14603c"
BRAND_LIGHT = "#e7f2eb"
BRAND_EDGE = "#b9d8c6"

INK = "#14281d"
INK_SECONDARY = "#4a5b51"
MUTED = "#5c6f63"
LINE = "#d9e3dc"
GRID = "#eef3f0"
SURFACE = "#ffffff"
PANEL = "#f4f8f5"

# --------------------------------------------------------------------------- #
# Categorical — validated fixed order (see module docstring)
# --------------------------------------------------------------------------- #
CATEGORICAL: List[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

#: Hard cap on simultaneous categorical series. Past this the guidance is to
#: fold the tail into "Other" or facet, never to generate a ninth hue.
MAX_SERIES: int = 8

# --------------------------------------------------------------------------- #
# Diverging — warm/cool poles with a neutral midpoint
# --------------------------------------------------------------------------- #
DIVERGING_HIGH = "#2a78d6"  # above the reference mean
DIVERGING_LOW = "#e34948"   # below the reference mean
DIVERGING_MID = "#f0efec"   # neutral zero

# --------------------------------------------------------------------------- #
# Status — reserved; never a series colour
# --------------------------------------------------------------------------- #
STATUS: Dict[str, str] = {
    "critical": "#c0392b",
    "warning": "#d98b0e",
    "info": BRAND,
    "good": BRAND,
}


def series_colour(index: int) -> str:
    """Return the categorical colour for slot ``index`` (0-based).

    Raises
    ------
    IndexError
        Past :data:`MAX_SERIES`. Generating a ninth hue would be
        indistinguishable from an existing slot under colour-vision
        deficiency, so callers must fold or facet instead.
    """
    if index >= MAX_SERIES:
        raise IndexError(
            f"Categorical slot {index} exceeds the {MAX_SERIES}-series cap; "
            f"fold the tail into 'Other' or facet into small multiples."
        )
    return CATEGORICAL[index]


def series_palette(count: int) -> List[str]:
    """Return the first ``count`` categorical colours, in fixed order."""
    return [series_colour(i) for i in range(min(count, MAX_SERIES))]


def diverging_colours(values: Sequence[float]) -> List[str]:
    """Map signed values onto the diverging pair."""
    return [DIVERGING_HIGH if float(v) >= 0 else DIVERGING_LOW for v in values]


# --------------------------------------------------------------------------- #
# Matplotlib
# --------------------------------------------------------------------------- #
def apply_matplotlib_theme() -> None:
    """Install the house chart style as the matplotlib default.

    Thin marks, hairline recessive axes, generous padding, no dashed grid —
    the chart's ink should be its data, not its chrome.
    """
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": LINE,
            "axes.linewidth": 0.8,
            "axes.labelcolor": MUTED,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "axes.titlecolor": INK,
            "axes.titlepad": 12,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",       # never dashed: dashing reads as a threshold
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "font.size": 11,
            "lines.linewidth": 2.0,      # thin marks
            "lines.markersize": 8,       # >= 8px markers
            "figure.autolayout": False,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        }
    )


def style_axes(axes, *, grid_axis: str = "x") -> None:
    """Strip the top/right spines and confine the grid to one axis."""
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(LINE)
    axes.grid(False)
    axes.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelsize=10.5, length=0)


__all__ = [
    "BRAND", "BRAND_DARK", "BRAND_LIGHT", "BRAND_EDGE",
    "INK", "INK_SECONDARY", "MUTED", "LINE", "GRID", "SURFACE", "PANEL",
    "CATEGORICAL", "MAX_SERIES", "STATUS",
    "DIVERGING_HIGH", "DIVERGING_LOW", "DIVERGING_MID",
    "series_colour", "series_palette", "diverging_colours",
    "apply_matplotlib_theme", "style_axes",
]

"""Shared plotting style for the workbooks and the verification report figures.

Kept in one place so every figure in the repository reads as one set, and so the
colour choices are stated once with their reasons rather than re-decided per plot.

Colour assignment follows the job each series does, not aesthetics:

* **TRUTH** and **ESTIMATE** are a categorical pair — two identities, no ordering
  between them. They are a blue/orange pair whose separation was checked for
  colour-vision deficiency rather than eyeballed: worst-pair ΔE 9.2 under deuteranopia
  and 24.0 under normal vision, against floors of 8 and 15.
* **OBSERVED** data is neutral grey. It is context the eye should read *past*, not a
  third identity competing with the two that matter.
* **Spectrograms are a single hue, light to dark.** The quantity is a magnitude, so
  the encoding must be monotonic in lightness. The rainbow maps still common in EEG
  work (`jet` and its relatives) are not monotonic: they manufacture visual edges
  where the data is smooth and hide real structure where it is not.

Nothing here is required to use the package — this module exists only to make the
figures, and matplotlib is a `docs` extra rather than a runtime dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

#: Categorical pair: two identities with no ordering between them.
TRUTH = "#eb6834"
ESTIMATE = "#2a78d6"

#: Context, deliberately recessive.
OBSERVED = "#9b9a94"

#: Interval fills share the estimate's hue at low alpha, so the band reads as
#: belonging to the estimate rather than as a separate series.
BAND = "#2a78d6"

#: A third categorical slot, used only where a direct label is also present: it sits
#: below 3:1 contrast on a light surface, so colour alone would not carry it.
THIRD = "#1baf7a"

#: Status, reserved and never reused as a series colour.
BAD = "#e34948"
GOOD = "#008300"

#: Single-hue sequential ramp for magnitude.
SEQUENTIAL = "Blues"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d8d7d2"

FIGURE_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"


def use_report_style() -> None:
    """Apply the shared rcParams. Call once at the top of a notebook or script."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def annotate(ax: Any, text: str, *, loc: str = "upper right") -> None:
    """A short result note on the axes, in text ink rather than a series colour."""
    positions = {
        "upper right": (0.975, 0.955, "right", "top"),
        "upper left": (0.025, 0.955, "left", "top"),
        "lower right": (0.975, 0.045, "right", "bottom"),
        "lower left": (0.025, 0.045, "left", "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8,
        color=INK_SECONDARY,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )


def save(fig: Any, name: str) -> Path:
    """Write a figure into docs/figures and return the path."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path

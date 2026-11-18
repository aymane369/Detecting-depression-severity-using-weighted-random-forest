"""Matplotlib helpers. Figure texts are in French because the figures are embedded in the
French report; all code and comments stay in English."""
from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

# colour-blind friendly palette (Okabe-Ito)
BLUE, ORANGE, GREEN, RED, PURPLE, SKY, YELLOW, GREY = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#7f7f7f")
SEVERITY_COLORS = [BLUE, SKY, ORANGE, RED]

CLASS_FR_MULTI = ["Absente", "Légère", "Modérée", "Sévère"]
CLASS_FR_BINARY = ["Non déprimé", "Déprimé"]


def class_names_fr(task: str) -> list[str]:
    return CLASS_FR_BINARY if task == "binary" else CLASS_FR_MULTI


def set_style() -> None:
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 110, "savefig.dpi": 300,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "grid.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
    })


def save(fig, name: str) -> None:
    """Save a figure as vector PDF (for LaTeX) and PNG (for quick viewing)."""
    C.FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(C.FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(C.FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

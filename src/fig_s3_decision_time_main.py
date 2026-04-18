"""Figure S3: Decision time distributions for the main experiment.

A (top left): Nominal first movers in SM
B (bottom left): All decisions in SM
C (top right): Observed first-mover decisions in RT
D (bottom right): Recovered first-mover decisions in RT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

from src.data import load_main
from src.data.utils import OUTPUT_FIGURES
from src.fig3_decision_time import compute_latent_stats

# Figure style
FONT_SIZE = 10
CI_FONT_SIZE = 8
FIG_WIDTH_MM = 180
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4

mpl.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
    }
)

BINS = np.arange(0, 62, 2)
C_COOP = "tab:orange"
C_DEFECT = "tab:blue"


def _setup_inset(ax, position):
    inset = ax.inset_axes(position)
    inset.set_facecolor("#f0f0f0")
    inset.set_box_aspect(1)
    inset.spines[["top", "right"]].set_visible(False)
    inset.set_xlim(0, 60.05)
    inset.set_ylim(0, 1.01)
    inset.set_yticks([0, 1])
    inset.set_xticks([0, 60])
    inset.set_xlabel("DT", labelpad=-5, fontsize=CI_FONT_SIZE)
    inset.set_ylabel("Scaled\nCDF", fontsize=CI_FONT_SIZE, labelpad=-5)
    inset.tick_params(labelsize=CI_FONT_SIZE)
    return inset


def _plot_observed_panel(ax, data, ylim=15, inset_pos=None):
    if inset_pos is None:
        inset_pos = [0.5, 0.5, 0.45, 0.45]

    for is_coop, grp in data.groupby("is_coop"):
        color = C_COOP if is_coop else C_DEFECT
        ax.hist(
            grp.decision_time,
            bins=BINS,
            color=color,
            alpha=0.75,
            zorder=2 - is_coop,
        )

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-2, 62)
    ax.set_ylim(0, ylim)
    ax.set_xlabel("Decision Time (sec)", fontsize=CI_FONT_SIZE)
    ax.set_ylabel("Number of Participants", fontsize=CI_FONT_SIZE)
    ax.grid(True, linestyle=":", alpha=0.5)

    inset = _setup_inset(ax, inset_pos)
    for is_coop, grp in data.groupby("is_coop"):
        color = C_COOP if is_coop else C_DEFECT
        inset.hist(
            grp.decision_time,
            bins=np.arange(0, 68, 2),
            color=color,
            histtype="step",
            cumulative=True,
            density=True,
            linewidth=1.5,
        )


def _plot_recovered_panel(ax, data, inset_pos=None):
    if inset_pos is None:
        inset_pos = [0.5, 0.5, 0.45, 0.45]

    stats = compute_latent_stats(data.decision_time.values, data.is_coop.values)

    ax.bar(
        BINS[:-1],
        stats["latent_coop_pdf"],
        color=C_COOP,
        alpha=0.75,
        width=2,
        align="edge",
    )
    ax.bar(
        BINS[:-1],
        stats["latent_defect_pdf"],
        color=C_DEFECT,
        alpha=0.75,
        width=2,
        align="edge",
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-2, 62)
    ax.set_xlabel("Decision Time (sec)", fontsize=CI_FONT_SIZE)
    ax.set_ylabel("Density", fontsize=CI_FONT_SIZE)
    ax.set_yticks([0, 0.1, 0.2])
    ax.grid(True, linestyle=":", alpha=0.5)

    inset = _setup_inset(ax, inset_pos)
    inset.plot(BINS, stats["latent_coop_cdf"], color=C_COOP, linewidth=1.5)
    inset.plot(BINS, stats["latent_defect_cdf"], color=C_DEFECT, linewidth=1.5)


def main():
    exp_data = load_main()

    # SM data
    sm_data = exp_data[exp_data.condition == "SM"]
    sm_first = sm_data[sm_data.is_first_mover]
    # RT: first movers only
    rt_data = exp_data[(exp_data.condition == "RT") & exp_data.is_first_mover]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(FIG_WIDTH_IN * 0.8, FIG_WIDTH_IN * 0.65),
        constrained_layout=dict(w_pad=0.2, h_pad=0.2),
    )

    # Panel A (top left): Nominal first movers in SM
    _plot_observed_panel(axes[0, 0], sm_first, ylim=30)
    axes[0, 0].set_title("Nominal first movers in SM", fontsize=FONT_SIZE, pad=8)
    axes[0, 0].set_yticks([0, 10, 20, 30])

    # Panel B (bottom left): All decisions in SM
    _plot_observed_panel(axes[1, 0], sm_data, ylim=60)
    axes[1, 0].set_title("All decisions in SM", fontsize=FONT_SIZE, pad=8)

    # Panel C (top right): Observed first movers in RT
    _plot_observed_panel(axes[0, 1], rt_data, ylim=30)
    axes[0, 1].set_title("Observed first-mover decisions in RT", fontsize=FONT_SIZE, pad=8)
    axes[0, 1].set_yticks([0, 10, 20, 30])

    # Panel D (bottom right): Recovered first movers in RT
    _plot_recovered_panel(axes[1, 1], rt_data)
    axes[1, 1].set_title("Recovered first-mover decisions in RT", fontsize=FONT_SIZE, pad=8)

    # Legend
    legend_elements = [
        Patch(facecolor=C_COOP, alpha=0.75, label="Cooperate"),
        Patch(facecolor=C_DEFECT, alpha=0.75, label="Defect"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="lower right", fontsize=CI_FONT_SIZE, frameon=False)

    # Panel labels
    fig.text(0.01, 1, "A", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")
    fig.text(0.01, 0.5, "B", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")
    fig.text(0.51, 1, "C", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")
    fig.text(0.51, 0.5, "D", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")

    out = OUTPUT_FIGURES / "fig_s3_decision_time_main.pdf"
    fig.savefig(out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

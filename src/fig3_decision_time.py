import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.data import load_main
from src.data.utils import OUTPUT_FIGURES

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


def compute_empirical_cdf(dt_array, grid):
    dt_sorted = np.sort(np.asarray(dt_array, float))
    ranks = np.searchsorted(dt_sorted, grid, side="right")
    return ranks / dt_sorted.size


def recover_latent_cdf(observed_cdf):
    return 1.0 - np.sqrt(1.0 - observed_cdf)


def compute_empirical_conditional_p_coop(dt_array, is_coop_array, grid):
    dt_array = np.asarray(dt_array, float)
    is_coop_array = np.asarray(is_coop_array, float)
    dt_group = np.digitize(dt_array, grid)
    p_coop = {g: is_coop_array[dt_group == g].mean() for g in np.unique(dt_group)}
    p_coop = pd.Series(p_coop).reindex(range(1, len(grid)), fill_value=np.nan).values
    return p_coop


def compute_latent_stats(dt_array, is_coop_array, grid=BINS):
    empirical_cdf = compute_empirical_cdf(dt_array, grid)
    latent_cdf = recover_latent_cdf(empirical_cdf)
    p_coop_conditional = compute_empirical_conditional_p_coop(dt_array, is_coop_array, grid)

    latent_pdf = np.diff(latent_cdf)
    latent_coop_pdf = np.where(
        latent_pdf * p_coop_conditional > 0, latent_pdf * p_coop_conditional, 0
    )
    latent_defect_pdf = np.where(
        latent_pdf * (1 - p_coop_conditional) > 0,
        latent_pdf * (1 - p_coop_conditional),
        0,
    )

    latent_coop_cdf = np.insert(np.cumsum(latent_coop_pdf), 0, 0)
    latent_coop_cdf = latent_coop_cdf / latent_coop_cdf[-1]
    latent_defect_cdf = np.insert(np.cumsum(latent_defect_pdf), 0, 0)
    latent_defect_cdf = latent_defect_cdf / latent_defect_cdf[-1]
    return dict(
        latent_cdf=latent_cdf,
        latent_coop_cdf=latent_coop_cdf,
        latent_defect_cdf=latent_defect_cdf,
        latent_coop_pdf=latent_coop_pdf,
        latent_defect_pdf=latent_defect_pdf,
    )


def _plot_observed(data, title, filename, ylim=30):
    """Histogram of observed decision times with CDF inset."""
    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN * 0.33, FIG_WIDTH_IN * 0.28),
        constrained_layout=True,
    )

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
    ax.set_xlabel("Decision Time (sec)")
    ax.set_ylabel("Number of Participants")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Inset: scaled CDF
    inset = ax.inset_axes([0.5, 0.55, 0.38, 0.38])
    inset.set_facecolor("#f0f0f0")
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
    inset.set_box_aspect(1)
    inset.spines[["top", "right"]].set_visible(False)
    inset.set_xlim(0, 60.05)
    inset.set_ylim(0, 1.01)
    inset.set_yticks([0, 1])
    inset.set_xticks([0, 60])
    inset.set_xlabel("DT", labelpad=-5, fontsize=CI_FONT_SIZE)
    inset.set_ylabel("Scaled\nCDF", fontsize=CI_FONT_SIZE, labelpad=-5)
    inset.tick_params(labelsize=CI_FONT_SIZE)

    out = OUTPUT_FIGURES / filename
    fig.savefig(out)
    print(f"Saved to {out}")


def _plot_recovered(data, filename):
    """Recovered latent distribution with CDF inset."""
    stats = compute_latent_stats(data.decision_time.values, data.is_coop.values)

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN * 0.33, FIG_WIDTH_IN * 0.28),
        constrained_layout=True,
    )

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
    ax.set_xlabel("Decision Time (sec)")
    ax.set_ylabel("Density")
    # set ytics 0, .1, .2
    ax.set_yticks([0, 0.1, 0.2])
    ax.grid(True, linestyle=":", alpha=0.5)

    # Inset: scaled CDF
    inset = ax.inset_axes([0.5, 0.55, 0.38, 0.38])
    inset.set_facecolor("#f0f0f0")
    inset.plot(BINS, stats["latent_coop_cdf"], color=C_COOP, linewidth=1.5)
    inset.plot(BINS, stats["latent_defect_cdf"], color=C_DEFECT, linewidth=1.5)
    inset.set_box_aspect(1)
    inset.spines[["top", "right"]].set_visible(False)
    inset.set_xlim(0, 60.05)
    inset.set_ylim(0, 1.01)
    inset.set_yticks([0, 1])
    inset.set_xticks([0, 60])
    inset.set_xlabel("DT", labelpad=-5, fontsize=CI_FONT_SIZE)
    inset.set_ylabel("Scaled\nCDF", fontsize=CI_FONT_SIZE, labelpad=-5)
    inset.tick_params(labelsize=CI_FONT_SIZE)

    out = OUTPUT_FIGURES / filename
    fig.savefig(out)
    print(f"Saved to {out}")


def main():
    exp_data = load_main()
    first_movers = exp_data[exp_data.is_first_mover]

    sq_data = first_movers[first_movers.condition == "SQ"]
    rt_data = first_movers[first_movers.condition == "RT"]
    sm_data = exp_data[exp_data.condition == "SM"]

    _plot_observed(sm_data, "SM", "fig3_decision_time_sm.pdf", ylim=60)
    _plot_observed(sq_data, "SQ", "fig3_decision_time_sq.pdf")
    _plot_observed(rt_data, "RT", "fig3_decision_time_rt.pdf")
    _plot_recovered(rt_data, "fig3_decision_time_rt_recovered.pdf")


if __name__ == "__main__":
    main()

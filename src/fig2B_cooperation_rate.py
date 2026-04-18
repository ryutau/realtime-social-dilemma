import numpy as np
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


def bootstrap_ci(data, n_bootstrap=1000):
    np.random.seed(42)
    means = [
        np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)
    ]
    return np.percentile(means, [2.5, 97.5])


def compute_coop_stats(df):
    """Compute cooperation rate and 95% bootstrap CI."""
    rate = df["is_coop"].mean()
    ci_lo, ci_hi = bootstrap_ci(df["is_coop"].values)
    return rate, ci_lo, ci_hi


_BAR_STYLES = {
    "first": ("tab:orange", "white", "/"),
    "second_c": ("tab:orange", "white", "\\"),
    "second_d": ("white", "tab:orange", ""),
    "sm": ("tab:orange", "white", ""),
}


def _draw_bar(ax, x, rate, ci_lo, ci_hi, style_key):
    fc, ec, hatch = _BAR_STYLES[style_key]
    ax.bar(
        x,
        rate,
        yerr=[[rate - ci_lo], [ci_hi - rate]],
        width=0.5,
        facecolor=fc,
        edgecolor=ec,
        hatch=hatch,
        linewidth=0 if ec == "white" else 1.5,
        alpha=0.8,
        capsize=0,
        ecolor="black",
        zorder=3,
    )
    # cooperation rate
    ax.annotate(
        f"{rate*100:.1f}%",
        xy=(x, 1),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    # 95% bootstrap CI
    ax.annotate(
        f"[{ci_lo*100:.1f}~{ci_hi*100:.1f}]",
        xy=(x, 1),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=CI_FONT_SIZE,
    )


def _format_ax(ax):
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Cooperation Rate")


def plot_condition_panel(ax, exp_data, condition):
    """Plot a single condition panel (RT or SQ) with 3 bars."""
    cond_df = exp_data[exp_data.condition == condition]

    first = cond_df[cond_df.is_first_mover]
    second_c = cond_df[~cond_df.is_first_mover & cond_df.is_partner_coop]
    second_d = cond_df[~cond_df.is_first_mover & ~cond_df.is_partner_coop]

    for x, subset, key in [
        (0, first, "first"),
        (1, second_c, "second_c"),
        (2, second_d, "second_d"),
    ]:
        rate, ci_lo, ci_hi = compute_coop_stats(subset)
        _draw_bar(ax, x, rate, ci_lo, ci_hi, key)

    _format_ax(ax)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["1st mover", "2nd mover\nafter C", "2nd mover\nafter D"])
    ax.set_xlim(-0.5, 2.5)


def plot_sm_panel(ax, exp_data):
    """Plot the SM panel with a single bar."""
    sm_df = exp_data.query("condition == 'SM'")
    rate, ci_lo, ci_hi = compute_coop_stats(sm_df)
    _draw_bar(ax, 0, rate, ci_lo, ci_hi, "sm")

    _format_ax(ax)
    ax.set_xticks([0])
    ax.set_xticklabels(["Independent\ndecisions"])
    ax.set_xlim(-0.5, 0.5)


def main():
    exp_data = load_main()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.3),
        gridspec_kw={"width_ratios": [3, 3, 1]},
    )

    plot_condition_panel(axes[0], exp_data, "RT")
    plot_condition_panel(axes[1], exp_data, "SQ")
    plot_sm_panel(axes[2], exp_data)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUTPUT_FIGURES / "fig2B_cooperation_rate.pdf", bbox_inches="tight")
    print(f"Saved to {OUTPUT_FIGURES / 'fig2B_cooperation_rate.pdf'}")


if __name__ == "__main__":
    main()

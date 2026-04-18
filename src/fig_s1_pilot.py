"""Figure S1: Pilot experiment cooperation rates.

Left panel (A): Individual cooperation rate by condition and decision-making position.
Right panel (B): Pair-level mutual cooperation rate by condition.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.data import load_pilot
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
    rate = df["is_coop"].mean()
    ci_lo, ci_hi = bootstrap_ci(df["is_coop"].values)
    return rate, ci_lo, ci_hi


_BAR_STYLES = {
    "first": ("tab:orange", "white", "/"),
    "second_c": ("tab:orange", "white", "\\"),
    "second_d": ("white", "tab:orange", ""),
    "sm": ("tab:orange", "white", ""),
    "mutual": ("tab:cyan", None, ""),
}


def _draw_bar(ax, x, rate, ci_lo, ci_hi, style_key, y_anchor=1.0):
    fc, ec, hatch = _BAR_STYLES[style_key]
    ax.bar(
        x,
        rate,
        yerr=[[rate - ci_lo], [ci_hi - rate]],
        width=0.5,
        facecolor=fc,
        edgecolor=ec,
        hatch=hatch,
        linewidth=0 if ec in ("white", None) else 1.5,
        alpha=0.8,
        capsize=0,
        ecolor="black",
        zorder=3,
    )
    ax.annotate(
        f"{rate*100:.1f}%",
        xy=(x, y_anchor),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=FONT_SIZE,
    )
    ax.annotate(
        f"[{ci_lo*100:.1f}~{ci_hi*100:.1f}]",
        xy=(x, y_anchor),
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


def main():
    exp_data = load_pilot()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.3),
        gridspec_kw={"width_ratios": [1, 3, 0.3, 2]},
    )

    # --- Panel A (left): Individual cooperation rate ---

    # SM subpanel
    ax = axes[0]
    sm_df = exp_data.query("condition == 'SM'")
    rate, ci_lo, ci_hi = compute_coop_stats(sm_df)
    _draw_bar(ax, 0, rate, ci_lo, ci_hi, "sm")

    _format_ax(ax)
    ax.set_xticks([0])
    ax.set_xticklabels(["Independent\ndecisions"])
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("Cooperation Rate")
    ax.set_xlabel("in SM", fontsize=FONT_SIZE, labelpad=8)

    # RT subpanel
    ax = axes[1]
    cond_df = exp_data[exp_data.condition == "RT"]
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
    ax.set_yticklabels([])
    ax.set_xlabel("in RT", fontsize=FONT_SIZE, labelpad=8)

    # Spacer
    axes[2].set_visible(False)

    # --- Panel B (right): Mutual cooperation rate ---

    ax = axes[3]
    pair_level_df = (
        exp_data.groupby(["condition", "group_id"])["is_coop"]
        .sum()
        .eq(2)
        .reset_index(name="is_mutual_coop")
    )

    conditions = ["SM", "RT"]
    xs = [0, 1]

    for x, condition in zip(xs, conditions):
        cond_data = pair_level_df[pair_level_df.condition == condition]
        rate = cond_data.is_mutual_coop.mean()
        ci_lo, ci_hi = bootstrap_ci(cond_data.is_mutual_coop.values)
        _draw_bar(ax, x, rate, ci_lo, ci_hi, "mutual", y_anchor=1)

    _format_ax(ax)
    ax.set_xticks(xs)
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Mutual Cooperation Rate")

    # Panel labels
    fig.text(0.01, 1, "A", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")
    fig.text(0.61, 1, "B", fontsize=16, fontweight="bold", fontfamily="Arial", va="top")

    # fig.tight_layout(w_pad=1.5)
    out = OUTPUT_FIGURES / "fig_s1_pilot.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()

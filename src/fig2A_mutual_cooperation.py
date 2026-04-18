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


def main():
    exp_data = load_main()

    # Compute mutual cooperation per group (pair)
    pair_level_df = (
        exp_data.groupby(["condition", "group_id"])["is_coop"]
        .sum()
        .eq(2)
        .reset_index(name="is_mutual_coop")
    )

    conditions = ["RT", "SQ", "SM"]
    labels = ["RM\n(real time)", "SQ\n(sequential)", "SM\n(simultaneous)"]
    xs = [0, 1, 2]

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH_IN * 0.5, FIG_WIDTH_IN * 0.27),
        constrained_layout=True,
    )

    for x, condition in zip(xs, conditions):
        cond_data = pair_level_df[pair_level_df.condition == condition]
        rate = cond_data.is_mutual_coop.mean()
        ci_lo, ci_hi = bootstrap_ci(cond_data.is_mutual_coop.values)

        ax.bar(
            x,
            rate,
            yerr=[[rate - ci_lo], [ci_hi - rate]],
            width=0.5,
            facecolor="tab:cyan",
            edgecolor=None,
            linewidth=1.5,
            alpha=0.8,
            capsize=0,
            ecolor="black",
            zorder=3,
        )
        ax.annotate(
            f"{rate*100:.1f}%",
            xy=(x, 0.8),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE,
        )
        ax.annotate(
            f"[{ci_lo*100:.1f}~{ci_hi*100:.1f}]",
            xy=(x, 0.8),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=CI_FONT_SIZE,
        )

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Mutual Cooperation Rate")

    fig.savefig(OUTPUT_FIGURES / "fig2A_mutual_cooperation.pdf")
    print(f"Saved to {OUTPUT_FIGURES / 'fig2A_mutual_cooperation.pdf'}")


if __name__ == "__main__":
    main()

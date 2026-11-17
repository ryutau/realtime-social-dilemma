"""Figure 2D: individual cooperation among all eligible SM participants."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import BOOTSTRAP, SEED, bootstrap_means, interval, load_main
from src.plotting import style


PROJECT = Path(__file__).resolve().parents[2]

def main():
    data = load_main()  # Applies post-game completion exclusions.
    # Keep all eligible individuals, including the completed partner in the incomplete SM pair.
    cooperation = data.loc[data.condition.eq("SM"), "is_coop"]
    estimate = cooperation.mean()
    draws = bootstrap_means(cooperation, BOOTSTRAP, SEED, "individual/SM")
    lower, upper = interval(draws)
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.94)
    ax.bar(
        0, estimate, width=0.56,
        facecolor="tab:orange", edgecolor="white",
        linewidth=0,
        alpha=0.85, zorder=3,
    )
    ax.vlines(0, lower, upper, color="black", linewidth=1.5, zorder=4)
    ax.text(0, 1.05, f"{100 * estimate:.1f}%", ha="center", va="bottom", fontsize=17)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    ax.set_ylabel("Cooperation rate", fontsize=17)
    ax.set_xlim(-1.65, 1.65)
    ax.set_xticks([0], ["Independent\ndecisions"], fontsize=14, ha="center", multialignment="center")
    output = PROJECT / "output/main/figures/figure2D.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure2 panel D", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

"""Figure 2A: mutual cooperation rates in complete RT, SQ, and SM pairs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import BOOTSTRAP, SEED, bootstrap_means, interval, load_main
from src.plotting import style

PROJECT = Path(__file__).resolve().parents[2]

def main():
    data = load_main()  # Applies post-game completion exclusions.
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.94)
    for position, condition in enumerate(("RT", "SQ", "SM")):
        members = data.loc[data.condition.eq(condition)]
        pair_sizes = members.groupby("group_id").label.transform("size")
        # One row per complete original pair; the single remaining SM partner is excluded.
        first = members.loc[pair_sizes.eq(2) & members.is_first_mover]
        mutual = first.is_coop & first.is_partner_coop
        estimate = mutual.mean()
        draws = bootstrap_means(mutual, BOOTSTRAP, SEED, f"pair/{condition}")
        lower, upper = interval(draws)
        ax.bar(
            position, estimate, width=0.56,
            facecolor="tab:cyan", edgecolor="none", hatch="",
            linewidth=0,
            alpha=0.85, zorder=3,
        )
        ax.vlines(position, lower, upper, color="black", linewidth=1.5, zorder=4)
        ax.text(position, 1.05, f"{100 * estimate:.1f}%", ha="center", va="bottom", fontsize=17)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    ax.set_ylabel("Mutual cooperation rate", fontsize=17)
    ax.set_xticks(range(3), ["RT\n(real time)", "SQ\n(sequential)", "SM\n(simultaneous)"],
                  fontsize=14, ha="center", multialignment="center")
    ax.set_xlim(-0.4, 2.4)
    output = PROJECT / "output/main/figures/figure2A.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure2 panel A", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

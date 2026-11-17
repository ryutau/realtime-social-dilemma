"""Figure 2B: RT first-mover and conditional second-mover cooperation rates."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import BOOTSTRAP, SEED, bootstrap_means, interval, load_main
from src.plotting import style

BAR_STYLES = {
    "first_mover": ("tab:orange", "white", "/"),
    "second_after_C": ("tab:orange", "white", "\\"),
    "second_after_D": ("white", "tab:orange", ""),
}

PROJECT = Path(__file__).resolve().parents[2]

def main():
    data = load_main()  # Applies post-game completion exclusions.
    rt = data.loc[data.condition.eq("RT")]
    first = rt.loc[rt.is_first_mover]
    second = rt.loc[~rt.is_first_mover]
    groups = [
        ("first_mover", first.is_coop, "pair/RT"),
        ("second_after_C", second.loc[second.is_partner_coop, "is_coop"], "conditional/RT/after_C"),
        ("second_after_D", second.loc[~second.is_partner_coop, "is_coop"], "conditional/RT/after_D"),
    ]
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.94)
    for position, (group, values, stream) in enumerate(groups):
        estimate = values.mean()
        draws = bootstrap_means(values, BOOTSTRAP, SEED, stream)
        lower, upper = interval(draws)
        face, edge, hatch = BAR_STYLES[group]
        ax.bar(
            position, estimate, width=0.56,
            facecolor=face, edgecolor=edge, hatch=hatch,
            linewidth=1.3 if edge == "tab:orange" else 0,
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
    ax.set_ylabel("Cooperation rate", fontsize=17)
    labels = ["Voluntary\n1st mover", "2nd mover\nafter C", "2nd mover\nafter D"]
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks(range(len(groups)), labels, fontsize=14, ha="center", multialignment="center")
    output = PROJECT / "output/main/figures/figure2B.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure2 panel B", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

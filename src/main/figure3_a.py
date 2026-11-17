"""Figure 3A: observed decision times of RT first movers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import load_main
from src.plotting import style, C_COOP, C_DEFECT


PROJECT = Path(__file__).resolve().parents[2]

def main():
    data = load_main()  # Applies post-game completion exclusions.
    first = data.loc[data.condition.eq("RT") & data.is_first_mover]
    bins = np.arange(0, 62, 2)
    style()
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.20, top=0.96)
    upper = 60
    inset = ax.inset_axes([0.53, 0.55, 0.41, 0.40])
    inset.set_facecolor("#f0f0f0")
    inset.spines[["top", "right"]].set_visible(False)
    inset.set(xlim=(0, upper), ylim=(0, 1.02), xticks=[0, upper], yticks=[0, 1])
    inset.set_xlabel("Time (s)", fontsize=13, labelpad=-1)
    inset.set_ylabel("Scaled CDF", fontsize=13, labelpad=-1)
    inset.tick_params(labelsize=12, length=2)
    for cooperate, color in ((True, C_COOP), (False, C_DEFECT)):
        times = first.loc[first.is_coop.eq(cooperate), "decision_time"].to_numpy()
        counts, _ = np.histogram(times, bins=bins)
        cumulative = np.cumsum(counts) / len(times)
        ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge", color=color, alpha=0.75)
        inset.step(bins, np.r_[0, cumulative], where="post", color=color, linewidth=2.6)
    ax.set(xlim=(0, upper), xlabel="Decision time (s)", ylabel="Number of participants")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(0, 30)
    output = PROJECT / "output/main/figures/figure3A.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure3 panel A", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

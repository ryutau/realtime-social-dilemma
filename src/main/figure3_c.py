"""Figure 3C: potential RT decision times recovered from observed first movers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.common import load_main
from src.plotting import style, C_COOP, C_DEFECT


PROJECT = Path(__file__).resolve().parents[2]

def main():
    data = load_main()  # Applies post-game completion exclusions.
    first = data.loc[data.condition.eq("RT") & data.is_first_mover]
    times = first.decision_time.to_numpy()
    cooperate = first.is_coop.to_numpy()
    bins = np.arange(0, 62, 2)
    if not len(times) or not np.all((times > 0) & (times < 60)):
        raise ValueError("Recovered RT first-mover times must be strictly within (0,60)")
    if np.isclose(times[:, None], bins[None, :], atol=1e-10, rtol=0).any():
        raise ValueError("Exact 2-second boundary requires reviewing the original bin/CDF convention")

    # Recover the individual potential-time CDF from the observed minimum of two times.
    observed_cdf = np.searchsorted(np.sort(times), bins, side="right") / len(times)
    potential_cdf = 1 - np.sqrt(1 - observed_cdf)
    total_count, _ = np.histogram(times, bins=bins)
    coop_count, _ = np.histogram(times[cooperate], bins=bins)
    cooperation_share = np.divide(coop_count, total_count,
                                  out=np.zeros_like(coop_count, dtype=float), where=total_count > 0)
    coop_mass = np.diff(potential_cdf) * cooperation_share
    defect_mass = np.diff(potential_cdf) * (1 - cooperation_share)
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
    for mass, color in ((coop_mass, C_COOP), (defect_mass, C_DEFECT)):
        cumulative = np.r_[0, np.cumsum(mass)] / mass.sum() if mass.sum() else np.zeros(len(bins))
        ax.bar(bins[:-1], mass, width=np.diff(bins), align="edge", color=color, alpha=0.75)
        inset.plot(bins, cumulative, color=color, linewidth=2.6)
    ax.set(xlim=(0, upper), xlabel="Decision time (s)", ylabel="Probability")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.set_ylim(bottom=0)
    output = PROJECT / "output/main/figures/figure3C.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure3 panel C", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

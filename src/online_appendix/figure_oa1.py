"""Main-experiment decision-time distributions, observed and recovered (Figure OA1)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from src.common import load_main
from src.plotting import style, panel_label, C_COOP, C_DEFECT

PROJECT = Path(__file__).resolve().parents[2]


def main():
    data = load_main()  # Applies post-game completion exclusions.
    pair_sizes = data.groupby(["condition", "group_id"]).label.transform("size")
    sm = data.condition.eq("SM")
    rt_first = data.loc[data.condition.eq("RT") & data.is_first_mover]
    samples = {
        "A": data.loc[sm & data.is_first_mover & pair_sizes.eq(2)],
        "B": data.loc[sm],  # All SM participants, including a completed partner left alone.
        "C": rt_first,
        "D": rt_first,
    }
    style()
    fig = plt.figure(figsize=(10.4, 10.2))
    subfigures = fig.subfigures(2, 2, wspace=0.02, hspace=0.02)
    for panel, subfigure in zip("ABCD", subfigures.flat):
        population = samples[panel]
        times = population.decision_time.to_numpy(dtype=float)
        cooperate = population.is_coop.to_numpy()
        recovered = panel == "D"
        if not len(times) or not np.isfinite(times).all() or (times < 0).any():
            raise ValueError("Decision times must be nonempty, finite, and nonnegative")
        # Observed bins include decisions above 60 s; recovered RT times lie within 0–60 s.
        upper = 60.0 if recovered else max(60.0, np.ceil(times.max() / 2) * 2)
        bins = np.arange(0, upper + 1, 2)
        coop_counts = np.histogram(times[cooperate], bins=bins)[0]
        defect_counts = np.histogram(times[~cooperate], bins=bins)[0]
        if recovered:
            if not np.all((times > 0) & (times < 60)):
                raise ValueError("Recovered RT first-mover times must be strictly within (0,60)")
            if np.isclose(times[:, None], bins[None, :], atol=1e-10, rtol=0).any():
                raise ValueError("Exact 2-second boundary requires reviewing the bin/CDF convention")
            # Recover the potential time CDF from the observed minimum of each RT pair.
            observed_cdf = np.searchsorted(np.sort(times), bins, side="right") / len(times)
            potential_cdf = 1 - np.sqrt(1 - observed_cdf)
            total_counts = coop_counts + defect_counts
            coop_share = np.divide(coop_counts, total_counts,
                                   out=np.zeros_like(coop_counts, dtype=float), where=total_counts > 0)
            # Split each time bin's probability using its observed cooperation share.
            heights_by_action = (np.diff(potential_cdf) * coop_share,
                                 np.diff(potential_cdf) * (1 - coop_share))
        else:
            heights_by_action = (coop_counts, defect_counts)
        panel_label(subfigure, panel)
        ax = subfigure.add_axes([0.20, 0.18, 0.75, 0.72])
        inset = ax.inset_axes([0.53, 0.55, 0.41, 0.40])
        inset.set_facecolor("#f0f0f0")
        inset.spines[["top", "right"]].set_visible(False)
        inset.set(xlim=(0, upper), ylim=(0, 1.02), xticks=[0, upper], yticks=[0, 1])
        inset.set_xlabel("Time (s)", fontsize=13, labelpad=-1)
        inset.set_ylabel("Scaled CDF", fontsize=13, labelpad=-1)
        inset.tick_params(labelsize=12, length=2)
        for heights, color in zip(heights_by_action, (C_COOP, C_DEFECT)):
            # Scale each action's CDF separately to end at one.
            cumulative = np.cumsum(heights) / heights.sum() if heights.sum() else np.zeros(len(heights))
            ax.bar(bins[:-1], heights, width=np.diff(bins), align="edge", color=color, alpha=0.75)
            if recovered:
                inset.plot(bins, np.r_[0, cumulative], color=color, linewidth=2.6)
            else:
                inset.step(bins, np.r_[0, cumulative], where="post", color=color, linewidth=2.6)
        ax.set(xlim=(0, upper), xlabel="Decision time (s)",
               ylabel="Probability" if recovered else "Number of participants")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.margins(y=0.10)
        ax.set_ylim(bottom=0)
        if not recovered:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    output = PROJECT / "output/online_appendix/figures/figure_oa1.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package",
                "Title": "Figure OA1", "CreationDate": None, "ModDate": None}
    fig.savefig(output, format="pdf", bbox_inches=None, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

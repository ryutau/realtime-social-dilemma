"""Estimate and plot potential RT versus observed SQ first-mover cooperation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from matplotlib.mlab import GaussianKDE

from src.common import BINS, BOOTSTRAP, SEED, bootstrap_means, interval, load_main, rng
from src.plotting import style

COLORS = {"RT": "C1", "SQ": "C0"}

def style_axes(ax, grid=False):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="0.85", lw=0.65, alpha=0.7)
    ax.tick_params(length=3, width=0.7, labelsize=14, colors="black")

PROJECT = Path(__file__).resolve().parents[2]

def main():
    # Completion exclusions are applied when loading the main experiment.
    data = load_main()
    statistics, draws = first_mover_statistics(data)
    differences = 100 * draws["RT_minus_SQ"]
    grid = np.linspace(differences.min(), differences.max(), 512)
    density = GaussianKDE(differences, bw_method="scott").evaluate(grid)
    style()
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_axes([0.20, 0.21, 0.59, 0.68])
    sq_x, rt_x, anchor = 0.12, 1.32, 2.52
    sq, rt = 100 * statistics.loc["SQ", "estimate"], 100 * statistics.loc["RT", "estimate"]
    gap = 100 * statistics.loc["RT_minus_SQ", "estimate"]
    low, high = 100 * statistics.loc["RT_minus_SQ", ["ci_low", "ci_high"]].to_numpy()
    ymin = min(20, 5 * np.floor((sq + grid.min() - 2) / 5))
    ymax = max(90, 5 * np.ceil((sq + grid.max() + 2) / 5))
    ax.set(xlim=(-0.45, 3.10), ylim=(ymin, ymax), ylabel="Cooperation", yticks=[20, 40, 60, 80])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.hlines(sq, sq_x, 3.10, color="black", alpha=0.45, lw=0.9, ls=(0, (3, 2)), zorder=1)
    ax.hlines(rt, rt_x, 3.10, color="black", alpha=0.35, lw=0.85, zorder=1)
    for position, condition, marker in [(sq_x, "SQ", "s"), (rt_x, "RT", "o")]:
        estimate, lo, hi = 100 * statistics.loc[condition, ["estimate", "ci_low", "ci_high"]].to_numpy()
        ax.errorbar(position, estimate, yerr=[[estimate - lo], [hi - estimate]],
                    fmt=marker, color=COLORS[condition], capsize=3.5, elinewidth=1.65,
                    markersize=9, markeredgewidth=0.7, markeredgecolor="white", zorder=5)
        label_y, label_va = (hi + 3, "bottom") if condition == "SQ" else (lo - 3, "top")
        ax.text(position, label_y, f"{estimate:.1f}%", color="black", ha="center",
                va=label_va, fontsize=17,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3}, zorder=6)
    halfwidth = 0.48 * density / density.max()
    ax.fill_betweenx(sq + grid, anchor, anchor + halfwidth, color="black", alpha=0.15, lw=0, zorder=2)
    ax.plot(anchor + halfwidth, sq + grid, color="black", alpha=0.6, lw=0.7, zorder=2)
    ax.plot([anchor, anchor], sq + np.array([low, high]), color="black", lw=2.8,
            solid_capstyle="round", zorder=4)
    ax.plot(anchor, sq + gap, "o", ms=9, color="black", mec="white", mew=0.8, zorder=5)
    ax.set_xticks([sq_x, rt_x, anchor], ["SQ\n(observed)", "RT\n(potential)", "RT − SQ"],
                  ha="center", multialignment="center")
    ax.tick_params(axis="x", length=0, pad=8)
    ax.spines["bottom"].set_visible(False)
    style_axes(ax)
    difference = ax.secondary_yaxis("right", functions=(lambda y: y - sq, lambda d: d + sq))
    difference.spines["right"].set_visible(True)
    difference.spines["right"].set_bounds(-40, 20)
    difference.set_yticks([-40, -20, 0, 20])
    difference.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace("-", "−")))
    difference.set_ylabel("Difference (pp)", color="black", labelpad=6, fontsize=17)
    difference.tick_params(length=3, width=0.7, labelsize=14, colors="black", pad=3)
    for label, tick in zip(difference.get_yticklabels(), [-40, -20, 0, 20]):
        if tick == 0:
            label.set_fontweight("bold")
            label.set_color("black")
    fig.canvas.draw()
    assert np.isclose(ax.transData.transform((0, sq))[1], difference.transData.transform((0, 0))[1])
    assert np.isclose(ax.transData.transform((0, rt))[1], difference.transData.transform((0, gap))[1])

    output = PROJECT / "output/main/figures/figure4A.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure4 panel A", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


def first_mover_statistics(data):
    # Each eligible RT/SQ pair contributes its first mover.
    rt = data.loc[data.condition.eq("RT") & data.is_first_mover].sort_values("group_id", kind="stable")
    sq = data.loc[data.condition.eq("SQ") & data.is_first_mover].sort_values("group_id", kind="stable")
    times = rt.decision_time.to_numpy()
    total = np.histogram(times, BINS)[0]
    coop = np.histogram(times[rt.is_coop.to_numpy()], BINS)[0]
    observed = np.searchsorted(np.sort(times), BINS, side="right") / len(times)
    # Recover the potential time distribution from the observed minimum of two times.
    latent = 1 - np.sqrt(1 - observed)
    conditional = np.divide(coop, total, out=np.zeros_like(coop, dtype=float), where=total > 0)
    coop_mass = (np.diff(latent) * conditional).sum()
    defect_mass = (np.diff(latent) * (1 - conditional)).sum()
    point = {"SQ": float(sq.is_coop.mean()), "RT": float(coop_mass / (coop_mass + defect_mass))}
    draws = {
        "SQ": bootstrap_means(sq.is_coop.to_numpy(), BOOTSTRAP, SEED, "pair/SQ"),
        "RT": bootstrap_latent(rt.decision_time.to_numpy(), rt.is_coop.to_numpy(),
                               BOOTSTRAP, SEED, "pair/RT"),
    }
    point["RT_minus_SQ"] = point["RT"] - point["SQ"]
    draws["RT_minus_SQ"] = draws["RT"] - draws["SQ"]
    rows = []
    for quantity in ("SQ", "RT", "RT_minus_SQ"):
        low, high = interval(draws[quantity])
        rows.append({"quantity": quantity, "estimate": point[quantity],
                     "ci_low": float(low), "ci_high": float(high)})
    return pd.DataFrame(rows).set_index("quantity"), draws


def bootstrap_latent(times, cooperate, repetitions, seed, stream):
    times, cooperate = np.asarray(times), np.asarray(cooperate, dtype=float)
    if len(times) == 0 or not np.all((times > 0) & (times < 60)):
        raise ValueError("Recovered RT first-mover times must be strictly within (0,60)")
    if np.isclose(times[:, None], BINS[None, :], atol=1e-10, rtol=0).any():
        raise ValueError("Exact 2-second boundary requires reviewing the original bin/CDF convention")
    bins = np.digitize(times, BINS) - 1
    random = rng(seed, stream)
    draws = np.empty(repetitions)
    for start in range(0, repetitions, 2000):
        stop = min(start + 2000, repetitions)
        size = stop - start
        indices = random.integers(len(times), size=(size, len(times)))
        total = np.zeros((size, len(BINS) - 1), dtype=float)
        coop = np.zeros_like(total)
        rows = np.repeat(np.arange(size), len(times))
        np.add.at(total, (rows, bins[indices].ravel()), 1)
        np.add.at(coop, (rows, bins[indices].ravel()), cooperate[indices].ravel())
        latent = 1 - np.sqrt(np.clip(1 - total.cumsum(axis=1) / len(times), 0, 1))
        mass = np.diff(latent, prepend=np.zeros((size, 1)), axis=1)
        conditional = np.divide(coop, total, out=np.zeros_like(coop), where=total > 0)
        draws[start:stop] = (mass * conditional).sum(axis=1)
    return draws


if __name__ == "__main__":
    main()

"""Estimate and plot pilot cooperation rates and their contrast in Figure OA2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.mlab import GaussianKDE
from matplotlib.ticker import FuncFormatter

from src.common import SEED, BOOTSTRAP, bootstrap_means, interval, load_pilot
from src.plotting import style, panel_label

CONDITIONS = ("SM", "RT")
GRID_POINTS = 512
BANDWIDTH_SCALE = 1.5


def calculate_statistics(data, seed, bootstrap):
    records, draws = [], {}

    def add(condition, panel, group, values, stream, sampling_unit):
        values = np.asarray(values, dtype=float)
        if not len(values):
            raise ValueError(f"Empty pilot subgroup: {condition}, {group}")
        samples = bootstrap_means(values, bootstrap, seed, stream)
        lower, upper = interval(samples)
        records.append({
            "condition": condition,
            "panel": panel,
            "group": group,
            "n": len(values),
            "n_cooperating": int(values.sum()),
            "estimate": float(values.mean()),
            "ci_low": float(lower),
            "ci_high": float(upper),
            "sampling_unit": sampling_unit,
            "stream": stream,
        })
        draws[f"{condition}_{group}"] = samples

    for condition in CONDITIONS:
        members = data.loc[data.condition.eq(condition)]
        # One first-mover row represents each complete pilot pair.
        pairs = members.loc[members.is_first_mover].sort_values(
            "group_id", kind="stable").reset_index(drop=True)
        mutual = pairs.is_coop & pairs.is_partner_coop
        add(condition, "B", "mutual_cooperation", mutual,
            f"pilot/pair/{condition}", "pairs within condition")

        if condition == "SM":
            add(condition, "A", "individual", members.is_coop,
                "pilot/individual/SM", "individuals within SM")
            continue

        add(condition, "A", "first_mover", pairs.is_coop,
            "pilot/pair/RT", "pairs within RT; one first mover per pair")
        second = members.loc[~members.is_first_mover]
        for partner_cooperated, suffix in [(True, "C"), (False, "D")]:
            values = second.loc[second.is_partner_coop.eq(partner_cooperated), "is_coop"]
            add(condition, "A", f"second_after_{suffix}", values,
                f"pilot/conditional/RT/after_{suffix}",
                f"individuals within fixed RT second-mover/partner-{suffix} subset")

    return pd.DataFrame.from_records(records), draws

def derive_contrast(statistics, rt, sm):
    """Subtract matched independent-condition draws; do not generate new draws."""
    points = {}
    for condition in ("RT", "SM"):
        selected = statistics.loc[
            statistics.condition.eq(condition)
            & statistics.panel.eq("B")
            & statistics["group"].eq("mutual_cooperation")
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one Figure OA2B mutual-cooperation estimate for {condition}")
        point = float(selected.iloc[0].estimate)
        if not np.isfinite(point) or not 0 <= point <= 1:
            raise ValueError(f"Figure OA2B {condition} estimate must be a probability")
        points[condition] = point

    differences = rt - sm
    low, high = interval(differences)
    difference_pp = 100 * differences
    variance = float(np.var(difference_pp, ddof=1))
    if len(differences) < 2 or not np.isfinite(variance) or variance <= 0:
        raise ValueError("RT-minus-SM draws must have positive finite variance for the Gaussian KDE")
    grid = np.linspace(difference_pp.min(), difference_pp.max(), GRID_POINTS)
    # Univariate Scott factor, widened to suppress small-sample lattice ripples.
    bandwidth_factor = BANDWIDTH_SCALE * len(difference_pp) ** (-1 / 5)
    density = GaussianKDE(difference_pp, bw_method=bandwidth_factor).evaluate(grid)
    if not np.isfinite(density).all() or density.max() <= 0:
        raise ValueError("The RT-minus-SM KDE is invalid; inspect the Figure OA2 bootstrap draws")
    result = pd.DataFrame([{
        "quantity": "RT_minus_SM",
        "estimate": points["RT"] - points["SM"],
        "ci_low": float(low),
        "ci_high": float(high),
        "bootstrap_mean": float(differences.mean()),
        "bootstrap_sd": float(differences.std(ddof=1)),
    }])
    violin = pd.DataFrame({"difference_pp": grid, "density": density})
    return result, violin, differences


BAR_STYLES = {
    "mutual_cooperation": ("tab:cyan", "none", ""),
    "first_mover": ("tab:orange", "white", "/"),
    "second_after_C": ("tab:orange", "white", "\\"),
    "second_after_D": ("white", "tab:orange", ""),
    "individual": ("tab:orange", "white", ""),
}

def draw_bar(ax, position, row):
    face, edge, hatch = BAR_STYLES[row["group"]]
    estimate, lower, upper = row.estimate, row.ci_low, row.ci_high
    ax.bar(
        position, estimate, width=0.56,
        facecolor=face, edgecolor=edge, hatch=hatch,
        linewidth=1.3 if edge == "tab:orange" else 0,
        alpha=0.85, zorder=3,
    )
    ax.vlines(position, lower, upper, color="black", linewidth=1.5, zorder=4)
    ax.text(position, 1.05, f"{100 * estimate:.1f}%", ha="center", va="bottom", fontsize=17)

def format_axis(ax, ylabel=None):
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_bounds(0, 1)
    ax.tick_params(axis="x", length=0, pad=7, labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.91", linewidth=0.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=17)

INDIVIDUAL_GROUPS = (
    ("SM", "individual"), ("RT", "first_mover"),
    ("RT", "second_after_C"), ("RT", "second_after_D"),
)

PROJECT = Path(__file__).resolve().parents[2]


def main():
    # Loading excludes incomplete participants and their partners.
    data = load_pilot()
    statistics, draws = calculate_statistics(data, SEED, BOOTSTRAP)
    contrast, violin, _ = derive_contrast(
        statistics, draws["RT_mutual_cooperation"], draws["SM_mutual_cooperation"])
    style()
    fig = plt.figure(figsize=(14.3, 5.1))
    individual, mutual = fig.subfigures(1, 2, width_ratios=[7.8, 6.2], wspace=0.025)
    for panel, letter in zip((individual, mutual), "AB"):
        panel_label(panel, letter)

    # A: individual cooperation rates.
    grid = individual.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.30,
                           left=0.11, right=0.97, bottom=0.25, top=0.87)
    sm_ax, rt_ax = individual.add_subplot(grid[0, 0]), individual.add_subplot(grid[0, 1])
    for index, (condition, group) in enumerate(INDIVIDUAL_GROUPS):
        row = statistics.loc[
            statistics.condition.eq(condition) & statistics["group"].eq(group)
        ].iloc[0]
        draw_bar(sm_ax if condition == "SM" else rt_ax, 0 if index == 0 else index - 1, row)
    for ax, condition in ((sm_ax, "SM"), (rt_ax, "RT")):
        format_axis(ax, "Cooperation rate" if condition == "SM" else None)
        ax.grid(False)
        ax.text(0.5, -0.27, f"in {condition}", transform=ax.transAxes,
                ha="center", va="top", fontsize=15)
    sm_ax.set_xticks([0], ["Independent\ndecisions"], fontsize=14)
    sm_ax.set_xlim(-0.5, 0.5)
    rt_ax.set_xticks(range(3), ["1st mover", "2nd mover\nafter C", "2nd mover\nafter D"],
                     fontsize=14, ha="center", multialignment="center")
    rt_ax.set_xlim(-0.5, 2.5)
    rt_ax.tick_params(axis="y", labelleft=False)

    # B: mutual cooperation and the RT-minus-SM contrast.
    ax = mutual.add_axes((0.14, 0.25, 0.68, 0.62))
    for position, condition in enumerate(CONDITIONS):
        row = statistics.loc[
            statistics.condition.eq(condition) & statistics.panel.eq("B")
        ].iloc[0]
        draw_bar(ax, position, row)
    format_axis(ax, "Mutual cooperation rate")
    ax.grid(False)
    sm = float(statistics.loc[statistics.condition.eq("SM") & statistics.panel.eq("B"), "estimate"].iloc[0])
    rt = float(statistics.loc[statistics.condition.eq("RT") & statistics.panel.eq("B"), "estimate"].iloc[0])
    gap, low, high = contrast.loc[0, ["estimate", "ci_low", "ci_high"]].to_numpy(dtype=float)
    differences = violin.difference_pp.to_numpy() / 100
    density = violin.density.to_numpy()
    anchor = 2.12
    halfwidth = 0.45 * density / density.max()
    ax.fill_betweenx(sm + differences, anchor, anchor + halfwidth,
                     color="black", alpha=0.15, linewidth=0, zorder=2)
    ax.plot(anchor + halfwidth, sm + differences, color="black", alpha=0.6, linewidth=0.7, zorder=2)
    ax.vlines(anchor, sm + low, sm + high, color="black", linewidth=2.8, zorder=4)
    ax.plot(anchor, sm + gap, "o", markersize=9, color="black", markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)
    ax.hlines(sm, 0.3, 2.76, color="black", alpha=0.45, linewidth=0.9,
              linestyle=(0, (3, 2)), zorder=1)
    ax.hlines(rt, 1.3, 2.76, color="black", alpha=0.35, linewidth=0.85, zorder=1)
    upper = max(1.14, np.ceil((sm + differences.max() + 0.04) / 0.05) * 0.05)
    ax.set(xlim=(-0.55, 2.76), ylim=(min(0, sm + differences.min() - 0.04), upper))
    ax.set_xticks([0, 1, anchor], ["SM", "RT", "RT − SM"],
                  fontsize=14, ha="center", multialignment="center")
    difference_axis = ax.secondary_yaxis("right", functions=(lambda rate: 100 * (rate - sm),
                                                             lambda pp: sm + pp / 100))
    difference_axis.spines["right"].set_visible(True)
    difference_axis.spines["right"].set_bounds(-40, 60)
    difference_axis.set_yticks([-40, -20, 0, 20, 40, 60])
    difference_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace("-", "−")))
    difference_axis.set_ylabel("Difference (pp)", labelpad=6, fontsize=17)
    difference_axis.tick_params(length=3, width=0.7, labelsize=14, pad=3)
    for label, value in zip(difference_axis.get_yticklabels(), [-40, -20, 0, 20, 40, 60]):
        if value == 0:
            label.set_fontweight("bold")
    fig.canvas.draw()
    assert np.isclose(ax.transData.transform((0, sm))[1], difference_axis.transData.transform((0, 0))[1])
    assert np.isclose(ax.transData.transform((0, rt))[1], difference_axis.transData.transform((0, 100 * gap))[1])

    output = PROJECT / "output/online_appendix/figures/figure_oa2.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package", "Title": "Figure OA2",
                "CreationDate": None, "ModDate": None}
    fig.savefig(output, format="pdf", bbox_inches=None, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

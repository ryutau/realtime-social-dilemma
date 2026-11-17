"""Estimate and plot the SM reconstruction diagnostics in Figure A1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import SEED, BOOTSTRAP, interval, load_main, rng
from src.plotting import style, panel_label

PROJECT = Path(__file__).resolve().parents[2]


WIDTHS = (1, 2, 5)
GRID_END = 60
BATCH_SIZE = 1000


def prepare_sample(data):
    """Keep the original SM pairs and verify the nominal earlier decision."""
    if "session_no" not in data or data.session_no.isna().any():
        raise ValueError("SM validation requires a session_no for every observation")
    sm = data.loc[data.condition.eq("SM")]
    pair_sizes = sm.groupby("group_id")["label"].transform("size")
    sm = sm.loc[pair_sizes.eq(2)].sort_values(
        ["session_no", "group_id", "label"], kind="stable")
    if sm.empty or not np.isfinite(sm.decision_time).all() or (sm.decision_time <= 0).any():
        raise ValueError("SM decision times must be finite and positive")
    if not sm.groupby("group_id").session_no.nunique().eq(1).all():
        raise ValueError("An original SM pair crosses session boundaries")
    grouped = list(sm.groupby(["session_no", "group_id"], sort=True))
    if any(len(group) != 2 for _, group in grouped):
        raise ValueError("Every original SM pair must contain two participants")
    times = np.array([group.decision_time.to_numpy(float) for _, group in grouped])
    choices = np.array([group.is_coop.to_numpy(bool) for _, group in grouped])
    flags = np.array([group.is_first_mover.to_numpy(bool) for _, group in grouped])
    if np.any(times[:, 0] == times[:, 1]):
        raise ValueError("An exact within-pair timing tie requires an explicit tie rule")
    first_index = times.argmin(axis=1)
    expected = np.arange(2)[None, :] == first_index[:, None]
    if not np.array_equal(flags, expected):
        raise ValueError("SM first-mover flags disagree with the earlier original decision")
    first_times = times[np.arange(len(times)), first_index]
    first_choices = choices[np.arange(len(times)), first_index]
    if not np.all((first_times > 0) & (first_times < GRID_END)):
        raise ValueError("Nominal first decisions must lie strictly inside the 0–60 s recovery grid")
    for width in WIDTHS:
        grid = np.arange(0, GRID_END + width, width)
        if np.isclose(first_times[:, None], grid[None, :], atol=1e-10, rtol=0).any():
            raise ValueError("An exact bin-edge first decision requires reviewing the CDF/bin convention")
    return {
        "times": times,
        "choices": choices,
        "first_times": first_times,
        "first_choices": first_choices,
        "pair_sessions": np.array([str(key[0]) for key, _ in grouped]),
    }

def individual_cdfs(sample):
    """Evaluate both right-continuous CDFs at every SM decision-time jump."""
    full_times = np.sort(sample["times"].ravel())
    first_times = np.sort(sample["first_times"])
    plot_end = float(np.ceil(full_times[-1] / 5) * 5)
    grid = np.unique(np.r_[0.0, full_times, plot_end])
    empirical = np.searchsorted(full_times, grid, side="right") / len(full_times)
    first = np.searchsorted(first_times, grid, side="right") / len(first_times)
    reconstructed = 1 - np.sqrt(1 - first)
    return pd.DataFrame({
        "time_seconds": grid,
        "empirical_all_sm_cdf": empirical,
        "observed_nominal_first_cdf": first,
        "reconstructed_potential_cdf": reconstructed,
        "reconstructed_minus_empirical": reconstructed - empirical,
        "absolute_difference": np.abs(reconstructed - empirical),
    })

def reconstructed_rates(weights, first_times, first_choices, width):
    """Recover cooperation rates from weighted nominal first-mover decisions."""
    grid = np.arange(0, GRID_END + width, width)
    bin_ids = np.digitize(first_times, grid)
    membership = (bin_ids[:, None] == np.arange(1, len(grid))[None, :]).astype(float)
    cdf = weights @ (first_times[:, None] <= grid[None, :]).astype(float)
    cdf /= weights.sum(axis=1, keepdims=True)
    latent_mass = np.diff(1 - np.sqrt(np.clip(1 - cdf, 0, 1)), axis=1)
    counts = weights @ membership
    cooperators = weights @ (membership * first_choices[:, None])
    conditional = np.divide(
        cooperators, counts, out=np.full_like(cooperators, np.nan), where=counts > 0)
    coop_mass = np.where(latent_mass * conditional > 0, latent_mass * conditional, 0)
    defect_mass = np.where(
        latent_mass * (1 - conditional) > 0, latent_mass * (1 - conditional), 0)
    denominator = (coop_mass + defect_mass).sum(axis=1)
    if np.any(denominator <= 0):
        raise ValueError("Reconstruction has no recoverable probability mass")
    return coop_mass.sum(axis=1) / denominator

def bootstrap_reconstruction(sample, repetitions, seed):
    """Resample original pairs independently within each observed session."""
    pair_sessions = sample["pair_sessions"]
    npairs = len(pair_sessions)
    if npairs > np.iinfo(np.uint16).max:
        raise ValueError("Too many SM pairs for the resampling-count dtype")
    sessions = [
        (np.flatnonzero(pair_sessions == session), rng(seed, f"sm_validation/session/{session}"))
        for session in np.unique(pair_sessions)
    ]
    counts = np.zeros((repetitions, npairs), dtype=np.uint16)
    full = np.empty(repetitions)
    recovered = np.empty((repetitions, len(WIDTHS)))
    pair_cooperators = sample["choices"].sum(axis=1).astype(float)
    for start in range(0, repetitions, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, repetitions)
        weights = counts[start:stop]
        rows = np.arange(stop - start)[:, None]
        for indices, random in sessions:
            chosen = indices[random.integers(len(indices), size=(stop - start, len(indices)))]
            np.add.at(weights, (rows, chosen), 1)
        full[start:stop] = weights @ pair_cooperators / (2 * npairs)
        for column, width in enumerate(WIDTHS):
            recovered[start:stop, column] = reconstructed_rates(
                weights, sample["first_times"], sample["first_choices"], width)
    return {
        "recovered_rate": recovered,
        "recovery_error": recovered - full[:, None],
    }

def rate_statistics(sample, draws):
    full_rate = float(sample["choices"].mean())
    weights = np.ones((1, len(sample["first_times"])))
    records = []
    for column, width in enumerate(WIDTHS):
        rate = float(reconstructed_rates(
            weights, sample["first_times"], sample["first_choices"], width)[0])
        low, high = interval(draws["recovery_error"][:, column])
        recovered_low, recovered_high = interval(draws["recovered_rate"][:, column])
        records.append({
            "width_seconds": width,
            "full_rate": full_rate,
            "recovered_rate": rate,
            "error": rate - full_rate,
            "error_ci95_low": float(low),
            "error_ci95_high": float(high),
            "recovered_ci95_low": float(recovered_low),
            "recovered_ci95_high": float(recovered_high),
        })
    return pd.DataFrame(records)


def main():
    # Loading applies completion exclusions; the reconstruction uses complete SM pairs.
    sample = prepare_sample(load_main())
    curves = individual_cdfs(sample)
    draws = bootstrap_reconstruction(sample, BOOTSTRAP, SEED)
    statistics = rate_statistics(sample, draws)
    style()
    fig = plt.figure(figsize=(10.4, 5.1))
    panels = fig.subfigures(1, 2, wspace=0.04)
    axes = []
    for panel, letter in zip(panels, "AB"):
        ax = panel.add_axes([0.20, 0.18, 0.74, 0.72])
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        panel_label(panel, letter)
        axes.append(ax)

    # A: empirical and reconstructed decision-time distributions.
    ax = axes[0]
    ax.step(curves.time_seconds, curves.empirical_all_sm_cdf, where="post",
            color="tab:blue", linewidth=2.6, label="Empirical")
    ax.step(curves.time_seconds, curves.reconstructed_potential_cdf, where="post",
            color="tab:orange", linewidth=2.6, linestyle="--", label="Reconstructed")
    ax.set(xlim=(0, curves.time_seconds.iloc[-1]), ylim=(0, 1.03),
           xlabel="Decision time (s)", ylabel="Cumulative probability")
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(loc="lower right", frameon=False)

    # B: cooperation-rate reconstruction errors and percentile intervals.
    ax = axes[1]
    positions = np.arange(len(statistics))
    points = statistics.error.to_numpy() * 100
    low = statistics.error_ci95_low.to_numpy() * 100
    high = statistics.error_ci95_high.to_numpy() * 100
    ax.axhline(0, color="0.4", linewidth=1, linestyle="--", zorder=1)
    ax.vlines(positions, low, high, color="tab:orange", linewidth=2, zorder=3)
    ax.hlines(low, positions - 0.07, positions + 0.07, color="tab:orange", linewidth=1.4)
    ax.hlines(high, positions - 0.07, positions + 0.07, color="tab:orange", linewidth=1.4)
    ax.scatter(positions, points, color="tab:orange", s=45, zorder=4)
    limit = max(5, float(np.ceil(np.max(np.abs(np.r_[points, low, high])) / 5) * 5))
    ax.set(xlim=(-0.5, len(statistics) - 0.5), ylim=(-limit, limit),
           xlabel="Bin width (s)", ylabel="Cooperation-rate error (pp)")
    ax.set_xticks(positions, statistics.width_seconds.astype(int).astype(str))

    output = PROJECT / "output/appendix/figures/figure_a1.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package", "Title": "Figure A1",
                "CreationDate": None, "ModDate": None}
    fig.savefig(output, format="pdf", bbox_inches=None, metadata=metadata)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

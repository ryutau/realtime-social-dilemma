"""Fit and plot SVO-dependent cooperation among second movers after cooperation."""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from src.common import SOURCE, SEED, BOOTSTRAP, load_main, rng
from src.plotting import style

COLORS = {"RT": "C1", "SQ": "C0"}
CONDITIONS = ("RT", "SQ")
SPAN, RIDGE, TOLERANCE = 0.90, 0.001, 1e-8
GRID_POINTS, BATCH_SIZE = 201, 32
HISTOGRAM_BINS = np.arange(-20, 51, 10)

def style_axes(ax, grid=False):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis="y", color="0.85", lw=0.65, alpha=0.7)
    ax.tick_params(length=3, width=0.7, labelsize=14, colors="black")

PROJECT = Path(__file__).resolve().parents[2]

def main():
    # Completion exclusions are applied when loading the main experiment.
    data = load_main()
    curves, groups = svo_curves(data)
    style()
    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_axes([0.225, 0.42, 0.685, 0.47])
    hist = fig.add_axes([0.225, 0.16, 0.685, 0.20])
    for condition, linestyle in [("RT", "-"), ("SQ", (0, (4, 2)))]:
        ax.fill_between(curves.svo, curves[f"{condition}_ci_low"], curves[f"{condition}_ci_high"],
                        color=COLORS[condition], alpha=0.16, linewidth=0, zorder=1)
        ax.plot(curves.svo, curves[condition], color=COLORS[condition], ls=linestyle,
                lw=2.05, label=condition, zorder=3)
    ax.set(xlim=(-20, 50), ylim=(0, 1.045), yticks=[0, 0.25, 0.5, 0.75, 1],
           xticks=[-20, 0, 20, 40], ylabel="Cooperation after C")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(axis="x", labelbottom=False)
    style_axes(ax, grid=True)
    ax.legend(loc="lower left", ncols=2, fontsize=14, frameon=False, handlelength=2, columnspacing=1.2)
    for condition, offset in [("RT", -1.7), ("SQ", 1.7)]:
        group = groups[condition]
        counts, _ = np.histogram(group.svo, bins=HISTOGRAM_BINS)
        assert counts.sum() == len(group)
        centers = (HISTOGRAM_BINS[:-1] + HISTOGRAM_BINS[1:]) / 2
        hist.bar(centers + offset, counts / len(group), width=3.2,
                 color=COLORS[condition], alpha=0.85, zorder=3)
    hist.set(xlim=(-20, 50), ylim=(0, 0.62), xticks=[-20, 0, 20, 40], yticks=[0, 0.3, 0.6],
             ylabel="Frequency", xlabel="SVO angle")
    hist.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    hist.yaxis.labelpad, hist.xaxis.labelpad = 3, 4
    style_axes(hist, grid=True)
    for panel in (ax, hist):
        panel.yaxis.set_label_coords(-0.21, 0.5)

    output = PROJECT / "output/main/figures/figure4B.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "Real-time PD replication package"}
    metadata.update(Title="figure4 panel B", CreationDate=None, ModDate=None)
    fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0.04, metadata=metadata)
    plt.close(fig)
    print(output)


def svo_curves(data):
    svo = pd.read_csv(SOURCE / "main_svo_params.csv")
    selected = data.merge(svo[["label", "svo"]], on="label", how="left", validate="one_to_one")
    selected = selected.loc[selected.condition.isin(CONDITIONS)
                            & ~selected.is_first_mover & selected.is_partner_coop].copy()
    assert selected.svo.notna().all() and selected.group_id.is_unique
    groups = {c: selected.loc[selected.condition.eq(c)].sort_values("group_id") for c in CONDITIONS}
    assert {c: (len(g), int(g.is_coop.sum())) for c, g in groups.items()} == {
        "RT": (65, 61), "SQ": (63, 47)}
    lower = max(group.svo.min() for group in groups.values())
    upper = min(group.svo.max() for group in groups.values())
    grid = np.linspace(lower, upper, GRID_POINTS)
    curves = {"svo": grid}
    started = time.monotonic()
    last_progress = started
    for condition, group in groups.items():
        x, y = group.svo.to_numpy(), group.is_coop.to_numpy(dtype=float)
        counts = rng(SEED, f"svo/{condition}").multinomial(
            len(group), np.full(len(group), 1 / len(group)), size=BOOTSTRAP)
        center = fit_batch(x, y, grid, np.ones((1, len(group))), SPAN, RIDGE, TOLERANCE)
        curves[condition] = center[0]
        draws = np.empty((BOOTSTRAP, len(grid)))
        for first in range(0, BOOTSTRAP, BATCH_SIZE):
            last = min(first + BATCH_SIZE, BOOTSTRAP)
            fitted = fit_batch(x, y, grid, counts[first:last],
                               SPAN, RIDGE, TOLERANCE)
            draws[first:last] = fitted
            if time.monotonic() - last_progress >= 15 or last == BOOTSTRAP:
                print(f"Figure 4B {condition}: {last:,}/{BOOTSTRAP:,}; "
                      f"{time.monotonic() - started:.1f}s", flush=True)
                last_progress = time.monotonic()
        curves[f"{condition}_ci_low"], curves[f"{condition}_ci_high"] = np.quantile(
            draws, [0.025, 0.975], axis=0, method="linear")
    return pd.DataFrame(curves), groups


def sigmoid(eta):
    return 1 / (1 + np.exp(-np.clip(eta, -700, 700)))


def fit_batch(x, y, grid, counts, span=0.90, ridge=0.001, tol=1e-8):
    """Return probabilities at each grid point for each row of resampling counts."""
    x, y, grid = (np.asarray(values, dtype=float) for values in (x, y, grid))
    counts = np.asarray(counts)
    assert x.ndim == y.ndim == grid.ndim == 1 and x.shape == y.shape
    assert counts.ndim == 2 and counts.shape[1] == len(x)
    assert np.isfinite(x).all() and np.isfinite(grid).all()
    assert np.isin(y, [0, 1]).all() and np.all(counts >= 0)
    assert np.all(counts.sum(axis=1) == len(x)) and 0 < span <= 1 and ridge > 0

    distance = np.abs(x[None, :] - grid[:, None])
    order = np.argsort(distance, axis=1, kind="stable")
    sorted_distances = np.take_along_axis(distance, order, axis=1)
    cumulative = counts[:, order].cumsum(axis=2)
    kth_index = (cumulative >= int(np.ceil(span * len(x)))).argmax(axis=2)
    bandwidth = np.take_along_axis(
        np.broadcast_to(sorted_distances, cumulative.shape), kth_index[:, :, None], axis=2
    )[:, :, 0]
    # Multiplicities determine the neighborhood; repeated SVO values are never jittered.
    zero_bandwidth = bandwidth == 0
    if zero_bandwidth.any():
        nearest_positive = np.where(distance > 0, distance, np.inf).min(axis=1)
        bandwidth = np.where(zero_bandwidth, nearest_positive[None, :], bandwidth)
    if not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0):
        raise ValueError("A local fit has no positive finite SVO bandwidth.")

    u = (x[None, None, :] - grid[None, :, None]) / bandwidth[:, :, None]
    weights = np.maximum(1 - np.abs(u) ** 3, 0) ** 3
    weights[weights <= 1e-12] = 0
    weights *= counts[:, None, :]
    weight_sum = weights.sum(axis=2)
    assert np.all(weight_sum > 0)
    positive = (weights * y).sum(axis=2)
    beta0 = np.log((positive + 0.5) / (weight_sum - positive + 0.5))
    beta1 = np.zeros_like(beta0)
    outcome = y[None, None, :]

    # Penalize both coefficients so all-cooperator bootstrap samples remain well defined.
    for iteration in range(100):
        eta = beta0[:, :, None] + beta1[:, :, None] * u
        probability = sigmoid(eta)
        residual = weights * (outcome - probability)
        g0 = residual.sum(axis=2) - ridge * beta0
        g1 = (residual * u).sum(axis=2) - ridge * beta1
        scores = np.maximum(np.abs(g0), np.abs(g1))
        max_score = float(scores.max())
        if max_score < tol:
            break
        variance = weights * probability * (1 - probability)
        h00 = variance.sum(axis=2) + ridge
        h01 = (variance * u).sum(axis=2)
        h11 = (variance * u * u).sum(axis=2) + ridge
        determinant = h00 * h11 - h01 * h01
        assert np.all(determinant > 0)
        step0 = (h11 * g0 - h01 * g1) / determinant
        step1 = (h00 * g1 - h01 * g0) / determinant
        scale = np.minimum(1, 10 / np.maximum(np.maximum(np.abs(step0), np.abs(step1)), 1e-12))
        scale[scores < tol] = 0
        objective = (weights * (outcome * eta - np.logaddexp(0, eta))).sum(axis=2)
        objective -= ridge * (beta0 ** 2 + beta1 ** 2) / 2
        for _ in range(35):
            new0, new1 = beta0 + scale * step0, beta1 + scale * step1
            trial = new0[:, :, None] + new1[:, :, None] * u
            new_objective = (weights * (outcome * trial - np.logaddexp(0, trial))).sum(axis=2)
            new_objective -= ridge * (new0 ** 2 + new1 ** 2) / 2
            decreased = new_objective < objective - 1e-12
            if not decreased.any():
                break
            scale[decreased] *= 0.5
        else:
            raise RuntimeError("Local logistic line search failed.")
        beta0, beta1 = new0, new1
    else:
        raise RuntimeError(f"Local logistic fit did not converge: maximum score {max_score}.")

    fitted = sigmoid(beta0)
    assert np.isfinite(fitted).all() and np.all((fitted >= 0) & (fitted <= 1))
    return fitted


if __name__ == "__main__":
    main()

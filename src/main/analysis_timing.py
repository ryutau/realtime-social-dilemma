"""Recovered RT defection timing and observed first-mover cooperation by time bin."""
import json

import numpy as np
import pandas as pd

from src.common import BINS, BOOTSTRAP, PROJECT, load_pd, summary


TIMING_SEED = 20260928


def rt_timing(data, repetitions, seed=TIMING_SEED):
    # Preserve source-row order when drawing RT first movers.
    first = data.loc[(data.condition == "RT") & data.is_first_mover]
    times = first.decision_time.to_numpy()
    cooperate = first.is_coop.to_numpy()
    bins = np.digitize(times, BINS) - 1
    crossing = (BINS[:-1] < 45) & (BINS[1:] > 45)
    late = BINS[:-1] >= 45
    if np.any((bins < 0) | (bins >= len(late))) or np.any(crossing[bins]):
        raise ValueError("The 45 s threshold requires an empty crossing bin and times within 0–60 s")

    def reconstruct(total, coop):
        cdf = total.cumsum(axis=-1) / total.sum(axis=-1, keepdims=True)
        potential_cdf = 1 - np.sqrt(np.clip(1 - cdf, 0, 1))
        mass = np.diff(potential_cdf, prepend=np.zeros_like(potential_cdf[..., :1]), axis=-1)
        conditional = np.divide(coop, total, out=np.zeros_like(coop, dtype=float), where=total > 0)
        return mass * conditional, mass * (1 - conditional)

    n = len(first)
    counts = np.bincount(bins, minlength=len(late))
    c_counts = np.bincount(bins, weights=cooperate, minlength=len(late)).astype(int)
    c_mass, d_mass = reconstruct(counts, c_counts)
    random = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
    draws = []
    for start in range(0, repetitions, 2000):
        size = min(2000, repetitions - start)
        selected = random.integers(n, size=(size, n))
        group = np.repeat(np.arange(size), n)
        times_selected = bins[selected].ravel()
        total = np.zeros((size, len(late)), dtype=int)
        coop = np.zeros_like(total)
        np.add.at(total, (group, times_selected), 1)
        np.add.at(coop, (group, times_selected), cooperate[selected].ravel())
        _, potential_d = reconstruct(total, coop)
        denominator = potential_d.sum(axis=1)
        if np.any(denominator == 0):
            raise ValueError("An RT bootstrap draw has no defections")
        draws.append(potential_d[:, late].sum(axis=1) / denominator)
    q5 = pd.DataFrame({"start_s": BINS[:-1], "end_s": BINS[1:],
                       "first_movers": counts, "cooperators": c_counts})
    q5["p_C_given_time"] = q5.cooperators / q5.first_movers.replace(0, np.nan)
    return {
        "threshold_seconds": 45,
        "bootstrap_seed": seed,
        "estimated_defection_fraction_after_threshold": summary(
            d_mass[late].sum() / d_mass.sum(), np.concatenate(draws)),
        "estimated_cooperation_mass_before_threshold": float(c_mass[~late].sum()),
        "estimated_cooperation_mass_after_threshold": float(c_mass[late].sum()),
    }, q5


def main():
    data = load_pd("main")
    timing, by_time = rt_timing(data, BOOTSTRAP, TIMING_SEED)
    result = {"bootstrap_repetitions": BOOTSTRAP, "RT_latent_timing": timing}
    output = PROJECT / "output/main"
    output.mkdir(parents=True, exist_ok=True)
    by_time.to_csv(output / "rt_first_cooperation_by_time_q5.csv", index=False)
    path = output / "timing.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

"""Estimate the potential (latent) cooperation rate of RT first movers.

Uses the latent CDF recovery from fig3 to decompose observed RT first-mover
decisions into latent cooperate/defect densities, then computes the overall
potential cooperation rate with pair-level bootstrap 95% CI.
"""

import numpy as np
import pandas as pd

from src.data import load_main
from src.fig_3_decision_time import compute_latent_stats, BINS


def potential_coop_rate(dt_array, is_coop_array):
    """Compute the latent cooperation rate from recovered distributions."""
    stats = compute_latent_stats(dt_array, is_coop_array, grid=BINS)
    total_coop = stats["latent_coop_pdf"].sum()
    total_defect = stats["latent_defect_pdf"].sum()
    return total_coop / (total_coop + total_defect)


def bootstrap_ci_pair_level(rt_data, n_bootstrap=10000, seed=42):
    """Bootstrap CI by resampling at the pair (group_id) level."""
    rng = np.random.RandomState(seed)
    group_ids = rt_data.group_id.unique()
    n_groups = len(group_ids)

    # Pre-index: store dt and is_coop arrays per group for fast lookup
    group_arrays = {}
    for gid in group_ids:
        mask = rt_data.group_id == gid
        group_arrays[gid] = (
            rt_data.loc[mask, "decision_time"].values,
            rt_data.loc[mask, "is_coop"].values,
        )

    boot_rates = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled_ids = rng.choice(group_ids, size=n_groups, replace=True)
        dt_parts = [group_arrays[gid][0] for gid in sampled_ids]
        coop_parts = [group_arrays[gid][1] for gid in sampled_ids]
        dt = np.concatenate(dt_parts)
        is_coop = np.concatenate(coop_parts)
        boot_rates[i] = potential_coop_rate(dt, is_coop)

    ci_lo, ci_hi = np.percentile(boot_rates, [2.5, 97.5])
    return boot_rates, ci_lo, ci_hi


def main():
    exp_data = load_main()
    rt_first = exp_data[(exp_data.condition == "RT") & (exp_data.is_first_mover)].copy()

    # Point estimate
    observed_rate = rt_first.is_coop.mean()
    latent_rate = potential_coop_rate(
        rt_first.decision_time.values, rt_first.is_coop.values
    )

    # Bootstrap CI
    boot_rates, ci_lo, ci_hi = bootstrap_ci_pair_level(rt_first)

    print("=" * 50)
    print("RT First-Mover Cooperation Rate Analysis")
    print("=" * 50)
    print(f"N pairs (first movers): {len(rt_first)}")
    print(f"Observed cooperation rate:  {observed_rate:.3f} ({observed_rate*100:.1f}%)")
    print(f"Potential cooperation rate: {latent_rate:.3f} ({latent_rate*100:.1f}%)")
    print(f"  95% CI (pair-level bootstrap, n=10000):")
    print(f"  [{ci_lo:.3f}, {ci_hi:.3f}] ([{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])")
    print(f"  Bootstrap mean: {boot_rates.mean():.3f}")
    print(f"  Bootstrap SD:   {boot_rates.std():.3f}")


if __name__ == "__main__":
    main()

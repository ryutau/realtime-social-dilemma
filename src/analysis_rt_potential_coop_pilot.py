"""Estimate the potential (latent) cooperation rate of RT first movers (pilot).

Uses the latent CDF recovery from fig3 to decompose observed RT first-mover
decisions into latent cooperate/defect densities, then computes the overall
potential cooperation rate with pair-level bootstrap 95% CI.
"""

import numpy as np
import pandas as pd

from src.data import load_pilot
from src.analysis_rt_potential_coop import potential_coop_rate, bootstrap_ci_pair_level


def main():
    exp_data = load_pilot()
    rt_first = exp_data[(exp_data.condition == "RT") & (exp_data.is_first_mover)].copy()

    # Point estimate
    observed_rate = rt_first.is_coop.mean()
    latent_rate = potential_coop_rate(
        rt_first.decision_time.values, rt_first.is_coop.values
    )

    # Bootstrap CI
    boot_rates, ci_lo, ci_hi = bootstrap_ci_pair_level(rt_first)

    print("=" * 55)
    print("RT First-Mover Cooperation Rate Analysis (Pilot)")
    print("=" * 55)
    print(f"N pairs (first movers): {len(rt_first)}")
    print(f"Observed cooperation rate:  {observed_rate:.3f} ({observed_rate*100:.1f}%)")
    print(f"Potential cooperation rate: {latent_rate:.3f} ({latent_rate*100:.1f}%)")
    print(f"  95% CI (pair-level bootstrap, n=10000):")
    print(f"  [{ci_lo:.3f}, {ci_hi:.3f}] ([{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])")
    print(f"  Bootstrap mean: {boot_rates.mean():.3f}")
    print(f"  Bootstrap SD:   {boot_rates.std():.3f}")


if __name__ == "__main__":
    main()

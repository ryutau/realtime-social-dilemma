"""Compare SM nominal first/second movers and RT voluntary first movers.

Reports cooperation rates with bootstrap CIs, and compares the decision time
distributions of nominal first vs second movers in SM using Mann-Whitney U
and Kolmogorov-Smirnov tests.
"""

import numpy as np
from scipy import stats as sp_stats

from src.data import load_pilot, load_main


def bootstrap_ci(data, n_bootstrap=1000, seed=42):
    rng = np.random.RandomState(seed)
    means = [
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ]
    return np.percentile(means, [2.5, 97.5])


def analyze_experiment(exp_data, exp_name):
    sm = exp_data[exp_data.condition == "SM"]
    rt = exp_data[exp_data.condition == "RT"]

    sm_first = sm[sm.is_first_mover]
    sm_second = sm[~sm.is_first_mover]
    rt_first = rt[rt.is_first_mover]

    sm_first_rate = sm_first.is_coop.mean()
    sm_first_ci = bootstrap_ci(sm_first.is_coop.values)
    sm_second_rate = sm_second.is_coop.mean()
    sm_second_ci = bootstrap_ci(sm_second.is_coop.values)
    rt_first_rate = rt_first.is_coop.mean()
    rt_first_ci = bootstrap_ci(rt_first.is_coop.values)

    # DT comparison: cooperators vs defectors in SM
    sm_coop_dt = sm.loc[sm.is_coop, "decision_time"].values
    sm_defect_dt = sm.loc[~sm.is_coop, "decision_time"].values
    u_stat, u_pval = sp_stats.mannwhitneyu(
        sm_coop_dt, sm_defect_dt, alternative="two-sided"
    )
    ks_stat, ks_pval = sp_stats.ks_2samp(sm_coop_dt, sm_defect_dt)

    print(f"{'=' * 60}")
    print(f"SM vs RT First-Mover Comparison ({exp_name})")
    print(f"{'=' * 60}")
    print()
    print(f"SM nominal first movers (n={len(sm_first)}):")
    print(f"  Cooperation rate: {sm_first_rate:.3f} ({sm_first_rate*100:.1f}%)")
    print(f"  95% CI: [{sm_first_ci[0]*100:.1f}, {sm_first_ci[1]*100:.1f}]")
    print()
    print(f"SM nominal second movers (n={len(sm_second)}):")
    print(f"  Cooperation rate: {sm_second_rate:.3f} ({sm_second_rate*100:.1f}%)")
    print(f"  95% CI: [{sm_second_ci[0]*100:.1f}, {sm_second_ci[1]*100:.1f}]")
    print()
    print(f"SM DT by decision (cooperators n={len(sm_coop_dt)}, defectors n={len(sm_defect_dt)}):")
    print(f"  Cooperators: mean={sm_coop_dt.mean():.2f}s, median={np.median(sm_coop_dt):.2f}s")
    print(f"  Defectors:   mean={sm_defect_dt.mean():.2f}s, median={np.median(sm_defect_dt):.2f}s")
    print(f"  Mann-Whitney U = {u_stat:.0f}, p = {u_pval:.3f}")
    print(f"  Kolmogorov-Smirnov D = {ks_stat:.3f}, p = {ks_pval:.3f}")
    print()
    print(f"RT voluntary first movers (n={len(rt_first)}):")
    print(f"  Cooperation rate: {rt_first_rate:.3f} ({rt_first_rate*100:.1f}%)")
    print(f"  95% CI: [{rt_first_ci[0]*100:.1f}, {rt_first_ci[1]*100:.1f}]")
    print()


def main():
    analyze_experiment(load_pilot(), "Pilot")
    analyze_experiment(load_main(), "Main")


if __name__ == "__main__":
    main()

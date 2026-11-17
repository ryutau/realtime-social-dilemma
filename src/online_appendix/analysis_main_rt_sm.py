"""Main RT–SM cooperation and action-time comparisons."""

import json

from scipy.stats import ks_2samp, mannwhitneyu

from src.common import (
    BOOTSTRAP, PROJECT, SEED, bootstrap_means, load_main, pair_rows, summary,
)


def main():
    data = load_main()
    rt, sm = (pair_rows(data, condition) for condition in ("RT", "SM"))
    rt_draws = bootstrap_means(rt.is_coop, BOOTSTRAP, SEED, "pair/RT")
    sm_first_draws = bootstrap_means(sm.is_coop, BOOTSTRAP, SEED, "pair/SM")
    sm_second_draws = bootstrap_means(sm.is_partner_coop, BOOTSTRAP, SEED, "pair/SM")
    rt_first, sm_first, sm_second = (v.mean() for v in (rt.is_coop, sm.is_coop, sm.is_partner_coop))
    ks = ks_2samp(rt.decision_time, sm.decision_time, alternative="two-sided")
    sm_individuals = data.loc[data.condition.eq("SM")]
    coop_times = sm_individuals.loc[sm_individuals.is_coop, "decision_time"]
    defect_times = sm_individuals.loc[~sm_individuals.is_coop, "decision_time"]
    u = mannwhitneyu(coop_times, defect_times, alternative="two-sided")
    sm_times = dict(cooperation_n=len(coop_times), defection_n=len(defect_times),
                    U=float(u.statistic), p_two_sided=float(u.pvalue))
    estimates = {
        "RT_observed_first": summary(rt_first, rt_draws),
        "SM_nominal_first": summary(sm_first, sm_first_draws),
        "SM_nominal_second": summary(sm_second, sm_second_draws),
        "RT_first_minus_SM_nominal_first": summary(rt_first - sm_first, rt_draws - sm_first_draws),
        "SM_all_individual_decision_times_C_vs_D_Mann_Whitney": sm_times,
        "Q3_RT_first_vs_SM_nominal_first_timing_distribution": {
            "RT_n": len(rt), "SM_n": len(sm),
            "RT_median_s": float(rt.decision_time.median()), "SM_median_s": float(sm.decision_time.median()),
            "KS_D": float(ks.statistic), "KS_p_two_sided": float(ks.pvalue),
        },
    }
    result = {"bootstrap_repetitions": BOOTSTRAP, "main_RT_SM": estimates}
    output = PROJECT / "output/online_appendix"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "main_rt_sm.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

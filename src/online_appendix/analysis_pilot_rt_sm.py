"""Pilot RT–SM cooperation and action-time comparisons."""

import json

import numpy as np
from scipy.stats import mannwhitneyu

from src.common import (
    BOOTSTRAP, PROJECT, load_pilot, pair_rows, summary,
)


PILOT_SEED = 20260930


def main():
    data = load_pilot()
    sm, rt = (pair_rows(data, condition) for condition in ("SM", "RT"))
    # Draw each condition's pairs independently, retaining
    # both choices in each draw for the within-SM first-minus-second comparison.
    random = np.random.default_rng(PILOT_SEED)
    sm_actions = sm[["is_coop", "is_partner_coop"]].to_numpy(dtype=float)
    rt_actions = rt[["is_coop", "is_partner_coop"]].to_numpy(dtype=float)
    sm_draws = sm_actions[random.integers(len(sm), size=(BOOTSTRAP, len(sm)))].mean(axis=1)
    rt_draws = rt_actions[random.integers(len(rt), size=(BOOTSTRAP, len(rt)))].mean(axis=1)
    sm_first, sm_second = sm_actions.mean(axis=0)
    rt_first = rt_actions[:, 0].mean()
    after_c = rt.loc[rt.is_coop, "is_partner_coop"]
    after_d = rt.loc[~rt.is_coop, "is_partner_coop"]
    groups = {
        "a_SM_all": data.loc[data.condition == "SM", "is_coop"],
        "b_RT_first": rt.is_coop,
        "c_RT_second_after_C": after_c,
        "d_RT_second_after_D": after_d,
    }
    sm_individuals = data.loc[data.condition.eq("SM")]
    coop_times = sm_individuals.loc[sm_individuals.is_coop, "decision_time"]
    defect_times = sm_individuals.loc[~sm_individuals.is_coop, "decision_time"]
    u = mannwhitneyu(coop_times, defect_times, alternative="two-sided")
    sm_times = dict(cooperation_n=len(coop_times), defection_n=len(defect_times),
                    U=float(u.statistic), p_two_sided=float(u.pvalue))
    estimates = {
        "bootstrap_seed": PILOT_SEED,
        "SM_first_minus_second": summary(sm_first - sm_second, sm_draws[:, 0] - sm_draws[:, 1]),
        "RT_first_minus_SM_nominal_first": summary(rt_first - sm_first, rt_draws[:, 0] - sm_draws[:, 0]),
        "SM_all_individual_decision_times_C_vs_D_Mann_Whitney": sm_times,
        "pilot_prereg_Q1_descriptive": {
            "rates": {name: {"cooperations": int(g.sum()), "n": len(g), "estimate": float(g.mean())}
                      for name, g in groups.items()},
            "c_minus_a": float(after_c.mean() - groups["a_SM_all"].mean()),
            "c_minus_d": float(after_c.mean() - after_d.mean()),
            "b_minus_a": float(rt_first - groups["a_SM_all"].mean()),
            "both_waited_RT_pairs": int((rt.decision_time >= 60).sum()),
        },
    }
    result = {"bootstrap_repetitions": BOOTSTRAP, "pilot_RT_SM": estimates}
    output = PROJECT / "output/online_appendix"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "pilot_rt_sm.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

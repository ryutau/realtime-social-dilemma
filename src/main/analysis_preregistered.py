"""Preregistered expected mutual-cooperation comparisons and two-proportion z tests."""
import json

import numpy as np
from scipy.stats import fisher_exact

from src.common import (
    BOOTSTRAP, SEED, PROJECT, bootstrap_means, load_main, pair_rows, pooled_z, rng, summary,
)


def sm_expected(sm, repetitions, seed):
    # Resample original dyads, including the eligible single member of one SM dyad.
    # Recompute the individual cooperation proportion as a ratio in each draw.
    clusters = sm.groupby("group_id", sort=True).is_coop.agg(["sum", "size"])
    values = clusters.to_numpy()
    random = rng(seed, "prereg/SM/expected_mutual")
    draws = []
    for start in range(0, repetitions, 2000):
        count = min(2000, repetitions - start)
        indices = random.integers(len(values), size=(count, len(values)))
        sampled = values[indices].sum(axis=1)
        draws.append((sampled[:, 0] / sampled[:, 1]) ** 2)
    return float(sm.is_coop.mean() ** 2), np.concatenate(draws)


def main():
    data = load_main()
    sm = data.loc[data.condition.eq("SM")]
    pairs = {condition: pair_rows(data, condition) for condition in ("RT", "SQ", "SM")}
    mutual = {condition: group.is_coop & group.is_partner_coop
              for condition, group in pairs.items()}
    first = {condition: group.is_coop for condition, group in pairs.items()}
    second = {condition: {
        "C": group.loc[group.is_coop, "is_partner_coop"],
        "D": group.loc[~group.is_coop, "is_partner_coop"],
    } for condition, group in pairs.items() if condition != "SM"}

    # With no RT both-wait dyads, the registered product equals observed mutual
    # cooperation in RT and SQ, including each pair-level bootstrap draw.
    if pairs["RT"].decision_time.ge(60).any():
        raise ValueError("RT both-wait dyads require the full preregistered formula")
    cc_draws = {
        condition: bootstrap_means(mutual[condition], BOOTSTRAP, SEED, f"pair/{condition}")
        for condition in ("RT", "SQ")
    }
    expected_sm, expected_sm_draws = sm_expected(sm, BOOTSTRAP, SEED)

    z_tests = {
        "Q2_RT_second_after_C_vs_after_D": pooled_z(second["RT"]["C"], second["RT"]["D"]),
        "Q3_RT_first_vs_SM_nominal_first": pooled_z(first["RT"], first["SM"]),
        "Q5_RT_first_vs_SQ_first": pooled_z(first["RT"], first["SQ"]),
        "Q6_after_C_RT_vs_SQ": pooled_z(second["RT"]["C"], second["SQ"]["C"]),
        "Q6_after_D_RT_vs_SQ": pooled_z(second["RT"]["D"], second["SQ"]["D"]),
    }
    d_test = z_tests["Q6_after_D_RT_vs_SQ"]
    d_test["fisher_p_two_sided_sensitivity"] = float(fisher_exact([
        [x, n - x] for x, n in zip(d_test["cooperations"], d_test["n"])
    ]).pvalue)
    result = {
        "bootstrap_repetitions": BOOTSTRAP,
        "main_seed": SEED,
        "individuals_by_condition": data.condition.value_counts().sort_index().to_dict(),
        "complete_pairs_by_condition": {k: len(v) for k, v in pairs.items()},
        "prereg_Q1_Q4_expected_CC": {
            "SM": summary(expected_sm, expected_sm_draws),
            "RT": summary(mutual["RT"].mean(), cc_draws["RT"]),
            "SQ": summary(mutual["SQ"].mean(), cc_draws["SQ"]),
            "RT_minus_SM": summary(mutual["RT"].mean() - expected_sm, cc_draws["RT"] - expected_sm_draws),
            "RT_minus_SQ": summary(
                mutual["RT"].mean() - mutual["SQ"].mean(), cc_draws["RT"] - cc_draws["SQ"]
            ),
        },
        "prereg_z_tests": z_tests,
    }
    output = PROJECT / "output/main"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preregistered.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

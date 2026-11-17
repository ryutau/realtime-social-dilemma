"""Observed cooperation rates, condition differences, and supplementary z tests."""
import json

from src.common import (
    BOOTSTRAP, SEED, PROJECT, bootstrap_means, load_main, pair_rows, pooled_z, summary,
)


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

    # Use the same sample order and random streams as Figure 2.
    cc_draws = {
        condition: bootstrap_means(values, BOOTSTRAP, SEED, f"pair/{condition}")
        for condition, values in mutual.items()
    }
    first_draws = {
        condition: bootstrap_means(first[condition], BOOTSTRAP, SEED, f"pair/{condition}")
        for condition in ("RT", "SQ")
    }
    second_draws = {
        condition: {
            action: bootstrap_means(values, BOOTSTRAP, SEED, f"conditional/{condition}/after_{action}")
            for action, values in groups.items()
        }
        for condition, groups in second.items()
    }
    sm_draws = bootstrap_means(sm.is_coop, BOOTSTRAP, SEED, "individual/SM")
    result = {
        "bootstrap_repetitions": BOOTSTRAP,
        "main_seed": SEED,
        "individuals_by_condition": data.condition.value_counts().sort_index().to_dict(),
        "complete_pairs_by_condition": {k: len(v) for k, v in pairs.items()},
        "RT_pairs_with_first_before_60s": int(pairs["RT"].decision_time.lt(60).sum()),
        "observed_CC_by_condition": {
            k: summary(mutual[k].mean(), cc_draws[k]) for k in pairs
        },
        "SM_individual_C": summary(sm.is_coop.mean(), sm_draws),
        "observed_CC_differences_RT_minus": {
            k: summary(mutual["RT"].mean() - mutual[k].mean(), cc_draws["RT"] - cc_draws[k])
            for k in ("SM", "SQ")
        },
        "observed_first_difference_RT_minus_SQ": summary(
            first["RT"].mean() - first["SQ"].mean(),
            first_draws["RT"] - first_draws["SQ"]
        ),
        "observed_second_difference_RT_minus_SQ": {
            k: summary(second["RT"][k].mean() - second["SQ"][k].mean(),
                       second_draws["RT"][k] - second_draws["SQ"][k])
            for k in ("C", "D")
        },
        "supplementary_z_tests": {
            "observed_CC_RT_vs_SM": pooled_z(mutual["RT"], mutual["SM"]),
            "observed_CC_RT_vs_SQ": pooled_z(mutual["RT"], mutual["SQ"]),
            "SQ_second_after_C_vs_after_D": pooled_z(second["SQ"]["C"], second["SQ"]["D"]),
        },
    }
    output = PROJECT / "output/main"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cooperation.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

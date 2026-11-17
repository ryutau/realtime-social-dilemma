"""Table A1 and Appendix SVO comparisons: angles, categories, and standardized cooperation."""

import json
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy import stats

from src.common import SOURCE, INTERMEDIATE, PROJECT, load_pd

CONDITIONS = ("RT", "SQ")
CATEGORIES = ("Competitive", "Individualistic", "Prosocial", "Altruistic")


def main():
    # RT/SQ second movers whose first mover cooperated, with a measured SVO angle.
    choices = load_pd("main")
    svo = pd.read_csv(SOURCE / "main_svo_params.csv")
    data = choices.loc[choices.condition.isin(CONDITIONS)
                       & ~choices.is_first_mover & choices.is_partner_coop]
    data = data.merge(svo, on="label", how="left", validate="one_to_one")
    if data.group_id.duplicated().any() or data.svo.isna().any():
        raise ValueError("Expected one second mover per selected pair with a measured SVO angle")
    data = data.assign(C=data.is_coop.astype(int)).sort_values(["condition", "group_id"])

    angles = describe_angles(data)  # Two-sided Mann–Whitney comparison.
    categories = compare_categories(data)  # Fisher tests and Newcombe intervals.
    path = INTERMEDIATE / "svo_standardization.json"
    if not path.is_file():
        raise FileNotFoundError("Run python -m src.appendix.prepare_svo first")
    standardized = json.loads(path.read_text())
    result = {"svo_angle_comparison": angles, **standardized}

    # Descriptive and inferential results reported in the Appendix text.
    summaries = PROJECT / "output/appendix/svo"
    summaries.mkdir(parents=True, exist_ok=True)
    categories.to_csv(summaries / "category_comparisons.csv", index=False)
    (summaries / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    # Table A1A: category counts, rates, and condition comparisons.
    categories["rt_rate"] = categories.rt_c / categories.rt_n.replace(0, np.nan)
    categories["sq_rate"] = categories.sq_c / categories.sq_n.replace(0, np.nan)
    tables = PROJECT / "output/appendix/tables"
    tables.mkdir(parents=True, exist_ok=True)
    categories.to_csv(tables / "table_a1a.csv", index=False)

    # Table A1B: estimates and percentile intervals from the shared bootstrap.
    rows = [{"quantity": key, "estimate": value["estimate"],
             "ci_low": value["ci95_percentile"][0], "ci_high": value["ci95_percentile"][1]}
            for key, value in standardized["standardized"].items()]
    pd.DataFrame(rows).to_csv(tables / "table_a1b.csv", index=False)
    print(summaries / "summary.json")
    print(tables / "table_a1a.csv")
    print(tables / "table_a1b.csv")


def wilson(k, n):
    """Two-sided 95% Wilson score interval without continuity correction."""
    if n == 0:
        return np.nan, np.nan
    z = NormalDist().inv_cdf(.975)
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return max(0, center-half), min(1, center+half)


def newcombe(k1, n1, k2, n2):
    """Newcombe method 10: square-and-add hybrid Wilson interval."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan
    p1, p2 = k1/n1, k2/n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    diff = p1-p2
    low = diff - np.sqrt((p1-l1)**2 + (u2-p2)**2)
    high = diff + np.sqrt((u1-p1)**2 + (p2-l2)**2)
    return diff, low, high


def describe_angles(data):
    groups = {c: data.loc[data.condition.eq(c), "svo"] for c in CONDITIONS}
    u = stats.mannwhitneyu(groups["RT"], groups["SQ"], alternative="two-sided", method="asymptotic")
    return {
        "groups": {c: {"n": len(series), "mean_degrees": float(series.mean()), "sd_degrees": float(series.std(ddof=1))}
                   for c, series in groups.items()},
        "mann_whitney_u_rt": float(u.statistic), "p_two_sided": float(u.pvalue),
    }


def compare_categories(data):
    data = data.assign(category=np.select(
        [data.svo.lt(-12.04), data.svo.le(22.45), data.svo.le(57.15)],
        CATEGORIES[:3], default=CATEGORIES[3],
    ))
    results = []
    for category in CATEGORIES:
        group = data.loc[data.category.eq(category)]
        y = {c: group.loc[group.condition.eq(c), "C"] for c in CONDITIONS}
        n_rt, n_sq = len(y["RT"]), len(y["SQ"])
        c_rt, c_sq = int(y["RT"].sum()), int(y["SQ"].sum())
        diff, lo, hi = newcombe(c_rt, n_rt, c_sq, n_sq)
        fisher = stats.fisher_exact(
            [[c_rt, n_rt - c_rt], [c_sq, n_sq - c_sq]], alternative="two-sided"
        ).pvalue if n_rt and n_sq else np.nan
        results.append(dict(category=category, rt_n=n_rt, sq_n=n_sq, rt_c=c_rt, sq_c=c_sq,
                            difference=diff, ci_low=lo, ci_high=hi, fisher_p=fisher))
    return pd.DataFrame(results)


if __name__ == "__main__":
    main()

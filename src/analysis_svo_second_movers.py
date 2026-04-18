"""Compare SVO scores between RT and SQ second movers.

RT second movers are voluntary (decided slower), while SQ second movers
are assigned. This analysis tests whether their social value orientations
differ using a Mann-Whitney U test.
"""

from scipy import stats

from src.data import load_main


def main():
    exp_data = load_main()
    second_movers = exp_data[~exp_data.is_first_mover]

    rt_svo = second_movers.loc[second_movers.condition == "RT", "svo"].dropna()
    sq_svo = second_movers.loc[second_movers.condition == "SQ", "svo"].dropna()

    u_stat, p_value = stats.mannwhitneyu(rt_svo, sq_svo, alternative="two-sided")

    print("=" * 55)
    print("SVO Comparison: RT vs SQ Second Movers")
    print("=" * 55)
    print(f"RT second movers: n={len(rt_svo)}, mean={rt_svo.mean():.2f}, median={rt_svo.median():.2f}")
    print(f"SQ second movers: n={len(sq_svo)}, mean={sq_svo.mean():.2f}, median={sq_svo.median():.2f}")
    print()
    print(f"Mann-Whitney U test (two-sided):")
    print(f"  U = {u_stat:.0f}, p = {p_value:.3f}")


if __name__ == "__main__":
    main()

"""Table OA4: RT two-step estimates with bootstrap standard errors and percentile intervals."""

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common import INTERMEDIATE, PROJECT


def main():
    with np.load(INTERMEDIATE / "rt_regression_bootstrap_coefficients.npz") as saved:
        beta, draws, variables = saved["point"], saved["beta"], saved["labels"]
    context = json.loads((INTERMEDIATE / "rt_regression_summary.json").read_text())
    if not np.isfinite(draws).all():
        raise ValueError("Table OA4 requires a complete bootstrap with no failed draws.")

    se = draws.std(axis=0, ddof=1)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    z = beta / se
    table = pd.DataFrame(dict(
        variable=variables, coef=beta, bootstrap_se=se,
        coef_ci_low=low, coef_ci_high=high, wald_z=z,
        wald_p=2 * norm.sf(np.abs(z)),
    ))
    table = table.assign(
        experiment="main", model="rt_twostep", manuscript_table="table_oa4",
        n=context["step2_n"], odds_ratio=np.exp(table.coef),
        or_ci_low=np.exp(table.coef_ci_low), or_ci_high=np.exp(table.coef_ci_high),
        significance=np.select(
            [table.wald_p.lt(.001), table.wald_p.lt(.01), table.wald_p.lt(.05)],
            ["***", "**", "*"], default="",
        ),
    )
    diagnostics = dict(
        model="rt_twostep",
        inference="Within-session pair bootstrap, percentile CI and two-sided bootstrap-SE Wald",
        **context,
    )
    output = PROJECT / "output/online_appendix/tables"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "table_oa4.csv", index=False)
    (output / "table_oa4_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(output / "table_oa4.csv")


if __name__ == "__main__":
    main()

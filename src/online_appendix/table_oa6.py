"""Table OA6: Pilot SM participants with complete predictors."""

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.common import SOURCE, PROJECT, load_pd
from src.online_appendix.logistic import fit_logit, predictor_matrix, standardization


def main():
    data = load_pd("pilot").set_index("label")
    for measure in ("svo_params", "ra_params", "demographics"):
        data = data.join(pd.read_csv(SOURCE / f"pilot_{measure}.csv", index_col=0),
                         how="left", validate="one_to_one")

    # Standardize continuous predictors using the full eligible pilot sample.
    columns, scales = standardization(data, "pilot")
    sample = data.loc[data.condition.eq("SM")]
    variables = ['const', 'svo', 'risk', 'ambiguity', 'is_male', 'age', 'cnt_approvals']
    x = predictor_matrix(sample, columns, scales, variables)
    y = sample.is_coop.to_numpy(float)
    complete = np.isfinite(x).all(axis=1) & np.isfinite(y)
    beta, se, fit = fit_logit(x[complete], y[complete])

    z = beta / se
    critical = norm.ppf(.975)
    table = pd.DataFrame(dict(
        variable=variables, coef=beta, model_se=se, wald_z=z,
        coef_ci_low=beta - critical * se, coef_ci_high=beta + critical * se,
        wald_p=2 * norm.sf(np.abs(z)),
    ))
    table = table.assign(
        experiment="pilot", model="sm_all", manuscript_table="table_oa6",
        n=int(complete.sum()), odds_ratio=np.exp(table.coef),
        or_ci_low=np.exp(table.coef_ci_low), or_ci_high=np.exp(table.coef_ci_high),
        significance=np.select(
            [table.wald_p.lt(.001), table.wald_p.lt(.01), table.wald_p.lt(.05)],
            ["***", "**", "*"], default="",
        ),
    )
    diagnostics = dict(
        experiment="pilot", model="sm_all", n_context=len(sample),
        n=int(complete.sum()), n_cooperative=int(y[complete].sum()),
        n_missing_predictor_or_outcome=int((~complete).sum()),
        omitted_variables=[], standardization=scales,
        inference="Ordinary maximum likelihood, model SE, two-sided Wald", **fit,
    )

    output = PROJECT / "output/online_appendix/tables"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "table_oa6.csv", index=False)
    (output / "table_oa6_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print(output / "table_oa6.csv")


if __name__ == "__main__":
    main()

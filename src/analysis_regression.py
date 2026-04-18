"""Regression analyses for SI: individual-level predictors of cooperation.

Implements Analyses 1-5 from docs/regression_analysis_specification.md
for both main and pilot experiments.

Analyses:
  1. SM: logistic regression on all SM participants
  2. SQ first movers: logistic regression (main only)
  3. SQ second movers after C: logistic regression (main only)
  4. RT: two-step MLE with selection correction
  5. RT second movers after C: logistic regression
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

from src.data import load_main, load_pilot
from src.data.utils import OUTPUT_TABLES
from src.fig3_decision_time import compute_empirical_cdf, recover_latent_cdf, BINS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_predictor_cols(df):
    """Identify available predictor columns."""
    cols = {}
    cols["svo"] = "svo"
    cols["age"] = "age"
    cols["is_male"] = "is_male"
    cols["cnt_approvals"] = "cnt_approvals"
    # Risk / ambiguity: different names in pilot vs main
    if "risk_aversion_change_point" in df.columns:
        cols["risk"] = "risk_aversion_change_point"
        cols["ambiguity"] = "ambiguity_aversion_change_point"
    else:
        cols["risk"] = "risk_aversion_modeling"
        cols["ambiguity"] = "ambiguity_aversion_modeling"
    return cols


def _compute_population_stats(exp_data, pred_cols):
    """Compute mean/std from the full experimental population for standardization."""
    continuous = ["svo", "risk", "ambiguity", "age", "cnt_approvals"]
    pop_stats = {}
    for key in continuous:
        col = pred_cols[key]
        vals = exp_data[col].astype(float).dropna()
        pop_stats[key] = (vals.mean(), vals.std())
    return pop_stats


def _prepare_predictors(df, pred_cols, pop_stats):
    """Build standardized predictor matrix. Returns X (with const), y, labels.

    Continuous variables are z-scored using the full experimental population
    mean/std (not the subsample), so coefficients are comparable across analyses.
    """
    continuous = ["svo", "risk", "ambiguity", "age", "cnt_approvals"]
    X = pd.DataFrame(index=df.index)
    labels = []
    for key in ["svo", "risk", "ambiguity", "is_male", "age", "cnt_approvals"]:
        col = pred_cols[key]
        if key in continuous:
            mean, std = pop_stats[key]
            vals = df[col].astype(float)
            X[key] = (vals - mean) / std
        else:
            X[key] = df[col].astype(float)
        labels.append(key)

    # Drop rows with any missing predictor
    mask = X.notna().all(axis=1) & df["is_coop"].notna()
    X = X.loc[mask]
    y = df.loc[mask, "is_coop"].astype(float)
    X = sm.add_constant(X)
    labels = ["const"] + labels
    return X, y, labels, mask


def _run_logit(X, y, labels, title):
    """Fit logistic regression and print results."""
    try:
        model = sm.Logit(y, X)
        result = model.fit(disp=0, maxiter=100)
    except Exception:
        # Try Firth-like penalized (regularized) logit
        model = sm.Logit(y, X)
        result = model.fit_regularized(disp=0, alpha=0.1)

    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"  N = {int(result.nobs)}, "
          f"Pseudo R² = {getattr(result, 'prsquared', float('nan')):.3f}")
    print(f"{'─' * 70}")
    print(f"  {'Variable':<16} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8} {'OR':>8}")
    print(f"  {'─' * 64}")

    params = np.asarray(result.params)
    bse = np.asarray(result.bse)
    zvals = params / bse
    pvals = 2 * stats.norm.sf(np.abs(zvals))

    rows = []
    for i, label in enumerate(labels):
        p_str = f"{pvals[i]:.3f}" if pvals[i] >= 0.001 else "<.001"
        or_val = np.exp(params[i])
        sig = ""
        if pvals[i] < 0.001:
            sig = "***"
        elif pvals[i] < 0.01:
            sig = "**"
        elif pvals[i] < 0.05:
            sig = "*"
        print(f"  {label:<16} {params[i]:>8.3f} {bse[i]:>8.3f} "
              f"{zvals[i]:>8.3f} {p_str:>8} {or_val:>8.3f} {sig}")
        rows.append({
            "variable": label, "coef": params[i], "se": bse[i],
            "z": zvals[i], "p": pvals[i], "odds_ratio": or_val,
        })

    df_result = pd.DataFrame(rows)
    df_result["analysis"] = title
    df_result["n"] = int(result.nobs)
    df_result["pseudo_r2"] = getattr(result, "prsquared", float("nan"))
    return df_result


# ---------------------------------------------------------------------------
# Analysis 4: Two-step MLE for RT
# ---------------------------------------------------------------------------

def _estimate_survival_functions(rt_first_movers):
    """Step 1: estimate action-specific survival functions nonparametrically."""
    dt = rt_first_movers.decision_time.values
    is_coop = rt_first_movers.is_coop.values

    # Observed CDF of first-mover decision times, separately by action
    grid = BINS  # 0, 2, 4, ..., 60
    coop_dt = dt[is_coop.astype(bool)]
    defect_dt = dt[~is_coop.astype(bool)]

    # Overall observed CDF → latent CDF via inversion
    obs_cdf = compute_empirical_cdf(dt, grid)
    latent_cdf = recover_latent_cdf(obs_cdf)

    # Conditional cooperation probability at each grid point
    dt_bins = np.digitize(dt, grid)
    p_coop = {}
    for b in np.unique(dt_bins):
        mask_b = dt_bins == b
        if mask_b.sum() > 0:
            p_coop[b] = is_coop[mask_b].mean()
    p_coop_arr = pd.Series(p_coop).reindex(range(1, len(grid)), fill_value=np.nan).values

    latent_pdf = np.diff(latent_cdf)
    latent_coop_pdf = np.where(
        latent_pdf * p_coop_arr > 0, latent_pdf * p_coop_arr, 0
    )
    latent_defect_pdf = np.where(
        latent_pdf * (1 - p_coop_arr) > 0, latent_pdf * (1 - p_coop_arr), 0
    )

    # CDF from PDFs
    coop_cdf = np.insert(np.cumsum(latent_coop_pdf), 0, 0)
    coop_cdf = coop_cdf / coop_cdf[-1] if coop_cdf[-1] > 0 else coop_cdf
    defect_cdf = np.insert(np.cumsum(latent_defect_pdf), 0, 0)
    defect_cdf = defect_cdf / defect_cdf[-1] if defect_cdf[-1] > 0 else defect_cdf

    # Survival = 1 - CDF
    s_coop = 1.0 - coop_cdf
    s_defect = 1.0 - defect_cdf

    return grid, s_coop, s_defect


def _get_survival_at_time(t, grid, s_coop, s_defect):
    """Interpolate survival functions at time t."""
    idx = np.searchsorted(grid, t, side="right") - 1
    idx = np.clip(idx, 0, len(s_coop) - 1)
    return s_coop[idx], s_defect[idx]


def _rt_twostep_negloglik(beta, X_first, y_first, X_second, t_first_for_second,
                           grid, s_coop, s_defect):
    """Negative log-likelihood for the two-step RT estimator."""
    # First movers: log P(a_i | X_i)
    logit_first = X_first @ beta
    p_coop_first = 1.0 / (1.0 + np.exp(-logit_first))
    p_coop_first = np.clip(p_coop_first, 1e-10, 1 - 1e-10)
    ll_first = np.sum(
        y_first * np.log(p_coop_first) + (1 - y_first) * np.log(1 - p_coop_first)
    )

    # Second movers: log[P(C|X_j) * S_C(t_i) + P(D|X_j) * S_D(t_i)]
    logit_second = X_second @ beta
    p_coop_second = 1.0 / (1.0 + np.exp(-logit_second))
    p_coop_second = np.clip(p_coop_second, 1e-10, 1 - 1e-10)

    sc_vals = np.array([
        _get_survival_at_time(t, grid, s_coop, s_defect)[0]
        for t in t_first_for_second
    ])
    sd_vals = np.array([
        _get_survival_at_time(t, grid, s_coop, s_defect)[1]
        for t in t_first_for_second
    ])

    mixture = p_coop_second * sc_vals + (1 - p_coop_second) * sd_vals
    mixture = np.clip(mixture, 1e-10, None)
    ll_second = np.sum(np.log(mixture))

    return -(ll_first + ll_second)


def _run_rt_twostep(exp_data, pred_cols, pop_stats, title):
    """Run the two-step MLE for RT (Analysis 4)."""
    rt = exp_data[exp_data.condition == "RT"].copy()
    rt_first = rt[rt.is_first_mover]
    rt_second = rt[~rt.is_first_mover]

    # Step 1: survival functions
    grid, s_coop, s_defect = _estimate_survival_functions(rt_first)

    # Prepare predictors for all RT participants
    X_first, y_first, labels, mask_first = _prepare_predictors(rt_first, pred_cols, pop_stats)
    X_second_full, _, _, mask_second = _prepare_predictors(rt_second, pred_cols, pop_stats)

    # Get first-mover decision time for each second mover's pair
    # second mover's partner_decision_time = first mover's decision_time
    t_first_for_second = rt_second.loc[mask_second, "partner_decision_time"].values

    X_first_arr = X_first.values
    y_first_arr = y_first.values
    X_second_arr = X_second_full.values
    n_params = X_first_arr.shape[1]

    # Initial guess from naive logit on first movers
    try:
        naive = sm.Logit(y_first, X_first).fit(disp=0)
        beta0 = naive.params.values
    except Exception:
        beta0 = np.zeros(n_params)

    # Optimize
    result = optimize.minimize(
        _rt_twostep_negloglik,
        beta0,
        args=(X_first_arr, y_first_arr, X_second_arr, t_first_for_second,
              grid, s_coop, s_defect),
        method="BFGS",
    )
    beta_hat = result.x

    # Standard errors from Hessian
    hess = result.hess_inv if hasattr(result, 'hess_inv') else np.eye(n_params)
    if isinstance(hess, optimize.LbfgsInvHessProduct):
        hess = hess.todense()
    se = np.sqrt(np.diag(np.abs(hess)))

    zvals = beta_hat / se
    pvals = 2 * stats.norm.sf(np.abs(zvals))

    n_first = len(y_first_arr)
    n_second = len(X_second_arr)

    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"  N_first = {n_first}, N_second = {n_second}, N_total = {n_first + n_second}")
    print(f"  (Two-step MLE with selection correction)")
    print(f"{'─' * 70}")
    print(f"  {'Variable':<16} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8} {'OR':>8}")
    print(f"  {'─' * 64}")

    rows = []
    for i, label in enumerate(labels):
        p_str = f"{pvals[i]:.3f}" if pvals[i] >= 0.001 else "<.001"
        or_val = np.exp(beta_hat[i])
        sig = ""
        if pvals[i] < 0.001:
            sig = "***"
        elif pvals[i] < 0.01:
            sig = "**"
        elif pvals[i] < 0.05:
            sig = "*"
        print(f"  {label:<16} {beta_hat[i]:>8.3f} {se[i]:>8.3f} "
              f"{zvals[i]:>8.3f} {p_str:>8} {or_val:>8.3f} {sig}")
        rows.append({
            "variable": label, "coef": beta_hat[i], "se": se[i],
            "z": zvals[i], "p": pvals[i], "odds_ratio": or_val,
        })

    df_result = pd.DataFrame(rows)
    df_result["analysis"] = title
    df_result["n"] = n_first + n_second
    df_result["pseudo_r2"] = float("nan")

    return df_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _significance(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def _save_result(df_result, exp_name, analysis_tag):
    """Save a single analysis result to CSV (per-variable detail)."""
    df_out = df_result.copy()
    df_out["significance"] = df_out["p"].apply(_significance)
    col_order = ["variable", "coef", "se", "z", "p", "odds_ratio", "significance"]
    df_out = df_out[col_order]
    filename = f"regression_{exp_name.lower()}_{analysis_tag}.csv"
    out_path = OUTPUT_TABLES / filename
    df_out.to_csv(out_path, index=False)
    print(f"  Saved to {out_path}")
    # Return original with metadata for summary
    return df_result


def run_all_analyses(exp_data, exp_name, has_sq=True):
    pred_cols = _get_predictor_cols(exp_data)
    pop_stats = _compute_population_stats(exp_data, pred_cols)
    summary_rows = []

    print(f"\n{'=' * 70}")
    print(f"  REGRESSION ANALYSES — {exp_name} Experiment")
    print(f"  (Predictors z-scored using full {exp_name} population)")
    print(f"{'=' * 70}")

    def _collect(df_result, tag):
        row = df_result.iloc[0]
        summary_rows.append({
            "experiment": exp_name,
            "analysis": row["analysis"],
            "tag": tag,
            "n": int(row["n"]),
            "pseudo_r2": row["pseudo_r2"],
        })

    # Analysis 1: SM
    sm_data = exp_data[exp_data.condition == "SM"]
    X, y, labels, _ = _prepare_predictors(sm_data, pred_cols, pop_stats)
    df = _run_logit(X, y, labels, f"Analysis 1: SM — all participants ({exp_name})")
    _save_result(df, exp_name, "sm_all")
    _collect(df, "sm_all")

    if has_sq:
        # Analysis 2: SQ first movers
        sq_first = exp_data[(exp_data.condition == "SQ") & exp_data.is_first_mover]
        X, y, labels, _ = _prepare_predictors(sq_first, pred_cols, pop_stats)
        df = _run_logit(X, y, labels,
                        f"Analysis 2: SQ — assigned first movers ({exp_name})")
        _save_result(df, exp_name, "sq_first")
        _collect(df, "sq_first")

        # Analysis 3: SQ second movers after C
        sq_second_c = exp_data[
            (exp_data.condition == "SQ") & ~exp_data.is_first_mover & exp_data.is_partner_coop
        ]
        X, y, labels, _ = _prepare_predictors(sq_second_c, pred_cols, pop_stats)
        df = _run_logit(X, y, labels,
                        f"Analysis 3: SQ — 2nd movers after C ({exp_name})")
        _save_result(df, exp_name, "sq_second_after_c")
        _collect(df, "sq_second_after_c")

    # Analysis 4: RT two-step
    df = _run_rt_twostep(exp_data, pred_cols, pop_stats,
                         f"Analysis 4: RT — two-step MLE ({exp_name})")
    _save_result(df, exp_name, "rt_twostep")
    _collect(df, "rt_twostep")

    # Analysis 5: RT second movers after C
    rt_second_c = exp_data[
        (exp_data.condition == "RT") & ~exp_data.is_first_mover & exp_data.is_partner_coop
    ]
    X, y, labels, _ = _prepare_predictors(rt_second_c, pred_cols, pop_stats)
    df = _run_logit(X, y, labels,
                    f"Analysis 5: RT — 2nd movers after C ({exp_name})")
    _save_result(df, exp_name, "rt_second_after_c")
    _collect(df, "rt_second_after_c")

    return summary_rows


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    summary = []
    summary.extend(run_all_analyses(load_main(), "Main", has_sq=True))
    print("\n\n")
    summary.extend(run_all_analyses(load_pilot(), "Pilot", has_sq=False))

    # Save summary CSV
    df_summary = pd.DataFrame(summary)
    out_path = OUTPUT_TABLES / "regression_summary.csv"
    df_summary.to_csv(out_path, index=False)
    print(f"\n  Saved summary to {out_path}")


if __name__ == "__main__":
    main()

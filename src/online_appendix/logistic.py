"""Shared standardization, design matrices, and ordinary logistic estimation."""

import numpy as np
from scipy import optimize, special

LABELS = ["const", "svo", "risk", "ambiguity", "is_male", "age", "cnt_approvals"]
CONTINUOUS = ["svo", "risk", "ambiguity", "age", "cnt_approvals"]


def standardization(data, experiment):
    risk = "risk_aversion_change_point" if experiment == "main" else "risk_aversion_modeling"
    ambiguity = "ambiguity_aversion_change_point" if experiment == "main" else "ambiguity_aversion_modeling"
    columns = dict(svo="svo", risk=risk, ambiguity=ambiguity,
                   is_male="is_male", age="age", cnt_approvals="cnt_approvals")
    scales = {key: (float(data[column].mean()), float(data[column].std(ddof=1)))
              for key, column in columns.items() if key in CONTINUOUS}
    return columns, scales

def predictor_matrix(frame, columns, scales, labels=LABELS):
    values = [np.ones(len(frame))]
    for key in labels[1:]:
        value = frame[columns[key]].to_numpy(float)
        if key in scales:
            mean, sd = scales[key]
            value = (value - mean) / sd
        values.append(value)
    return np.column_stack(values)

def check_overlap(x, y):
    """Fail if the design is rank deficient or the data are separated."""
    if len(y) == 0 or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError("The design matrix does not have full column rank.")
    signed_x = (2 * y - 1)[:, None] * x
    # A direction with nonnegative signed margins and a positive sum separates.
    lp = optimize.linprog(
        -signed_x.sum(axis=0), A_ub=-signed_x, b_ub=np.zeros(len(y)),
        bounds=[(-1., 1.)] * x.shape[1], method="highs",
    )
    if not lp.success:
        raise RuntimeError(f"Separation check failed: {lp.message}")
    if -lp.fun > 1e-7:
        raise ValueError("The regression data exhibit separation.")

def fit_logit(x, y):
    def evaluate(beta):
        eta = x @ beta
        prob = special.expit(eta)
        loss = np.sum(np.logaddexp(0., eta) - y * eta)
        score = x.T @ (prob - y)
        information = (x.T * (prob * (1 - prob))) @ x
        return loss, score, information

    check_overlap(x, y)
    result = optimize.minimize(
        lambda beta: evaluate(beta)[:2], np.zeros(x.shape[1]),
        method="BFGS", jac=True, options={"gtol": 1e-8, "maxiter": 500},
    )
    beta = result.x.copy()
    _, score, information = evaluate(beta)
    # Finish at a tight score tolerance after the line-search optimizer.
    for _ in range(6):
        if np.max(np.abs(score)) < 1e-9:
            break
        beta -= np.linalg.solve(information, score)
        _, score, information = evaluate(beta)
    max_abs_score = float(np.max(np.abs(score)))
    min_eigenvalue = float(np.linalg.eigvalsh(information)[0])
    if max_abs_score >= 1e-8 or min_eigenvalue <= 0:
        raise RuntimeError(f"The maximum-likelihood fit did not converge: {result.message}")
    se = np.sqrt(np.diag(np.linalg.inv(information)))
    return beta, se, dict(max_abs_score=max_abs_score, minimum_information_eigenvalue=min_eigenvalue)

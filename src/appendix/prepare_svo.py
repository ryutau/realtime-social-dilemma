"""Prepare bootstrap-standardized second-mover cooperation rates for the Appendix."""
import json

import numpy as np
import pandas as pd

from src.common import BOOTSTRAP, SOURCE, INTERMEDIATE, load_pd


CONDITIONS = ("RT", "SQ")
SVO_SEED = 2026093006
TOLERANCE = 1e-9


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

    estimates, methods, draws = standardize(data, BOOTSTRAP, SVO_SEED)
    result = {"standardized": estimates, "standardization_methods": methods}
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INTERMEDIATE / "svo_standardization_draws.npz", **draws)
    path = INTERMEDIATE / "svo_standardization.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


def sigmoid(eta):
    return 1 / (1 + np.exp(-np.clip(eta, -700, 700)))


def classify_separation(x, y, counts):
    """Complete/quasi separation for an intercept and one scalar predictor."""
    n = len(x)
    positive = (counts * y).sum(axis=1)
    all_one, all_zero = positive == n, positive == 0
    mixed = ~(all_one | all_zero)
    present = counts > 0
    lo0 = np.where(present & (y == 0), x, np.inf).min(axis=1)
    hi0 = np.where(present & (y == 0), x, -np.inf).max(axis=1)
    lo1 = np.where(present & (y == 1), x, np.inf).min(axis=1)
    hi1 = np.where(present & (y == 1), x, -np.inf).max(axis=1)
    complete = mixed & ((hi0 < lo1) | (hi1 < lo0))
    quasi = mixed & ~complete & (
        ((hi0 == lo1) & ((lo0 < hi0) | (hi1 > lo1)))
        | ((hi1 == lo0) & ((lo1 < hi1) | (hi0 > lo0))))
    rank_deficient = mixed & (lo0 == hi0) & (lo1 == hi1) & (lo0 == lo1)
    return {"all_C": all_one, "all_D": all_zero, "complete_mixed": complete,
            "quasi_mixed": quasi, "rank_deficient_mixed": rank_deficient}


def fit_logit_batch(x, y, counts, tolerance=TOLERANCE):
    """Unpenalised binomial MLE; weighted counts exactly encode pair resampling."""
    assert np.all(counts.sum(axis=1) == len(x))
    classification = classify_separation(x, y, counts)
    assert not any(classification[k].any() for k in
                   ("complete_mixed", "quasi_mixed", "rank_deficient_mixed")), \
        "Unexpected non-single-outcome separation or unidentified slope: stop, do not penalise/drop."
    finite = ~(classification["all_C"] | classification["all_D"])
    coefs = np.zeros((len(counts), 2))
    coefs[classification["all_C"], 0] = np.inf
    coefs[classification["all_D"], 0] = -np.inf
    w = counts[finite].astype(float)
    if len(w) == 0:
        return coefs, dict(max_abs_score=0.0, iterations=0, finite_fits=0,
                           boundary_fits=int(len(counts)))
    positive = (w * y).sum(axis=1)
    intercept = np.log((positive + 0.5) / (len(x) - positive + 0.5))
    slope = np.zeros_like(intercept)
    for iteration in range(100):
        eta = intercept[:, None] + slope[:, None] * x
        p = sigmoid(eta)
        residual = w * (y - p)
        g0, g1 = residual.sum(axis=1), (residual * x).sum(axis=1)
        score = np.maximum(np.abs(g0), np.abs(g1))
        max_score = float(score.max())
        if max_score < tolerance:
            break
        variance = w * p * (1 - p)
        h00 = variance.sum(axis=1)
        h01 = (variance * x).sum(axis=1)
        h11 = (variance * x * x).sum(axis=1)
        det = h00 * h11 - h01 * h01
        assert np.all(det > 0), "Singular observed information in a purported finite fit."
        step0 = (h11 * g0 - h01 * g1) / det
        step1 = (h00 * g1 - h01 * g0) / det
        scale = np.minimum(1, 10 / np.maximum(np.maximum(np.abs(step0), np.abs(step1)), 1e-12))
        scale[score < tolerance] = 0
        objective = (w * (y * eta - np.logaddexp(0, eta))).sum(axis=1)
        for _ in range(35):
            new0, new1 = intercept + scale * step0, slope + scale * step1
            trial = new0[:, None] + new1[:, None] * x
            trial_objective = (w * (y * trial - np.logaddexp(0, trial))).sum(axis=1)
            bad = trial_objective < objective - 1e-12
            if not bad.any():
                break
            scale[bad] *= 0.5
        else:
            raise RuntimeError("Unpenalised logit line search failed.")
        intercept, slope = new0, new1
    else:
        raise RuntimeError(f"Unpenalised logit failed to converge: score={max_score}")
    coefs[finite, 0], coefs[finite, 1] = intercept, slope
    return coefs, dict(max_abs_score=max_score, iterations=iteration,
                       finite_fits=int(finite.sum()), boundary_fits=int((~finite).sum()))


def predict(coefs, x):
    result = sigmoid(coefs[:, 0, None] + coefs[:, 1, None] * x)
    # Assign exact boundary probabilities; do not use clipped near-one values.
    result[np.isposinf(coefs[:, 0])] = 1.0
    result[np.isneginf(coefs[:, 0])] = 0.0
    return result


def standardize(d, repetitions, seed):
    batch_size = 256
    groups = {c: d[d.condition.eq(c)] for c in CONDITIONS}
    raw_x = {c: groups[c].svo.to_numpy() for c in CONDITIONS}
    y = {c: groups[c].C.to_numpy() for c in CONDITIONS}
    n = {c: len(groups[c]) for c in CONDITIONS}
    # A fixed affine rescaling improves numerical conditioning, without changing the model.
    center, scale = float(d.svo.mean()), 10.0
    x = {c: (raw_x[c] - center) / scale for c in CONDITIONS}
    knots = np.unique(d.svo.to_numpy())
    scaled_knots = (knots - center) / scale
    indices = {c: np.searchsorted(knots, raw_x[c]) for c in CONDITIONS}
    reference_weights = sum(0.5 * np.bincount(indices[c], minlength=len(knots)) / n[c]
                            for c in CONDITIONS)
    assert np.isclose(reference_weights.sum(), 1, atol=1e-14, rtol=0)

    point, original_coefficients, original_predictions, diagnostics = {}, {}, {}, []
    for c in CONDITIONS:
        coefficients, dg = fit_logit_batch(x[c], y[c], np.ones((1, n[c])))
        assert np.isfinite(coefficients).all()
        a, b = coefficients[0]
        original_coefficients[c] = {
            "intercept_at_center": float(a), "slope_per_10_degrees": float(b),
            "intercept_at_zero_degrees": float(a - b * center / scale),
            "slope_per_degree": float(b / scale),
        }
        original_predictions[c] = predict(coefficients, scaled_knots)[0]
        point[c] = float(original_predictions[c] @ reference_weights)
        diagnostics.append(dict(condition=c, kind="original", first_rep=0, reps=1, **dg))
        # The unpenalised intercept score implies fitted means match observed means.
        assert abs(predict(coefficients, x[c])[0].mean() - y[c].mean()) < 1e-10
    point["RT_minus_SQ"] = point["RT"] - point["SQ"]

    streams = np.random.SeedSequence(seed).spawn(2)
    counts = {c: np.random.default_rng(stream).multinomial(
        n[c], np.full(n[c], 1/n[c]), size=repetitions).astype(np.uint8)
        for c, stream in zip(CONDITIONS, streams)}
    classification = {c: classify_separation(raw_x[c], y[c], counts[c]) for c in CONDITIONS}
    separation_counts = {c: {k: int(v.sum()) for k, v in classification[c].items()} for c in CONDITIONS}
    assert all(separation_counts[c][k] == 0 for c in CONDITIONS
               for k in ("complete_mixed", "quasi_mixed", "rank_deficient_mixed"))
    draws = {c: np.empty(repetitions) for c in CONDITIONS}
    for first in range(0, repetitions, batch_size):
        last = min(first + batch_size, repetitions)
        weights = np.zeros((last - first, len(knots)))
        for c in CONDITIONS:
            for col, knot in enumerate(indices[c]):
                weights[:, knot] += 0.5 * counts[c][first:last, col] / n[c]
        assert np.allclose(weights.sum(axis=1), 1, atol=1e-14, rtol=0)
        for c in CONDITIONS:
            coefficients, dg = fit_logit_batch(x[c], y[c], counts[c][first:last])
            draws[c][first:last] = (predict(coefficients, scaled_knots) * weights).sum(axis=1)
            diagnostics.append(dict(condition=c, kind="bootstrap", first_rep=first,
                                    reps=last-first, **dg))
    assert all(np.isfinite(z).all() for z in draws.values())
    assert all(np.all((draws[c] >= 0) & (draws[c] <= 1 + 1e-14)) for c in CONDITIONS)
    for c in CONDITIONS:
        # Floating sums of reference weights can differ from unity by machine epsilon.
        draws[c][classification[c]["all_C"]] = 1.0
        draws[c][classification[c]["all_D"]] = 0.0
    draws["RT_minus_SQ"] = draws["RT"] - draws["SQ"]
    frame = pd.DataFrame(diagnostics)
    assert frame.max_abs_score.max() < TOLERANCE
    results = {key: {
        "estimate": point[key], "ci95_percentile": np.quantile(draws[key], [.025, .975]).tolist(),
        "bootstrap_mean": float(draws[key].mean()), "bootstrap_sd": float(draws[key].std(ddof=1)),
    } for key in draws}
    metadata = {
        "bootstrap_repetitions": repetitions, "seed": seed,
        "separation_counts": separation_counts,
        "max_abs_finite_score": float(frame.max_abs_score.max()),
        "original_coefficients": original_coefficients,
        "baseline_reference": "Equal weights for the empirical SVO distributions in RT and SQ",
        "bootstrap_reference": "Refit both models and recompute the empirical reference in each draw",
        "bootstrap_unit": "One eligible second mover per pair, sampled within condition",
        "boundary_rule": "All-cooperate or all-defect draws use constant probabilities of 1 or 0",
    }
    return results, metadata, draws


if __name__ == "__main__":
    main()

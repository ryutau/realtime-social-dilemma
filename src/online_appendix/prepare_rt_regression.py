"""Prepare the RT two-step estimates and within-session pair bootstrap draws."""

from dataclasses import dataclass
import json
import time

import numpy as np
import pandas as pd
from scipy import optimize, special

from src.common import BOOTSTRAP, SOURCE, INTERMEDIATE, SEED, load_pd, rng
from src.online_appendix.logistic import LABELS, predictor_matrix, standardization

GRID = np.arange(0., 62., 2.)


def main():
    data = load_pd("main").set_index("label")
    for measure in ("svo_params", "ra_params", "demographics"):
        data = data.join(pd.read_csv(SOURCE / f"main_{measure}.csv", index_col=0),
                         how="left", validate="one_to_one")
    started = time.monotonic()
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    design = prepare(data)
    unit_counts = np.ones(len(design.pair_ids))
    original = Objective(design, unit_counts, *survival(design, unit_counts))
    p = len(LABELS)
    origin = np.zeros(p)
    point = fit_objective(original, [origin, np.repeat(0.25, p), np.repeat(-0.25, p)])
    if point["status"] != "ok":
        raise RuntimeError(f"Original two-step fit failed: {point['status']}")
    beta = point["beta"]
    counts = make_counts(design, BOOTSTRAP, SEED)
    np.savez_compressed(INTERMEDIATE / "rt_regression_pair_counts.npz", counts=counts, sessions=design.sessions)
    estimates = np.full((BOOTSTRAP, p), np.nan)
    diagnostics = []
    for index, pair_counts in enumerate(counts):
        detail = {"replicate": index + 1, "first_defectors": int(pair_counts @ (1. - design.action))}
        try:
            obj = Objective(design, pair_counts, *survival(design, pair_counts))
            result = fit_objective(obj, [beta, origin])
            if result["status"] != "ok":
                retry = fit_objective(obj, [beta * 2., -beta * .5, beta * .5])
                if retry.get("objective", np.inf) < result.get("objective", np.inf):
                    result = retry
            if result["status"] == "ok":
                estimates[index] = result["beta"]
            detail.update({key: value for key, value in result.items() if key != "beta"})
        except ValueError as exc:
            detail["status"] = str(exc)
        diagnostics.append(detail)
        completed = index + 1
        if completed % 5000 == 0 or completed == BOOTSTRAP:
            okay = int(np.isfinite(estimates[:completed, 0]).sum())
            print(f"RT two-step: {completed:,}/{BOOTSTRAP:,}; finite fits {okay:,}; "
                  f"{time.monotonic() - started:.1f}s", flush=True)

    np.savez_compressed(INTERMEDIATE / "rt_regression_bootstrap_coefficients.npz", beta=estimates, labels=LABELS, point=beta)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(INTERMEDIATE / "rt_regression_diagnostics.csv", index=False)
    valid = np.isfinite(estimates).all(axis=1)
    summary = {
        "experiment": "main", "master_seed": SEED, "attempted": BOOTSTRAP,
        "valid_finite_fits": int(valid.sum()), "status_counts": diag.status.value_counts().to_dict(),
        "pairs": len(design.pair_ids),
        "session_pair_counts": {str(int(s)): int((design.sessions == s).sum())
                                for s in np.unique(design.sessions)},
        "step1_first_movers": len(design.pair_ids), "step2_first_movers": len(design.xf),
        "step2_second_movers": len(design.xs), "step2_n": len(design.xf) + len(design.xs),
        "first_defectors": int((1. - design.action).sum()),
        "point_max_abs_score": point["gradient"], "point_minimum_hessian_eigenvalue": point["min_eigenvalue"],
        "standardization_reference": "Eligible full-experiment means and sample SDs held fixed in resamples",
        "standardization": design.standardization,
        "failure_policy": "Record failed attempts without redrawing. Report inference only for a complete bootstrap."
    }
    (INTERMEDIATE / "rt_regression_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not valid.all():
        raise RuntimeError(f"Bootstrap failed on {int((~valid).sum())} draws. See {INTERMEDIATE}.")


@dataclass
class Design:
    name: str
    pair_ids: np.ndarray
    sessions: np.ndarray
    action: np.ndarray
    bin_index: np.ndarray
    xf: np.ndarray
    yf: np.ndarray
    xs: np.ndarray
    first_pair: np.ndarray
    second_pair: np.ndarray
    standardization: dict

def prepare(data):
    columns, scales = standardization(data, "main")
    rt = data[data.condition.eq("RT")].copy()
    groups = list(rt.groupby(["session_no", "group_id"], sort=True))
    first, second, ids, sessions = [], [], [], []
    for (session, gid), group in groups:
        assert len(group) == 2 and int(group.is_first_mover.sum()) == 1
        f = group[group.is_first_mover].iloc[0]
        s = group[~group.is_first_mover].iloc[0]
        assert f.decision_time < s.decision_time
        assert np.isclose(s.partner_decision_time, f.decision_time)
        first.append(f)
        second.append(s)
        ids.append(gid)
        sessions.append(session)
    first, second = pd.DataFrame(first), pd.DataFrame(second)
    x_first = predictor_matrix(first, columns, scales)
    x_second = predictor_matrix(second, columns, scales)
    mf = np.isfinite(x_first).all(axis=1) & first.is_coop.notna().to_numpy()
    ms = np.isfinite(x_second).all(axis=1) & second.is_coop.notna().to_numpy()
    xf, xs = x_first[mf], x_second[ms]
    times = first.decision_time.to_numpy(float)
    assert ((times > 0) & (times < 60)).all()
    assert not np.isclose(times[:, None], GRID[None, :], atol=1e-10, rtol=0).any()
    return Design("main", np.array(ids), np.array(sessions),
                  first.is_coop.to_numpy(float), np.digitize(times, GRID)-1,
                  xf, first.is_coop.to_numpy(float)[mf], xs,
                  np.flatnonzero(mf), np.flatnonzero(ms), scales)

def survival(design, counts):
    """Exact multiplicity-weighted version of the original 2-second-bin Step 1."""
    total = np.bincount(design.bin_index, weights=counts, minlength=30)
    coop = np.bincount(design.bin_index, weights=counts*design.action, minlength=30)
    if coop.sum() == 0 or (total-coop).sum() == 0:
        raise ValueError("missing_action")
    observed = np.r_[0., np.cumsum(total)] / total.sum()
    latent = 1. - np.sqrt(np.maximum(0., 1.-observed))
    conditional = np.divide(coop, total, out=np.zeros_like(coop), where=total>0)
    cm = np.diff(latent)*conditional
    dm = np.diff(latent)*(1.-conditional)
    if cm.sum() <= 0 or dm.sum() <= 0:
        raise ValueError("missing_action")
    sc = np.clip(1.-np.r_[0., np.cumsum(cm)]/cm.sum(), 0., 1.)
    sd = np.clip(1.-np.r_[0., np.cumsum(dm)]/dm.sum(), 0., 1.)
    return sc, sd

class Objective:
    def __init__(self, design, counts, sc, sd):
        mf = counts[design.first_pair] > 0
        ms = counts[design.second_pair] > 0
        self.xf, self.yf = design.xf[mf], design.yf[mf]
        self.xs = design.xs[ms]
        self.wf = counts[design.first_pair][mf]
        self.ws = counts[design.second_pair][ms]
        idx = design.bin_index[design.second_pair][ms]
        self.sc, self.sd = sc[idx], sd[idx]
        if ((self.sc+self.sd) <= 0).any():
            raise ValueError("zero_survival")
        with np.errstate(divide="ignore"):
            self.lc, self.ld = np.log(self.sc), np.log(self.sd)
        self.rank = np.linalg.matrix_rank(np.vstack([self.xf, self.xs]))

    def evaluate(self, beta, hessian=False):
        ef, es = self.xf@beta, self.xs@beta
        qf, qs = special.expit(ef), special.expit(es)
        logmix = np.logaddexp(self.lc+es, self.ld)-np.logaddexp(0., es)
        value = (self.wf@(np.logaddexp(0., ef)-self.yf*ef)
                 - self.ws@logmix)
        posterior = special.expit(es+self.lc-self.ld)
        gradient = (self.xf.T@(self.wf*(qf-self.yf))
                    + self.xs.T@(self.ws*(qs-posterior)))
        if not hessian:
            return float(value), gradient
        vf = self.wf*qf*(1.-qf)
        vs = self.ws*(qs*(1.-qs)-posterior*(1.-posterior))
        h = (self.xf.T*vf)@self.xf + (self.xs.T*vs)@self.xs
        return float(value), gradient, h

def fit_objective(objective, starts):
    if objective.rank < len(LABELS):
        return dict(status="rank_deficient")
    candidates = []
    for start in starts:
        result = optimize.minimize(
            objective.evaluate, start, jac=True, method="BFGS",
            options={"gtol": 1e-7, "maxiter": 500})
        value, gradient, h = objective.evaluate(result.x, True)
        grad = float(np.max(np.abs(gradient)))
        eig = np.linalg.eigvalsh(h)
        # Refine converged ordinary optima with exact-Hessian Newton steps.
        if np.isfinite(value) and grad < 1e-3 and eig[0] > 1e-7:
            beta = result.x.copy()
            for _ in range(6):
                step = np.linalg.solve(h, gradient)
                nv, ng, nh = objective.evaluate(beta-step, True)
                if nv > value+1e-9:
                    break
                beta -= step
                value, gradient, h = nv, ng, nh
                if np.max(np.abs(gradient)) < 1e-8:
                    break
            grad = float(np.max(np.abs(gradient)))
            eig = np.linalg.eigvalsh(h)
        else:
            beta = result.x
        if np.isfinite(beta).all() and np.isfinite(value):
            candidates.append((value, beta.copy(), grad, float(eig[0]),
                               float(np.max(eig)/max(eig[0], 1e-300)), result.success))
    if not candidates:
        return dict(status="numerical_failure")
    value, beta, grad, mineig, cond, success = min(candidates, key=lambda c:c[0])
    if grad > 1e-5:
        status = "nonconvergence"
    elif mineig <= 1e-7 or cond > 1e10:
        status = "boundary_or_ill_conditioned"
    else:
        status = "ok"
    return dict(status=status, beta=beta, objective=float(value), gradient=grad,
                min_eigenvalue=mineig, condition_number=cond,
                max_abs_beta=float(np.max(np.abs(beta))), optimizer_success=bool(success))

def make_counts(design, repetitions, seed):
    counts = np.zeros((repetitions, len(design.pair_ids)), dtype=np.uint16)
    for session in np.unique(design.sessions):
        loc = np.flatnonzero(design.sessions == session)
        stream = f"regression/twostep/{design.name}/session/{int(session)}"
        counts[:,loc] = rng(seed,stream).multinomial(
            len(loc), np.full(len(loc),1./len(loc)), size=repetitions)
    return counts


if __name__ == "__main__":
    main()

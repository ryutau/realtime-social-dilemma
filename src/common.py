"""Shared sample definitions, random streams, and statistical estimators."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data/source"
INTERMEDIATE = PROJECT / "data/intermediate"
SEED = 42
BOOTSTRAP = 50_000
CONDITIONS = ("RT", "SQ", "SM")
BINS = np.arange(0, 62, 2)


def load_pd(study):
    """Apply completion exclusions while preserving source-row order."""
    data = pd.read_csv(SOURCE / f"{study}_pd_data.csv")
    completion = pd.read_csv(SOURCE / f"{study}_completion.csv").set_index("label")
    eligible = data.label.map(completion.completed_postgame_tasks)
    if eligible.isna().any() or eligible.dtype != bool:
        raise ValueError("Each participant needs a True/False completion record")
    # Main keeps completed participants, including a completed partner.
    # Pilot keeps a pair only when both participants completed the tasks.
    if study == "pilot":
        eligible = eligible.groupby(data.group_id).transform("all")
    return data.loc[eligible].reset_index(drop=True)


def load_main():
    data = load_pd("main")
    required = {'label', 'group_id', 'condition', 'is_coop', 'is_partner_coop',
                'is_first_mover', 'decision_time', 'exp_no'}
    if not required.issubset(data.columns) or data[list(required)].isna().any().any():
        raise ValueError('Missing main-experiment columns or observations')
    if data.label.duplicated().any() or not data.exp_no.eq(2).all():
        raise ValueError('Expected one row per main-experiment participant')
    for column in ('is_coop', 'is_partner_coop', 'is_first_mover'):
        if data[column].dtype != bool:
            raise ValueError(f'{column} must contain True/False values')
    if set(data.condition) != set(CONDITIONS):
        raise ValueError('Expected RT, SQ, and SM conditions')
    pairs = data.groupby(['condition', 'group_id'], sort=True)
    sizes = pairs.size()
    if not sizes.isin([1, 2]).all() or not sizes[sizes.eq(1)].index.get_level_values('condition').isin(['SM']).all():
        raise ValueError('Expected two participants per pair, with singletons allowed in SM')
    complete = complete_pair_members(data).groupby(['condition', 'group_id'], sort=True)
    if not complete.is_first_mover.sum().eq(1).all():
        raise ValueError('Expected one first mover per complete pair')
    if not complete.is_coop.sum().eq(complete.is_partner_coop.sum()).all():
        raise ValueError('Inconsistent partner actions')
    return data.sort_values(['condition', 'group_id', 'label'], kind='stable').reset_index(drop=True)


def load_pilot():
    """Read eligible pilot pairs and check their paired records."""
    data = load_pd("pilot")
    required = {
        "label", "exp_no", "condition", "session_no", "group_no", "group_id",
        "is_coop", "is_partner_coop", "is_first_mover", "decision_time",
        "partner_decision_time",
    }
    if not required.issubset(data.columns) or data[list(required)].isna().any().any():
        raise ValueError("Missing pilot-experiment columns or observations")
    if data.label.duplicated().any() or not data.exp_no.eq(1).all():
        raise ValueError("Expected one row per pilot-experiment participant")
    if set(data.condition) != {"RT", "SM"}:
        raise ValueError("Expected RT and SM in the pilot experiment")
    for column in ("is_coop", "is_partner_coop", "is_first_mover"):
        if data[column].dtype != bool:
            raise ValueError(f"{column} must contain True/False values")
    for column in ("decision_time", "partner_decision_time"):
        if not np.isfinite(data[column]).all() or data[column].lt(0).any():
            raise ValueError("Pilot decision times must be finite and nonnegative")
    pairs = data.groupby(["condition", "group_id"], sort=True)
    if not pairs.size().eq(2).all() or not pairs.is_first_mover.sum().eq(1).all():
        raise ValueError("Expected two participants and one first mover per pilot pair")
    if not pairs.is_coop.sum().eq(pairs.is_partner_coop.sum()).all():
        raise ValueError("Inconsistent pilot partner actions")
    for _, pair in pairs:
        pair = pair.sort_values("label", kind="stable")
        if not np.array_equal(pair.is_partner_coop.to_numpy(), pair.is_coop.to_numpy()[::-1]):
            raise ValueError("Pilot partner actions do not match the paired participant")
        if not np.allclose(pair.partner_decision_time.to_numpy(), pair.decision_time.to_numpy()[::-1],
                           rtol=0, atol=1e-9):
            raise ValueError("Pilot partner decision times do not match")
        first = pair.loc[pair.is_first_mover].iloc[0]
        if first.decision_time > first.partner_decision_time:
            raise ValueError("Pilot first mover did not act first")
    return data.sort_values(["condition", "group_id", "label"], kind="stable").reset_index(drop=True)


def complete_pair_members(data):
    sizes = data.groupby(['condition', 'group_id'])['label'].transform('size')
    return data.loc[sizes.eq(2)]


def pair_rows(data, condition):
    pairs = complete_pair_members(data.loc[data.condition.eq(condition)])
    return pairs.loc[pairs.is_first_mover].sort_values(
        'group_id', kind='stable').reset_index(drop=True)


def rng(seed, stream):
    # Fixed names make streams independent of execution order and Python hash randomization.
    digest = hashlib.sha256(stream.encode('utf-8')).digest()
    words = np.frombuffer(digest, dtype='<u4').tolist()
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *words])))


def bootstrap_means(values, repetitions, seed, stream):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires a nonempty finite vector')
    random = rng(seed, stream)
    draws = np.empty(repetitions)
    for start in range(0, repetitions, 2000):
        stop = min(start + 2000, repetitions)
        indices = random.integers(len(values), size=(stop - start, len(values)))
        draws[start:stop] = values[indices].mean(axis=1)
    return draws


def interval(draws):
    return np.quantile(draws, [.025, .975], axis=0, method='linear')


def summary(estimate, draws):
    return {"estimate": float(estimate), "ci95": interval(draws).tolist()}


def pooled_z(first, second):
    successes = [int(r.sum()) for r in (first, second)]
    sizes = [len(r) for r in (first, second)]
    pooled = sum(successes) / sum(sizes)
    se = np.sqrt(pooled * (1 - pooled) * sum(1 / n for n in sizes))
    z = (first.mean() - second.mean()) / se
    return {"cooperations": successes, "n": sizes, "z": float(z),
            "p_two_sided": float(2 * norm.sf(abs(z))),
            "expected_successes_under_null": [pooled * n for n in sizes]}

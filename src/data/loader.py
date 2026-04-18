import hashlib
import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from .utils import DATA_RAW, DATA_PROCESSED

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Shared utilities ─────────────────────────────────────────────────────────

def _anonymize_label(input_string):
    """Hash a Prolific participant label with MD5."""
    return hashlib.md5(input_string.encode("utf-8")).hexdigest()


def _otree_col_dict(*, participant=None, player=None, group=None,
                    session=None, subsession=None, renames=None):
    """Build a column-rename dict for oTree CSV exports.

    >>> _otree_col_dict(participant=["label"], player=["payoff"],
    ...                 renames={"group.id_in_subsession": "group_no"})
    {'participant.label': 'label', 'player.payoff': 'payoff',
     'group.id_in_subsession': 'group_no'}
    """
    d = {}
    for prefix, cols in [("participant", participant), ("player", player),
                         ("group", group), ("session", session),
                         ("subsession", subsession)]:
        if cols:
            d |= {f"{prefix}.{c}": c for c in cols}
    if renames:
        d |= renames
    return d


def _derive_pd_game_cols(df, group_cols, action_col="action",
                         partner_action_col="partner_action"):
    """Derive standard PD game columns in-place.

    Creates: is_coop, is_partner_coop, partner_decision_time, is_first_mover.
    """
    df["is_coop"] = df[action_col] == "Invest"
    df["is_partner_coop"] = df[partner_action_col] == "Invest"
    df["partner_decision_time"] = df.groupby(group_cols)[
        "decision_time"
    ].transform(lambda x: x.sum() - x)
    df["is_first_mover"] = df["decision_time"] < df["partner_decision_time"]


# ── Prolific demographics ────────────────────────────────────────────────────

_PROLIFIC_DEMO_COLS = {
    "Participant id": "label",
    "Time taken": "time_taken",
    "Total approvals": "cnt_approvals",
    "Age": "age",
    "Sex": "sex",
    "Ethnicity simplified": "ethnicity",
}


def _clean_demographics(df):
    """Shared demographics cleaning for Prolific exports."""
    for placeholder in ["DATA_EXPIRED", "CONSENT_REVOKED"]:
        df.replace(placeholder, np.nan, inplace=True)
    df["is_male"] = df["sex"] == "Male"
    for col in ["age", "cnt_approvals", "time_taken"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


# ── SVO ───────────────────────────────────────────────────────────────────────

_SVO_COL_DICT = _otree_col_dict(
    participant=["label"], player=["choice"], subsession=["round_number"],
)


def _load_svo_options(svo_csv_path):
    """Load the SVO option table and compute choice ranks."""
    opts = pd.read_csv(svo_csv_path).reset_index()
    opts["choice"] = opts.groupby("round_number")["index"].rank() - 1
    return opts


def _compute_svo_degree(svo_raw, svo_options):
    """Compute SVO angle (degrees) from slider choices and option table."""
    merged = svo_raw.merge(
        svo_options.set_index("index"), on=["round_number", "choice"]
    )
    pivot = pd.concat(
        {
            tgt: merged.pivot(
                index="label", columns="round_number", values=f"to_{tgt}"
            ).mean(axis=1) - 50
            for tgt in ["self", "other"]
        },
        axis=1,
    )
    svo = pivot.apply(
        lambda x: math.degrees(math.atan(x["other"] / x["self"])), axis=1
    )
    svo.name = "svo"
    return svo


def _preprocess_svo(csv_path, svo_options, output_path, max_round=6):
    """SVO preprocessing for a single CSV file (used by main experiment)."""
    svo_raw = (
        pd.read_csv(csv_path, usecols=_SVO_COL_DICT.keys())
        .rename(columns=_SVO_COL_DICT)
        .dropna(subset=["choice"])
    )
    if max_round is not None:
        svo_raw = svo_raw.query("round_number <= @max_round")
    svo_raw = svo_raw.copy()
    svo_raw["label"] = svo_raw["label"].map(_anonymize_label)

    svo_degree = _compute_svo_degree(svo_raw, svo_options)
    all_labels = svo_raw[["label"]].drop_duplicates().set_index("label")
    all_labels.join(svo_degree).to_csv(output_path)


# ── Risk-aversion model (pilot only) ─────────────────────────────────────────

def _subjective_utility(lottery, alpha, beta):
    p, A, v = lottery
    return (p - beta * A / 2) * (v**alpha)


def _choice_probability(lottery_pair, params):
    alpha, beta, gamma = params
    su = [_subjective_utility(lot, alpha, beta) for lot in lottery_pair]
    normalised_gap = (su[1] - su[0]) / (su[0] + su[1])
    return 1 / (1 + np.exp(gamma * normalised_gap))


def _neg_log_likelihood(params, choices_df):
    nll = 0
    for record in choices_df.to_dict(orient="records"):
        p_first = _choice_probability(record["lottery_pair"], params)
        p_choice = record["choice"] * p_first + (1 - record["choice"]) * (1 - p_first)
        nll += -np.log(p_choice)
    return nll


def _fit_su_model(choices_df):
    x0 = [
        np.random.uniform(0.6, 1.3),
        np.random.uniform(-0.3, 0.3),
        15 * np.random.exponential(),
    ]
    bounds = [(0.1, 1.9), (-1, 1), (0, np.inf)]
    res = optimize.minimize(
        _neg_log_likelihood, args=choices_df, method="L-BFGS-B", x0=x0, bounds=bounds,
    )
    return res.x


# ── Pilot experiment preprocessing ───────────────────────────────────────────

_PILOT_CONDITIONS = {"opaque": ("02-01", "SM"), "transparent": ("02-02", "RT")}


def preprocess_pilot(raw_dir):
    """Run full preprocessing pipeline for the pilot experiment.

    Parameters
    ----------
    raw_dir : str or Path
        Root directory containing exp1_202302/ subdirectory.
    """
    exp_dir = f"{raw_dir}/exp1_202302"
    svo_options = _load_svo_options(f"{exp_dir}/SVO.csv")

    _preprocess_pilot_pd_data(exp_dir)
    _preprocess_pilot_demographics(exp_dir)
    _preprocess_pilot_svo(exp_dir, svo_options)
    _preprocess_pilot_ra(exp_dir)


def _preprocess_pilot_pd_data(exp_dir):
    col_dict = _otree_col_dict(
        participant=["label"],
        player=["payoff", "on_displayed_time", "on_decide_time",
                "on_decide_time_sim", "action", "partner_action"],
        renames={"group.id_in_subsession": "group_no"},
    )
    dfs = []
    for condition, (date, cond_str) in _PILOT_CONDITIONS.items():
        df = (
            pd.read_csv(
                f"{exp_dir}/2023-{date}_{condition}/CPD_{condition}_2023-{date}.csv",
                usecols=col_dict.keys(),
            )
            .rename(columns=col_dict)
            .dropna(subset=["label", "action"])
            .query("payoff > 0")
            .set_index("label")
        )
        # decision_time: use on_decide_time - on_displayed_time if available,
        # otherwise fall back to 60 + on_decide_time_sim (simultaneous decision)
        df["decision_time"] = df.apply(
            lambda x: x.on_decide_time - x.on_displayed_time
            if not np.isnan(x.on_decide_time)
            else 60 + x.on_decide_time_sim,
            axis=1,
        )
        _derive_pd_game_cols(df, group_cols="group_no")
        df["exp_no"] = 1
        df["session_no"] = 1
        df["condition"] = cond_str
        df["group_id"] = cond_str + "_1_" + df["group_no"].astype(str)
        df.index = df.index.map(_anonymize_label)
        df.drop(
            columns=["on_displayed_time", "on_decide_time", "on_decide_time_sim",
                     "action", "partner_action"],
            inplace=True,
        )
        dfs.append(df)
    pd.concat(dfs).to_csv(DATA_PROCESSED / "pilot_pd_data.csv")


def _preprocess_pilot_demographics(exp_dir):
    completion_code = "C1MW0K8M"
    col_dict = {**_PROLIFIC_DEMO_COLS, "Completion code": "comp_code"}
    dfs = []
    for condition, (date, _) in _PILOT_CONDITIONS.items():
        df = (
            pd.read_csv(
                f"{exp_dir}/2023-{date}_{condition}/prolific_demographics_2023-{date}.csv",
                usecols=col_dict.keys(),
            )
            .rename(columns=col_dict)
            .query("comp_code == @completion_code")
            .drop(columns=["comp_code"])
            .set_index("label")
        )
        df = _clean_demographics(df)
        df.index = df.index.map(_anonymize_label)
        dfs.append(df)
    pd.concat(dfs).to_csv(DATA_PROCESSED / "pilot_demographics.csv")


def _preprocess_pilot_svo(exp_dir, svo_options):
    dfs = []
    for condition, (date, _) in _PILOT_CONDITIONS.items():
        csv_path = f"{exp_dir}/2023-{date}_{condition}/svo_2023-{date}.csv"
        svo_raw = (
            pd.read_csv(csv_path, usecols=_SVO_COL_DICT.keys())
            .rename(columns=_SVO_COL_DICT)
            .dropna(subset=["choice"])
            .query("round_number <= 6")
            .copy()
        )
        svo_raw["label"] = svo_raw["label"].map(_anonymize_label)
        dfs.append(svo_raw)
    all_svo = pd.concat(dfs, ignore_index=True)
    svo_degree = _compute_svo_degree(all_svo, svo_options)
    all_labels = all_svo[["label"]].drop_duplicates().set_index("label")
    all_labels.join(svo_degree).to_csv(DATA_PROCESSED / "pilot_svo_params.csv")


def _preprocess_pilot_ra(exp_dir):
    lottery_choices = [f"chose_lottery_{i}" for i in range(12)]
    ra_col_dict = _otree_col_dict(
        participant=["label"], subsession=["round_number"],
        player=lottery_choices,
    )
    lottery_list = pd.read_csv(f"{exp_dir}/lottery_list.csv")

    all_choice_dfs = []
    for condition, (date, _) in _PILOT_CONDITIONS.items():
        session_dir = f"{exp_dir}/2023-{date}_{condition}"

        # completion status
        prolific = pd.read_csv(
            f"{session_dir}/prolific_demographics_2023-{date}.csv"
        )
        code_map = {"C1MW0K8M": "completed", "CE8ENN8O": "missed", "C6AF2W24": "missed"}
        completion = prolific.set_index("Participant id")["Completion code"].map(code_map)

        ra_data = (
            pd.read_csv(
                f"{session_dir}/risk_ambiguity_2023-{date}.csv",
                usecols=ra_col_dict.keys(),
            )
            .rename(columns=ra_col_dict)
            .dropna(subset=["label"])
        )

        # filter to completed participants
        valid = ra_data.merge(
            completion.reset_index(), left_on="label", right_on="Participant id",
        )
        valid = valid.query("`Completion code` == 'completed'").drop(
            columns=["Participant id", "Completion code"],
        )

        # stack into tidy format
        stacked = valid.set_index(["label", "round_number"]).stack().reset_index()
        stacked["set"] = (stacked["round_number"] + 1) % 2
        stacked["id"] = stacked["level_2"].str.replace("chose_lottery_", "").astype(int)
        stacked["choice"] = stacked[0].astype(bool)
        all_choice_dfs.append(stacked[["label", "set", "id", "choice"]])

    choice_df = pd.concat(all_choice_dfs, ignore_index=True)
    choice_df = choice_df.merge(lottery_list, on=["set", "id"], how="left")
    choice_df["lottery_pair"] = choice_df.apply(
        lambda x: ((x.p, x.A, x.v), (1, 0, 10)), axis=1,
    )

    # estimate parameters per participant
    results = []
    for label, individual_df in choice_df.groupby("label"):
        np.random.seed(0)
        pars = _fit_su_model(individual_df)
        results.append({
            "label": _anonymize_label(label),
            "risk_aversion_modeling": 1 - pars[0],       # 1 - alpha; higher = more averse
            "ambiguity_aversion_modeling": pars[1],       # beta; higher = more averse
            "decision_temperature": 1 / pars[2] if pars[2] > 0 else np.inf,  # 1/gamma; higher = noisier
        })
    pd.DataFrame(results).set_index("label").to_csv(
        DATA_PROCESSED / "pilot_ra_params.csv"
    )


# ── Main experiment preprocessing ────────────────────────────────────────────

_MAIN_SESSION_CODE_TO_NO = {
    "slvfalvu": 1,  # SM1
    "1aii5uj2": 2,  # SM2
    "u3pxiuqe": 1,  # RT1
    "9e1it1rv": 2,  # RT2
    "7reoo3kl": 1,  # SQ1
    "pwzoezhd": 2,  # SQ2
}

_MAIN_PROLIFIC_SESSION_IDS = [
    "65f902b6be2a4085c3322ea7", "65fb0fb27a8a78f7c7577bba",
    "65fb1e76b59fe23a2ef69d00", "65fb2c8a60a47c45cdc2fc49",
    "65fb24a2125374d4889e6f28", "65fb17153e92080519cf347c",
]


def preprocess_main(raw_dir):
    """Run full preprocessing pipeline for the main experiment.

    Parameters
    ----------
    raw_dir : str or Path
        Root directory containing exp2_202403/ and exp1_202302/ subdirectories.
    """
    data_dir = f"{raw_dir}/exp2_202403/exp-rt-202403-raw-data"
    svo_options = _load_svo_options(f"{raw_dir}/exp1_202302/SVO.csv")

    _preprocess_main_pd_data(data_dir)
    _preprocess_main_demographics(data_dir)
    _preprocess_svo(
        f"{data_dir}/svo_2024-03-21.csv",
        svo_options,
        DATA_PROCESSED / "main_svo_params.csv",
        max_round=None,
    )
    _preprocess_main_ra(data_dir)
    _preprocess_main_willingness(data_dir)
    _preprocess_main_trust(data_dir)


def _load_main_condition(data_dir, condition):
    """Load one condition of main experiment PD data."""
    col_dict = _otree_col_dict(
        participant=["label"],
        player=["condition", "id_in_group", "payoff", "action",
                "partner_action", "is_dropout"],
        renames={
            "group.id_in_subsession": "group_no",
            "session.code": "session_code",
            "player.rt": "decision_time",
        },
    )
    df = (
        pd.read_csv(f"{data_dir}/Play_{condition}_2024-03-21.csv")
        [col_dict.keys()]
        .rename(columns=col_dict)
        .query("payoff > 0 and is_dropout == 0")
    )
    return df


def _preprocess_main_pd_data(data_dir):
    df = pd.concat(
        [_load_main_condition(data_dir, c) for c in ["RT", "SQ", "ST"]],
        ignore_index=True,
    )
    df["condition"] = df["condition"].replace("ST", "SM")
    df["label"] = df["label"].map(_anonymize_label)
    df["exp_no"] = 2
    df["session_no"] = df["session_code"].map(_MAIN_SESSION_CODE_TO_NO)
    _derive_pd_game_cols(
        df, group_cols=["condition", "session_no", "group_no"],
    )
    # override first_mover for SQ condition (determined by role, not timing)
    sq_mask = df["condition"] == "SQ"
    df.loc[sq_mask, "is_first_mover"] = df.loc[sq_mask, "id_in_group"] == 1

    df["group_id"] = (
        df["condition"] + "_" + df["session_no"].astype(str) + "_" + df["group_no"].astype(str)
    )
    tgt_cols = [
        "label", "exp_no", "condition", "session_no", "group_no", "group_id",
        "payoff", "is_coop", "decision_time", "is_partner_coop", "is_first_mover",
        "partner_decision_time",
    ]
    df[tgt_cols].set_index("label").to_csv(DATA_PROCESSED / "main_pd_data.csv")


def _preprocess_main_demographics(data_dir):
    col_dict = {**_PROLIFIC_DEMO_COLS, "Status": "status"}
    dfs = []
    for sid in _MAIN_PROLIFIC_SESSION_IDS:
        df = (
            pd.read_csv(
                f"{data_dir}/prolific_export_{sid}.csv", usecols=col_dict.keys(),
            )
            .rename(columns=col_dict)
            .query("status == 'APPROVED'")
            .drop(columns=["status"])
        )
        df = _clean_demographics(df)
        df["label"] = df["label"].map(_anonymize_label)
        dfs.append(df)
    pd.concat(dfs).set_index("label").to_csv(DATA_PROCESSED / "main_demographics.csv")


def _preprocess_main_ra(data_dir):
    col_dict = _otree_col_dict(
        participant=["label"],
        player=["risk_p", "ambiguity_p"],
        session=["code"],
        renames={"session.code": "session_code"},
    )
    df = (
        pd.read_csv(
            f"{data_dir}/risk_ambiguity_2024-03-21.csv", usecols=col_dict.keys(),
        )
        .rename(columns=col_dict)
        .query("session_code != 'slvfalvu'")  # SM1 missing due to technical issue
        .dropna(subset=["risk_p", "ambiguity_p"])
    )
    df["label"] = df["label"].map(_anonymize_label)
    df["risk_aversion_change_point"] = 1 - df["risk_p"] / 32
    df["ambiguity_aversion_change_point"] = 1 - df["ambiguity_p"] / 100
    df[["label", "risk_aversion_change_point", "ambiguity_aversion_change_point"]].set_index("label").to_csv(
        DATA_PROCESSED / "main_ra_params.csv"
    )


def _preprocess_main_willingness(data_dir):
    col_dict = _otree_col_dict(
        participant=["label"],
        player=["willingness_first_mover", "willingness_second_mover"],
    )
    df = (
        pd.read_csv(f"{data_dir}/Intro_SQ_2024-03-21.csv")
        [col_dict.keys()]
        .rename(columns=col_dict)
        .dropna(subset=["willingness_first_mover", "willingness_second_mover"])
    )
    df["label"] = df["label"].map(_anonymize_label)
    gap = df["willingness_first_mover"] - df["willingness_second_mover"]
    df["preferred_role"] = np.where(
        gap > 0, "first_mover", np.where(gap < 0, "second_mover", "indifferent")
    )
    cols = ["label", "willingness_first_mover", "willingness_second_mover", "preferred_role"]
    df[cols].set_index("label").to_csv(DATA_PROCESSED / "main_pd_willingness_data.csv")


def _preprocess_main_trust(data_dir):
    col_dict = _otree_col_dict(
        participant=["label"], player=["trust_choice"],
    )
    df = (
        pd.read_csv(f"{data_dir}/trust_2024-03-21.csv", usecols=col_dict.keys())
        .rename(columns=col_dict)
        .dropna(subset=["trust_choice"])
    )
    df["trust_choice"] = df["trust_choice"].astype(int)
    df["label"] = df["label"].map(_anonymize_label)
    df.set_index("label").to_csv(DATA_PROCESSED / "main_trust_data.csv")


# ── Loading (from processed CSVs in data/processed) ──────────────────────────

GAME_COLS = [
    "payoff", "is_coop", "decision_time",
    "is_partner_coop", "is_first_mover", "partner_decision_time",
]
STRUCTURE_COLS = ["condition", "session_no", "group_no"]
DEMO_COLS = ["age", "sex", "ethnicity", "is_male", "time_taken", "cnt_approvals"]

_PILOT_TARGETS = ["pd_data", "svo_params", "ra_params", "demographics"]
_MAIN_TARGETS = [
    "pd_data", "svo_params", "ra_params", "demographics",
    "trust_data", "pd_willingness_data",
]


def _join_csvs(file_paths, index_col=0):
    """Load and left-join a list of CSVs on their index."""
    merged = None
    for path in file_paths:
        df = pd.read_csv(path, index_col=index_col)
        merged = df if merged is None else merged.join(df, how="left")
    return merged


def load_pilot():
    """Load pilot experiment data from processed CSVs."""
    paths = [DATA_PROCESSED / f"pilot_{tgt}.csv" for tgt in _PILOT_TARGETS]
    return _join_csvs(paths)


def load_main():
    """Load main experiment data from processed CSVs."""
    paths = [DATA_PROCESSED / f"main_{tgt}.csv" for tgt in _MAIN_TARGETS]
    return _join_csvs(paths, index_col="label")

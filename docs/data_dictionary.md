# Data Dictionary

## Overview

### Data flow

```
data/raw/          (source files from oTree & Prolific)
  ↓  preprocess_pilot() / preprocess_main()
data/processed/    (formatted CSVs, 10 files)
  ↓  load_pilot() / load_main()
DataFrame          (analysis-ready, one row per participant per round)
```

All functions are in `src/__exp_data_class.py`.

### Source files (`data/raw/`)

**Pilot experiment** — `data/raw/exp1_202302/`

| File | Contents |
|---|---|
| `2023-02-01_opaque/CPD_opaque_2023-02-01.csv` | oTree PD game data (SM condition) |
| `2023-02-01_opaque/prolific_demographics_2023-02-01.csv` | Prolific demographics (SM) |
| `2023-02-01_opaque/svo_2023-02-01.csv` | oTree SVO slider responses (SM) |
| `2023-02-01_opaque/risk_ambiguity_2023-02-01.csv` | oTree risk/ambiguity lottery choices (SM) |
| `2023-02-02_transparent/CPD_transparent_2023-02-02.csv` | oTree PD game data (RT condition) |
| `2023-02-02_transparent/prolific_demographics_2023-02-02.csv` | Prolific demographics (RT) |
| `2023-02-02_transparent/svo_2023-02-02.csv` | oTree SVO slider responses (RT) |
| `2023-02-02_transparent/risk_ambiguity_2023-02-02.csv` | oTree risk/ambiguity lottery choices (RT) |
| `SVO.csv` | SVO slider option definitions |
| `lottery_list.csv` | Lottery pair definitions for RA estimation |

**Main experiment** — `data/raw/exp2_202403/exp-rt-202403-raw-data/`

| File | Contents |
|---|---|
| `Play_RT_2024-03-21.csv` | oTree PD game data (RT condition) |
| `Play_SQ_2024-03-21.csv` | oTree PD game data (SQ condition) |
| `Play_ST_2024-03-21.csv` | oTree PD game data (SM condition, originally labelled ST) |
| `svo_2024-03-21.csv` | oTree SVO slider responses |
| `risk_ambiguity_2024-03-21.csv` | oTree risk/ambiguity responses |
| `Intro_SQ_2024-03-21.csv` | Willingness & role preference questionnaire |
| `trust_2024-03-21.csv` | Trust game responses |
| `prolific_export_{session_id}.csv` | Prolific demographics (6 session files) |

### Processed files (`data/processed/`)

**Pilot** (4 files):
`pilot_pd_data.csv`, `pilot_svo_params.csv`, `pilot_ra_params.csv`, `pilot_demographics.csv`

**Main** (6 files):
`main_pd_data.csv`, `main_svo_params.csv`, `main_ra_params.csv`, `main_demographics.csv`, `main_trust_data.csv`, `main_pd_willingness_data.csv`

---

## Processed Column Definitions

Each row represents one participant's decision in a single round of the Prisoner's Dilemma game.

### Shared Columns

#### Identifiers & Experiment Structure

| Column | Type | Description |
|--------|------|-------------|
| `label` | str | Anonymised participant identifier (MD5 hash; index column) |
| `exp_no` | int | Experiment number (1 = pilot, 2 = main) |
| `condition` | str | Experimental condition: `RT` (real-time), `SM` (simultaneous), `SQ` (sequential; main only) |
| `session_no` | int | Session number within the experiment |
| `group_no` | int | Group (pair) number within a session |

#### Game Play Data

| Column | Type | Description |
|--------|------|-------------|
| `payoff` | float | Participant's payoff for the round |
| `is_coop` | bool | Whether the participant cooperated |
| `decision_time` | float | Time (seconds) the participant took to make their decision |
| `is_partner_coop` | bool | Whether the partner cooperated |
| `is_first_mover` | bool | Whether the participant was the first mover in the pair |
| `partner_decision_time` | float | Time (seconds) the partner took to make their decision |

#### SVO

| Column | Type | Description |
|--------|------|-------------|
| `svo` | float | Social Value Orientation angle (degrees); higher = more prosocial |

#### Demographics

| Column | Type | Description |
|--------|------|-------------|
| `time_taken` | float | Total time (seconds) the participant spent on the experiment |
| `cnt_approvals` | float | Number of Prolific approvals at time of participation |
| `age` | float | Participant's age |
| `sex` | str | Self-reported sex (`Male`, `Female`, `Prefer not to say`) |
| `ethnicity` | str | Self-reported ethnicity |
| `is_male` | bool | Whether the participant is male |

---

### Main Experiment Only

| Column | Type | Description |
|--------|------|-------------|
| `risk_aversion_change_point` | float | Risk aversion via switching point (0 = seeking, 0.5 = neutral, 1 = averse); `1 - risk_p / 32` |
| `ambiguity_aversion_change_point` | float | Ambiguity aversion via switching point (0 = seeking, 0.5 = neutral, 1 = averse); `1 - ambiguity_p / 100` |
| `trust_choice` | int | Trust game choice (1–5 scale) |
| `willingness_first_mover` | float | Willingness to be the first mover (1–7 Likert scale) |
| `willingness_second_mover` | float | Willingness to be the second mover (1–7 Likert scale) |
| `preferred_role` | str | Preferred role: `first_mover`, `second_mover`, or `indifferent` |

---

### Pilot Experiment Only

| Column | Type | Description |
|--------|------|-------------|
| `risk_aversion_modeling` | float | Risk aversion via structural estimation; `1 - alpha` where alpha is CRRA curvature. Higher = more averse |
| `ambiguity_aversion_modeling` | float | Ambiguity aversion via structural estimation; `beta` in `(p - beta*A/2) * v^alpha`. Higher = more averse |
| `decision_temperature` | float | Decision noise; `1/gamma` where gamma is inverse temperature. Higher = noisier choices |

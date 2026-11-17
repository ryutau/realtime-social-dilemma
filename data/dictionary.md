# Data dictionary

All files are UTF-8 comma-separated text with a header. There is one row per
participant in each table. `label` is the join key within an experiment. Empty
fields mean missing values; Boolean fields contain `True` or `False`.
The Main and Pilot samples should never be joined by row position.

## Files and sample sizes

These are the supplied files in `source/`, before applying completion exclusions.
Completion records are described at the end of this document.

| Source file | Rows | Content |
|---|---:|---|
| `main_pd_data.csv` | 562 | Main PD choices, payoffs, timing, and original pair membership |
| `main_demographics.csv` | 561 | Matched recruitment-platform demographic and participation records |
| `main_svo_params.csv` | 561 | SVO angles |
| `main_ra_params.csv` | 470 | Derived risk and ambiguity measures |
| `main_pd_willingness_data.csv` | 184 | Pre-game SQ role-willingness ratings |
| `main_trust_data.csv` | 562 | General-trust item |
| `pilot_pd_data.csv` | 156 | Pilot PD choices, payoffs, timing, and original pair membership |
| `pilot_demographics.csv` | 154 | Matched recruitment-platform demographic and participation records |
| `pilot_svo_params.csv` | 154 | SVO angles |
| `pilot_ra_params.csv` | 154 | Lottery-choice model estimates |

## Common key

| Column | Definition |
|---|---|
| `label` | Study-specific participant code: `M` plus four digits for Main, `P` plus four digits for Pilot. |

## PD tables: `main_pd_data.csv`, `pilot_pd_data.csv`

| Column | Definition |
|---|---|
| `exp_no` | Experiment indicator: `1` = Pilot, `2` = Main. |
| `condition` | `RT` = real-time, `SM` = simultaneous, `SQ` = sequential. Pilot has only RT and SM. |
| `session_no` | Session number within condition and experiment. |
| `group_no` | Original pair number within a session. Gaps are expected. |
| `group_id` | Pair key formed as `condition_session_no_group_no`. Use the experiment as an additional key when combining studies. |
| `payoff` | PD payoff in monetary units (MUs), before conversion to a GBP bonus. |
| `is_coop` | `True` for choosing Invest (cooperation), `False` for Keep (defection). |
| `decision_time` | Decision time in seconds on the experiment's game clock. Values can exceed 60 for decisions completed after the timed phase. These values should not be capped at 60. |
| `is_partner_coop` | The original partner's recorded action, coded in the same way as `is_coop`. |
| `is_first_mover` | In SQ, the assigned first-mover role; in RT and SM, the earlier decision within the original pair. SM therefore identifies a nominal first mover. |
| `partner_decision_time` | Original partner's decision time in seconds. |

After completion filtering, one Main SM singleton retains the action and time
of its original partner as recorded in the game. Pair analyses exclude this
incomplete pair. Pair membership must be checked from the number of eligible
rows sharing `group_id`, not inferred from the presence of partner fields.

## Demographic tables: `main_demographics.csv`, `pilot_demographics.csv`

| Column | Definition |
|---|---|
| `time_taken` | Study completion duration in seconds from the recruitment platform's participation record. |
| `cnt_approvals` | Number of previously approved studies recorded by the recruitment platform; the manuscript's participant-proficiency covariate. |
| `age` | Recorded age in years. Missing for two Pilot SM participants. |
| `sex` | Recruitment platform's recorded sex field: `Female`, `Male`, or `Prefer not to say` where present. This is distinct from the post-experimental gender question. |
| `ethnicity` | Recruitment platform's simplified ethnicity category. Not used in the reported regressions. Missing for nine Main and six Pilot participants. |
| `is_male` | `True` when the recorded `sex` equals `Male`; otherwise `False`, including the one Main `Prefer not to say` response. |

## SVO tables: `main_svo_params.csv`, `pilot_svo_params.csv`

| Column | Definition |
|---|---|
| `svo` | Social Value Orientation Slider Measure angle in degrees. Higher values indicate more prosocial allocation preferences. |

## Main risk and ambiguity: `main_ra_params.csv`

| Column | Definition |
|---|---|
| `risk_aversion_change_point` | `1 - sure_amount / 32`, where `sure_amount` is the switching value from the risk bisection task. Higher values indicate greater risk aversion. |
| `ambiguity_aversion_change_point` | `1 - known_purple_percentage / 100`, derived from the ambiguity bisection task. Higher values indicate greater ambiguity aversion. |

Missing measures appear as absent rows rather than empty fields. Among eligible
participants, all 91 missing records belong to the first Main SM session.
Analyses join these measures onto the PD sample by `label`.

## Pilot risk and ambiguity: `pilot_ra_params.csv`

| Column | Definition |
|---|---|
| `risk_aversion_modeling` | `1 - alpha`, from the utility model fit to lottery choices. Higher values indicate greater risk aversion. |
| `ambiguity_aversion_modeling` | `beta`, the ambiguity-attitude parameter of the utility model. Higher values indicate greater ambiguity aversion. |
| `decision_temperature` | Reciprocal of the fitted choice-consistency/inverse-temperature parameter. Larger values indicate noisier choices; `inf` occurs when the inverse-temperature estimate equals zero. Not used as a covariate in the reported regressions. |

The utility specification is `(p - beta * ambiguity_width / 2) *
prize ** alpha`, followed by a softmax choice rule. Task descriptions and the
original estimation procedure should be consulted before interpreting the
parameters as comparable to the Main experiment's bisection measures.

## Role willingness: `main_pd_willingness_data.csv`

These ratings were collected in SQ before role assignment and the PD game.

| Column | Definition |
|---|---|
| `willingness_first_mover` | Willingness to take the first-mover role, from `1` (not willing at all) to `7` (very willing). |
| `willingness_second_mover` | Willingness to take the second-mover role, on the same scale. |
| `preferred_role` | `first_mover` if the first rating is higher, `second_mover` if the second is higher, or `indifferent` if they are equal. |

## General trust: `main_trust_data.csv`

| Column | Definition |
|---|---|
| `trust_choice` | Response to the general-trust question: `1` = have to be very careful, `3` = not sure, `5` = most people can be trusted. The complete response scale is 1 through 5. |

## Completion records: `main_completion.csv`, `pilot_completion.csv`

There are 562 Main and 156 Pilot records, covering the PD samples before
completion exclusions. These records determine eligibility for the analyses;
they are not regression predictors.

| Column | Definition |
|---|---|
| `label` | Study-specific code matching the source tables. |
| `completed_postgame_tasks` | `True` if the participant reached `Thanks/MyPage` in the terminal-page record. |
| `last_app` | Last oTree application recorded for the participant. |
| `last_page` | Last oTree page recorded for the participant. |

The Main incomplete record ends at `svo/Decide`; the two Pilot incomplete
records end at `risk_ambiguity/Instruction` and `risk_ambiguity/Decide`.
`src.common.load_pd` applies the experiment-specific own/partner completion rule
described in [README.md](README.md).

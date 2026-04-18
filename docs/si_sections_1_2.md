# SI 1. Implementation details of main and pilot experiment

## Study purpose and preregistration

The pilot experiment was conducted prior to the main experiment reported in the main text. The primary aim was to confirm that people use decision timing strategically in real-time Prisoner's Dilemma games. To do so, we focused on the comparison between real-time interaction and simultaneous decisions. We expected to observe evidence of (1) strategic delay and (2) a higher likelihood of mutual cooperation in real-time PD. These hypotheses and other implementation details (e.g., sample size) were preregistered at [https://osf.io/d9mp7](https://osf.io/d9mp7).

The main experiment was preregistered at [https://osf.io/xdphe](https://osf.io/xdphe). The main aim was to test whether the advantage of real-time interaction over simultaneous decision-making would replicate under a more stringent free-riding incentive (lower MPCR) and with a larger sample size. We also preregistered a comparison between real-time and sequential conditions, which was not included in the pilot experiment.

## Shared implementation across pilot and main experiments

Both experiments employed the same general implementation described in the Methods section of the main text, including recruitment through Prolific Academic (US-resident, English-speaking participants), 60-second decision windows, a comprehension check, and post-experimental measurement of SVO, risk/ambiguity attitudes, and demographics.

Participants received a fixed participation fee plus a performance-based bonus. The bonus was calculated from one randomly selected outcome: either PD payoff, SVO slider measure, or performance on the risk/ambiguity tasks.

## Key implementation differences between pilot and main experiments

The pilot experiment differed from the main experiment in three major respects.

1. Incentive structure (MPCR): The marginal per-capita return (MPCR) was 0.75 in the pilot and 0.5 in the main experiment. In the pilot, when a player invested 100 MUs, the total of 150 MUs was split equally between the two players, yielding 75 MUs each.
2. Experimental conditions: The pilot included only two conditions, real-time (RT) and simultaneous (SM), whereas the main experiment included three conditions (RT, SM, SQ).
3. Sample size: The pilot aimed for a smaller sample size (approximately 80 participants per condition) due to budget constraints at that stage.

## Tasks other than the Prisoner's Dilemma game

Pilot Experiment:
- SVO slider measure (Murphy et al., 2011). Participants adjusted a slider to indicate their preferences over different point distributions between themselves and another person. This method yielded an SVO angle (in degrees) as a continuous measure of social preferences.
- Risk and ambiguity aversion estimated from 12 binary lottery choices per participant. The pilot used 24 pre-specified lottery pairs (2 sets × 12 items), combining objective probabilities (0.25, 0.50, or 0.75), ambiguity levels (0, 0.24, 0.50, or 1.00), and prize magnitudes (10, 16, 40, or 100 MUs). Choices were modeled using a utility-function approach (Vives & Feldmanhall, 2018; Tymula et al., 2012). Each lottery was valued as `p * m^alpha` for risky lotteries and `p * m^alpha * (1 - beta * a)` for ambiguous lotteries, where p is the winning probability, m is the prize magnitude, a is the ambiguous width for participants, and alpha and beta are free parameters for risk and ambiguity attitudes, respectively. Value differences between the two options were fed into a softmax function to model choice probabilities, with an additional free parameter for choice consistency. We computed MLE estimates of three parameters (alpha, beta, and temperature) for each participant across the 12 items.s

Main Experiment:
- SVO slider measure (Murphy et al., 2011)
- Risk and ambiguity aversion measured via separate bisection tasks (Dimmock, Kouwenberg, & Wakker, 2016). In the risk task, participants chose between a safe box (certain payoff) and a risky box (50% chance of £32, otherwise £0). The safe payoff was adjusted via binary search across 32 steps to identify the switching point; risk aversion was computed as `1 - risk_p / 32`. In the ambiguity task, participants chose between an ambiguous box (unknown composition) and a box with a known color distribution. The known proportion was narrowed via binary search to identify the indifference point; ambiguity aversion was computed as `1 - ambiguity_p / 100`.
- General trust (Yamagishi & Yamagishi, 1994). Participants indicated their agreement with the statement "Generally speaking, would you say that most people can be trusted or that you have to be very careful in dealing with people?" on a 1–5 Likert scale.

## Pilot sample and compensation summary
Pilot Experiment: After applying exclusion criteria (see Methods), the final pilot sample comprised N = 156 participants (78 pairs in total: 38 in SM and 40 in RT). Participants received a GBP 4 fixed participation fee plus a performance-based bonus of up to GBP 4, with the bonus computed from PD payoff as `4 * payoff / 175`. The mean bonus was GBP 3.17 and the median completion time was 16.7 minutes.

Main Experiment: After applying exclusion criteria (see Methods), the final main sample comprised N = 562 participants (281 pairs in total: 95 in RT, 94 in SM, and 92 in SQ). Participants received a GBP 1.5 fixed participation fee plus a performance-based bonus of up to GBP 3, with the bonus computed from PD payoff as `3 * payoff / 200`. The mean bonus was GBP 1.96 and the median completion time was 12.5 minutes.

# SI 2. Results from pilot experiment

## Overall cooperation and mutual cooperation rates

Figure S1 shows individual cooperation rates by condition and decision-making position, and Figure S2 shows pair-level mutual cooperation rates.

In the SM condition, 71.1% (Bootstrapped 95% CI [60.5, 81.6]) of participants chose to cooperate. Despite this relatively high individual cooperation rate, only 47.4% (95% CI [31.6, 63.2]) of pairs achieved mutual cooperation. This gap illustrates the coordination difficulty inherent in simultaneous decision-making, where mutual cooperation requires two independent cooperative choices to coincide.

In the RT condition, all pairs had at least one player who committed before the 60-second deadline, producing a voluntary first mover in every pair. We instructed participants that if both players waited the full 60 seconds in RT, time would stop and both would end up choosing simultaneously. Among voluntary first movers, 87.5% (95% CI [75.0, 95.0]) chose to cooperate. Voluntary second movers reciprocated first-mover cooperation at a rate of 91.4% (95% CI [82.9, 100.0]) and reciprocated first-mover defection at 0%. As a result, 80.0% (95% CI [65.0, 90.0]) of RT pairs achieved mutual cooperation, approximately 30 percentage points higher than in SM.

## Comparison of first-mover behavior: RT vs. SM

Does the high first-mover cooperation rate in RT (87.5%) reflect participants' strategic choice of decision timing, or is it merely a reflection of a pre-existing (non-strategic) tendency for cooperators to decide faster? To assess this, we compared observed decisions by voluntary first movers in RT with decisions by nominal first movers in SM, which we define as the faster-deciding player within each SM pair. Note that nominal first movers' decision timing should have no strategic consequence because decisions were not revealed during the 60-second window in SM. Nominal first movers in SM cooperated at 73.7% (95% CI [57.9, 86.8]), which was slightly higher than the rate for nominal second movers in SM: 68.4% (95% CI [55.3, 81.6]). Thus, a correlation between cooperation and faster decisions existed even in the absence of strategic timing. However, crucially, the nominal first-mover cooperation rate (73.7%) remained below the voluntary first-mover cooperation rate in RT (87.5%).

## Decision-time distributions

Decision-time distributions revealed qualitative evidence of strategic timing in RT. Figure S3 displays the distributions for first movers (nominal in SM; voluntary in RT), separated by decision (cooperation in orange, defection in blue).

In SM, decision-time distributions for both cooperation and defection were unimodal with a single early peak at decision onset. The two distributions substantially overlapped (Mann-Whitney U = 580, p = .877). In RT, cooperative decisions showed a similar unimodal early-peak pattern, but defection decisions exhibited a distinctive bimodal distribution with an additional peak near the 60-second deadline. This bimodality suggests that some RT first movers strategically delayed their defection to observe their partner's action.

## Pilot summary

The pilot experiment established core patterns that replicated in the larger main experiment under more stringent free-riding incentives and with a larger sample. Specifically: (1) real-time interaction substantially increased mutual cooperation relative to simultaneous decision-making; (2) voluntary first movers in RT exhibited elevated cooperation rates that exceeded those predicted by non-strategic correlations between cooperation and decision speed; and (3) defection decisions in RT showed strategic delay, with a modal decision time near the deadline.

# SI 3. Additional results from main experiment

## Summary of results reported in main text

1. Real-time interaction (RT) yielded the highest rate of mutual cooperation (i.e., both players choosing Invest). In RT, 64.2% (95%CI [53.7, 73.7]) of pairs achieved pair-level mutual cooperation, compared to 51.1% (95%CI [40.2, 60.9]) in SQ and 30.9% (95%CI [21.3, 39.4]) in SM.
2. Compared with SQ (rather than SM), RT's main advantage stems from higher second-mover reciprocity rates:
   1. First-mover cooperation rates in SQ (68.5%) and RT (68.4%) were indistinguishable.
   2. Second movers whose partner cooperated beforehand reciprocated with cooperation more readily in RT (93.8%) than in SQ (74.6%).

## Additional comparison between RT and SM

Although the main text emphasizes the comparison between RT and SQ, results from the main experiment also replicated the pilot findings.

First, does the cooperation rate of voluntary first movers in RT exceed that of nominal first movers in SM? Again, we divided decisions in SM into nominal first and second movers and compared nominal first movers with (strategic) voluntary first movers in RT.

Notably, nominal first- and second-mover cooperation rates were indistinguishable in the SM condition of the main experiment: 57.4% (95% CI: [47.8, 67.0]) and 57.4% (95% CI: [47.8, 68.1]). This pattern aligns with literature proposing a single-process account linking non-strategic (value-based) cooperative decisions to decision speed. Under this account, when cooperation becomes more difficult (e.g., through lower MPCR), even existing positive correlations between faster decisions and cooperation can diminish or reverse. Compared with the pilot (where we observed a weak positive correlation), the lower MPCR in the main experiment eliminated this relationship. Critically, the first-mover cooperation rate in RT (68.4%) exceeded that of nominal first movers in SM (57.4%).

Second, looking at decision-time distributions, cooperators and defectors in SM again had overlapping distributions (Mann-Whitney U = 3856, two-sided p = 0.208), with a single early peak. As confirmed in the main text, only voluntary first movers in RT showed a decision-time distribution with a modal late-defection peak.

## Regression analyses

We exploratorily analyzed how individual characteristics predicted cooperation across decision contexts. The predictor set was common across analyses: SVO, risk attitude, ambiguity attitude (with different task implementations in pilot vs. main), gender, age, and Prolific proficiency (number of completed studies). Continuous predictors were standardized (z-scored) within each experiment, and gender was dummy-coded (reference = male).

For SM and SQ, we used standard logistic regression. For voluntary first-mover cooperation in RT, we used a two-step estimator because first-mover status is endogenously selected by decision timing.

In RT, the observed first mover is the player who commits earlier; because cooperators and defectors have different timing distributions, a naive logistic regression on observed first movers can be selection-biased. Specific procedures are as follows:
- Step 1 estimated action-specific timing survival functions nonparametrically from first-mover timing data: $S_C(t)$ for cooperators and $S_D(t)$ for defectors. These functions quantify the probability that a potential decision time occurs after $t$, separately by action.
- Step 2 then estimated the cooperation model by maximum likelihood using all RT participants. For observed first movers, the likelihood used their observed action directly; for observed second movers, the likelihood integrated over unobserved potential actions weighted by $S_C(t)$ and $S_D(t)$, reflecting that they were censored by their partner committing earlier. This yields coefficient estimates corrected for endogenous first-mover selection.

### Pilot experiment

Because the pilot included only SM and RT (no SQ), we estimated the subset of analyses applicable to those conditions:

- Analysis 1 (SM; all participants): logistic regression, $N = 74$. See SI Table S1 for estimated coefficients.
- Analysis 2 (RT; voluntary first-mover cooperation): two-step MLE with censoring correction, $N = 78$. See SI Table S2 for estimated coefficients.
- Analysis 3 (RT; second-mover reciprocation after observing first-mover cooperation): logistic regression, $N = 34$. See SI Table S3 for estimated coefficients.

### Main experiment

- Analysis 1 (SM; all participants): logistic regression, $N = 96$. See SI Table S4 for estimated coefficients.
- Analysis 2 (SQ; assigned first movers): logistic regression, $N = 92$. See SI Table S5 for estimated coefficients.
- Analysis 3 (SQ; assigned second movers after observing first-mover cooperation): logistic regression, $N = 63$. See SI Table S6 for estimated coefficients.
- Analysis 4 (RT; voluntary first-mover cooperation): two-step MLE with censoring correction, $N = 190$. See SI Table S7 for estimated coefficients.
- Analysis 5 (RT; second-mover reciprocation after observing first-mover cooperation): logistic regression, $N = 65$. See SI Table S8 for estimated coefficients.

### Summary
In brief, we observed that higher SVO (more prosocial orientation) and lower risk aversion were associated with higher cooperation rates across conditions. In RT, the two-step estimator revealed that these traits predicted voluntary first-mover cooperation.

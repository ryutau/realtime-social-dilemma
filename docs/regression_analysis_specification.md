# Regression Analysis Specification for SI

## Overview

This document specifies regression analyses examining how individual-level variables predict decisions across three timing conditions (SM, SQ, RT) in one-shot Prisoner's Dilemma games. The analyses serve two purposes: (1) characterizing the individual-level predictors of cooperation in each decision-making position, and (2) providing supplementary evidence for the main text findings by examining whether the composition of players differs across positions.

---

## 1. Individual Variables (Predictors)

All predictors are included in every regression unless otherwise noted. Continuous variables are standardized (z-scored) prior to analysis to facilitate comparison of effect sizes.

| Variable | Type | Standardization |
|---|---|---|
| SVO (Social Value Orientation) | Continuous (angle) | z-scored |
| Risk attitude | Continuous | z-scored |
| Ambiguity attitude | Continuous | z-scored |
| Gender | Categorical (dummy-coded; reference = male) | — |
| Age | Continuous | z-scored |
| Prolific proficiency | Continuous or categorical | z-scored if continuous |

The predictor vector for participant $i$ is denoted $X_i = (1, \text{SVO}_i, \text{Risk}_i, \text{Ambiguity}_i, \text{Gender}_i, \text{Age}_i, \text{Proficiency}_i)$.

---

## 2. Analysis Specifications by Decision Context

### Analysis 1: Cooperation in SM (Simultaneous condition)

**Sample**: All participants in the SM condition ($N \approx 188$).

**Model**: Standard logistic regression.

$$P(\text{Cooperate}_i = 1 \mid X_i) = \text{logit}^{-1}(X_i \beta)$$

**Rationale**: All SM participants face an identical decision situation (independent, no observation of partner). There is no selection, censoring, or conditioning issue. Standard logistic regression is appropriate.

---

### Analysis 2: Cooperation as assigned first movers in SQ

**Sample**: All assigned first movers in the SQ condition ($N \approx 92$).

**Model**: Standard logistic regression.

$$P(\text{Cooperate}_i = 1 \mid X_i) = \text{logit}^{-1}(X_i \beta)$$

**Rationale**: First-mover assignment is random (exogenous), so this sample is an unbiased draw from the participant population. No selection correction is needed.

---

### Analysis 3: Cooperation as assigned second movers in SQ

**Sample**: Assigned second movers in SQ who observed first-mover **cooperation** ($N \approx 63$, depending on first-mover cooperation rate).

**Model**: Standard logistic regression, restricted to second movers who observed cooperation.

$$P(\text{Reciprocate}_i = 1 \mid X_i, \text{1st mover cooperated}) = \text{logit}^{-1}(X_i \beta)$$

**Rationale**: The key outcome of interest is reciprocation of cooperation. Conditioning on first-mover cooperation is necessary because the decision context (and its strategic meaning) differs fundamentally between observing cooperation vs. defection. Including first-mover defectors with an interaction term would mix two qualitatively different decision problems.

Since first-mover assignment is random, first-mover cooperation is exogenous with respect to the second mover's characteristics, so conditioning on it does not introduce collider bias.

**Supplementary**: As a robustness check, a model including all second movers with first-mover action as a predictor (and its interactions with individual variables) can be reported:

$$P(\text{Cooperate}_i = 1 \mid X_i, a_{1\text{st}}) = \text{logit}^{-1}(X_i \beta_0 + a_{1\text{st}} \cdot X_i \beta_1)$$

---

### Analysis 4: Voluntary first-mover cooperation in RT

**Sample**: All participants in the RT condition ($N \approx 190$, i.e., both observed first movers and second movers).

**Model**: Two-step maximum likelihood estimator that accounts for the endogenous selection into the first-mover role.

#### Data generating process

Each participant $i$ in the RT condition has a latent potential decision $(a_i, t_i)$, where $a_i \in \{C, D\}$ is the action they would take as first mover and $t_i \in [0, 60]$ is the time at which they would commit. The model decomposes this as:

- **Action model**: $P(a_i = C \mid X_i) = \text{logit}^{-1}(X_i \beta)$
- **Timing model**: $t_i \mid a_i \sim f_{a_i}(t)$, independent of $X_i$ given $a_i$

The timing distributions $f_C(t)$ and $f_D(t)$ capture the empirical pattern that cooperators tend to commit early while defectors tend to delay.

#### Observation process

In each pair $(i, j)$: whichever participant has the smaller potential decision time becomes the observed first mover. The other participant is **censored** — their potential action $a_j$ and timing $t_j$ are unobserved, with only the constraint $t_j > t_i$ known.

#### Likelihood

For a pair where participant $i$ is the observed first mover (at time $t_i$ with action $a_i$) and participant $j$ is the second mover:

$$L_{\text{pair}} = \underbrace{P(a_i \mid X_i; \beta) \cdot f_{a_i}(t_i)}_{\text{first mover (complete data)}} \times \underbrace{\left[ P(C \mid X_j; \beta) \cdot S_C(t_i) + P(D \mid X_j; \beta) \cdot S_D(t_i) \right]}_{\text{second mover (censored)}}$$

where $S_C(t) = P(t_i > t \mid a_i = C)$ and $S_D(t) = P(t_i > t \mid a_i = D)$ are action-specific survival functions.

For estimating $\beta$, the $f_{a_i}(t_i)$ term in the first-mover contribution is a constant (does not depend on $\beta$), so the effective log-likelihood is:

$$\ell(\beta) = \sum_{\text{pairs}} \left[ \log P(a_i \mid X_i; \beta) + \log \left\{ P(C \mid X_j; \beta) \cdot S_C(t_i) + P(D \mid X_j; \beta) \cdot S_D(t_i) \right\} \right]$$

#### Estimation procedure

**Step 1** (Nonparametric): Estimate $S_C(t)$ and $S_D(t)$ from observed first-mover data using the minimum-of-two inversion procedure described in the main text (see Methods). This step uses the same approach as Figure 3C and does not depend on individual-level covariates.

**Step 2** (MLE): Fixing $S_C(t)$ and $S_D(t)$ from Step 1, maximize $\ell(\beta)$ with respect to $\beta$.

Standard errors are computed from the observed information matrix (numerical Hessian of the negative log-likelihood). Bootstrap standard errors accounting for Step 1 uncertainty can be reported as a robustness check.

#### Key assumption

The conditional independence assumption $f(t \mid a, X) = f(t \mid a)$ states that, given the action choice, timing does not further depend on individual characteristics. This means $X$ influences timing only indirectly through the action choice: $X \to a \to t$. This assumption is consistent with the nonparametric timing estimation in the main text (Figure 3C), which recovers $f_C(t)$ and $f_D(t)$ without conditioning on covariates.

#### Why this model is necessary

A naive logistic regression on observed first movers only would suffer from selection bias. Cooperators commit earlier than defectors, so they are over-represented among observed first movers. Simulation studies confirm that naive logistic regression produces severely biased intercept estimates (true $\beta_0 = 0.3$ is estimated as $\approx 1.5$), while the two-step estimator recovers all parameters with minimal bias. Slope parameters are less affected by the selection, but the two-step approach is preferred for consistent estimation of all parameters.

#### Parameter recovery validation

Simulation studies with the experimental sample size ($n_{\text{pairs}} = 95$) confirm:

- All parameters (intercept and slopes) are recovered with minimal bias across a range of true values.
- Estimated standard errors closely match simulation-based standard deviations.
- No substantial parameter interference (varying one true parameter does not bias estimation of others).

---

### Analysis 5: Reciprocating cooperation as second movers in RT

**Sample**: Voluntary second movers in RT who observed first-mover cooperation ($N \approx 60$–$65$, depending on how many pairs had first-mover cooperation).

**Model**: Standard logistic regression, restricted to second movers who observed cooperation.

$$P(\text{Reciprocate}_i = 1 \mid X_i, \text{1st mover cooperated}) = \text{logit}^{-1}(X_i \beta)$$

**Rationale**: The conditioning on first-mover cooperation is necessary for the same reason as in Analysis 3. 

A potential concern is that voluntary second movers in RT are not a random sample — they are participants whose partner happened to commit first. However, under the assumption that paired participants' potential decision times are independent (which holds by the experimental design, as pairs are formed by random matching of strangers), being a second mover is determined by the partner's timing, not by the focal participant's own characteristics. Therefore, the second-mover subsample is not systematically selected on individual variables, and standard logistic regression is appropriate.

**Note on sample size**: With a high reciprocation rate in RT ($\approx 94\%$), the effective variance in the outcome is very low. This analysis may have limited power to detect individual-level predictors, and near-separation may occur. If separation is detected, Firth's penalized logistic regression should be used.

---

## 3. Cross-Condition Comparisons

In addition to within-condition analyses, pooled regressions can directly test whether the same individual variables have different effects across conditions.

### Comparison A: First movers in SQ vs. RT

Pool assigned first movers (SQ) and all RT participants (using the two-step estimator framework). Test whether $\beta$ differs across conditions by comparing the estimated coefficients from Analyses 2 and 4.

Formal comparison: likelihood ratio test or Wald test comparing $\beta^{\text{SQ}}$ vs. $\beta^{\text{RT}}$.

### Comparison B: Second-mover reciprocation in SQ vs. RT

Pool second movers who observed cooperation from SQ and RT. Add a condition dummy and its interactions:

$$P(\text{Reciprocate}_i = 1 \mid X_i, \text{Condition}_i) = \text{logit}^{-1}(X_i \beta_0 + \text{RT}_i \cdot X_i \beta_1)$$

The interaction terms $\beta_1$ test whether individual variables predict reciprocation differently depending on whether the move order was exogenous (SQ) or endogenous (RT). This directly relates to the main text argument that voluntary first-mover cooperation is a stronger signal, potentially reducing the role of individual preferences in the reciprocation decision.

---

## 4. Multiple Comparisons

With 5 primary analyses × 6 predictors = 30 tests, correction for multiple comparisons is warranted. We apply Benjamini-Hochberg FDR correction (at $q = 0.05$) within each analysis, and report both uncorrected and corrected p-values.

---

## 5. Summary Table

| # | Decision context | Sample | Method | Key question |
|---|---|---|---|---|
| 1 | SM: independent cooperation | All SM participants | Logistic regression | What predicts cooperation without any sequential information? |
| 2 | SQ: assigned 1st-mover cooperation | Assigned 1st movers in SQ | Logistic regression | What predicts leading with cooperation when assigned? |
| 3 | SQ: assigned 2nd-mover reciprocation | Assigned 2nd movers observing C | Logistic regression | What predicts reciprocating assigned first-mover cooperation? |
| 4 | RT: voluntary 1st-mover cooperation | All RT participants | Two-step MLE | What predicts potential willingness to lead voluntarily? |
| 5 | RT: 2nd-mover reciprocation | Voluntary 2nd movers observing C | Logistic regression | What predicts reciprocating voluntary first-mover cooperation? |
| A | 1st movers: SQ vs. RT | Analyses 2 + 4 | Coefficient comparison | Do predictors of leading differ when order is exogenous vs. endogenous? |
| B | 2nd movers: SQ vs. RT | Analyses 3 + 5 | Pooled logistic with interaction | Does the signal strength of first-mover cooperation change the role of individual preferences? |

---

## 6. Implementation Notes

- Python code for the two-step estimator (Analysis 4) is provided as `rt_regression.py`.
- All other analyses use standard logistic regression (e.g., `statsmodels.Logit` or `sklearn`).
- For Analysis 5, check for separation; use `firthlogist` or `statsmodels` with Firth penalty if needed.
- Standardize all continuous predictors before fitting to enable direct comparison of odds ratios across predictors and analyses.

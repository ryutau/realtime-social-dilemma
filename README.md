# Real-time Social Dilemmas

This repository contains code for the paper "Real-time interaction mitigates coordination problems in social dilemmas by amplifying reciprocity" by Ryutaro Mori, Nobuyuki Hanaki, and Tatsuya Kameda.

## Setup

```bash
uv sync            # installs deps from pyproject.toml / uv.lock (Python ≥3.11)
source .venv/bin/activate
```

All scripts are run from the repository root as modules (`python -m src.<script>`)
so the `src.data` package resolves correctly.

## Experiment data

CSVs from the main and pilot experiments are stored under `data/processed/`.
Instruction slides used in the experiments are available as PDFs under `material/`.

## Analysis

### Reproduce main-text figures
```bash
python -m src.fig_2A_mutual_cooperation        # Fig 2A
python -m src.fig_2B_cooperation_rate          # Fig 2B
python -m src.fig_3_decision_time              # Fig 3 (SM, SQ, RT observed + RT recovered)
```

### Reproduce main-text analyses

```bash
python -m src.analysis_rt_potential_coop      # Latent RT first-mover coop rate in the main experiment
python -m src.analysis_svo_second_movers      # SVO: RT vs SQ second movers in the main experiment
```

### Reproduce SI figures and tables

```bash
# Figures
python -m src.fig_s1_pilot                    # Fig S1
python -m src.fig_s2_decision_time_pilot      # Fig S2 (pilot DT)
python -m src.fig_s3_decision_time_main       # Fig S3 (main DT combined)

# Other analyses
python -m src.analysis_rt_potential_coop_pilot # Latent RT first-mover coop rate in the pilot experiment
python -m src.analysis_sm_vs_rt_first_mover    # SM nominal vs RT voluntary comparison in both experiments

# Regressions
python -m src.analysis_regression # see details below
```

```
`analysis_regression.py` produces:

| Table | Analysis | Output CSV |
|---|---|---|
| S1 | Pilot SM logit | `regression_pilot_sm_all.csv` |
| S2 | Pilot RT two-step MLE | `regression_pilot_rt_twostep.csv` |
| S3 | Pilot RT 2nd-mover after C | `regression_pilot_rt_second_after_c.csv` |
| S4 | Main SM logit | `regression_main_sm_all.csv` |
| S5 | Main SQ 1st-mover logit | `regression_main_sq_first.csv` |
| S6 | Main SQ 2nd-mover after C | `regression_main_sq_second_after_c.csv` |
| S7 | Main RT two-step MLE | `regression_main_rt_twostep.csv` |
| S8 | Main RT 2nd-mover after C | `regression_main_rt_second_after_c.csv` |

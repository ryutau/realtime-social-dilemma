# Real-time Social Dilemmas

Data and code for **Real-time interaction and the signaling value of voluntary early cooperation**, by Ryutaro Mori, Nobuyuki Hanaki, and Tatsuya Kameda.

## Run all analyses and figures

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run these commands from the repository folder:

```sh
uv sync --locked
uv run --frozen python run.py
```

Results are saved in `output/`. Bootstrap calculations use 50,000 resamples with fixed random seeds. Figure 4B takes several minutes to run.

## Run a single figure

Main figures have one script per panel. Appendix and Online Appendix figures have one script per figure. After setup, you can run any figure script directly. For example:

```sh
# Figure 3B
uv run --frozen python -m src.main.figure3_b

# Figure OA1
uv run --frozen python -m src.online_appendix.figure_oa1
```

These commands save `figure3B.pdf` in `output/main/figures/` and `figure_oa1.pdf` in `output/online_appendix/figures/`.

## Files

- `data/source/`: input data and completion records.
- `data/intermediate/`: intermediate analysis results.
- `src/main/`, `src/appendix/`, `src/online_appendix/`: scripts for each section of the paper.
- `output/`: figures, tables, and analysis results.
- `material/`: [screenshots and instruction slides](material/experimental_materials_main_pilot.pdf) for the main and pilot experiments.

See the [data notes](data/README.md) for exclusions and sample sizes, and the [data dictionary](data/dictionary.md) for column definitions.

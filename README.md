# Real-time Social Dilemmas

Data and analysis code for **Real-time interaction and the signaling value of voluntary early cooperation**, by Ryutaro Mori, Nobuyuki Hanaki, and Tatsuya Kameda.

## Run

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync --locked
uv run --frozen python run.py
```

This prepares shared analysis estimates, then generates figures and analyses. Append `prepare`, `figures`, or `analyses` to run just that stage. Bootstrap analyses use 50,000 resamples and fixed seeds; figures are saved as PDFs. Figure 4B's bootstrap fits take several minutes.

## Individual outputs

Main figures are saved by panel; Appendix and Online Appendix figures are saved as complete figures. Each figure script contains its sample selection, calculations, and plotting, and runs directly without preparation or command-line options:

```sh
# Figure 3B
uv run --frozen python -m src.main.figure3_b

# Figure OA1
uv run --frozen python -m src.online_appendix.figure_oa1
```

Figures 1 and 5 are schematic illustrations and are not generated here.

## Files

- `data/source/`: derived participant data and completion records.
- `data/intermediate/`: generated estimates and bootstrap results.
- `src/main/`, `src/appendix/`, `src/online_appendix/`: preparation and output scripts by manuscript section.
- `src/common.py`: shared data loading and statistical functions; completion exclusions are applied when loading data.
- `src/plotting.py`: shared figure styling.
- `output/`: generated figures, tables, and numerical summaries.
- `material/`: [screenshots and instruction slides](material/experimental_materials_main_pilot.pdf) for the main and pilot experiments.

See [data notes](data/README.md) for sample definitions and [the data dictionary](data/dictionary.md) for variable descriptions.

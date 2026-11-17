"""SQ participants' willingness to act as first and second movers."""

import json

import pandas as pd

from src.common import SOURCE, PROJECT, load_main


def main():
    data = load_main()
    ratings = pd.read_csv(SOURCE / "main_pd_willingness_data.csv")
    eligible = data.loc[data.condition == "SQ", ["label"]]
    matched = eligible.merge(ratings, on="label", how="left", validate="one_to_one")
    cols = ["willingness_first_mover", "willingness_second_mover"]
    if not matched[cols].isin(range(1, 8)).to_numpy().all():
        raise ValueError("SQ role ratings must be available and in the range 1–7")
    first, second = (matched[column] for column in cols)
    estimates = {
        "n": len(matched),
        "first": {"mean": float(first.mean()), "sd": float(first.std(ddof=1))},
        "second": {"mean": float(second.mean()), "sd": float(second.std(ddof=1))},
        "first_higher": int((first > second).sum()),
        "second_higher": int((second > first).sum()),
        "equal": int((first == second).sum()),
    }
    output = PROJECT / "output/online_appendix"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sq_willingness.json"
    path.write_text(json.dumps({"main_SQ_willingness": estimates}, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

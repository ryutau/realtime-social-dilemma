"""Reproduce the paper, or run just preparation, figures, or analyses."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = {
    "prepare": [
        "src.appendix.prepare_svo",
        "src.online_appendix.prepare_rt_regression",
    ],
    "figures": [
        "src.main.figure2_a", "src.main.figure2_b", "src.main.figure2_c", "src.main.figure2_d",
        "src.main.figure3_a", "src.main.figure3_b", "src.main.figure3_c",
        "src.main.figure4_a", "src.main.figure4_b",
        "src.appendix.figure_a1",
        "src.online_appendix.figure_oa1", "src.online_appendix.figure_oa2",
        "src.online_appendix.figure_oa3",
    ],
    "analyses": [
        "src.main.analysis_cooperation", "src.main.analysis_timing", "src.main.analysis_preregistered",
        "src.appendix.table_a1",
        "src.online_appendix.analysis_main_rt_sm", "src.online_appendix.analysis_pilot_rt_sm",
        "src.online_appendix.analysis_willingness",
        "src.online_appendix.table_oa1", "src.online_appendix.table_oa2", "src.online_appendix.table_oa3",
        "src.online_appendix.table_oa4", "src.online_appendix.table_oa5", "src.online_appendix.table_oa6",
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all", choices=("all", *STEPS))
    stage = parser.parse_args().stage
    stages = STEPS if stage == "all" else (stage,)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for stage in stages:
        for module in STEPS[stage]:
            print(f"python -m {module}", flush=True)
            subprocess.run([sys.executable, "-B", "-m", module], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()

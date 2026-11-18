"""Run the whole pipeline end to end (about 1-2 hours on a 12-thread CPU).

    python run_all.py            # all steps
    python run_all.py --from 3   # resume from step 3

Steps: 1 EDA + leakage audit | 2 hold-out experiments | 3 nested CV | 4 statistical tests |
       5 diagnostics + figures | 6 LaTeX tables | 7 compile the French report (needs pdflatex)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("EDA and leakage audit", ["scripts/01_eda_and_leakage_audit.py"]),
    ("Hold-out experiments (default / grid / random / optuna)", ["scripts/02_holdout_experiments.py"]),
    ("Nested cross-validation", ["scripts/03_nested_cv.py"]),
    ("Statistical tests", ["scripts/04_statistical_tests.py"]),
    ("Diagnostics and figures", ["scripts/05_diagnostics_and_figures.py"]),
    ("LaTeX tables", ["scripts/06_make_report_tables.py"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run (1-7)")
    args = ap.parse_args()
    for i, (title, cmd) in enumerate(STEPS, start=1):
        if i < args.start:
            continue
        print(f"\n##### Step {i}: {title}", flush=True)
        subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True)
    if args.start <= 7 and shutil.which("pdflatex"):
        print("\n##### Step 7: compile report", flush=True)
        for _ in range(2):  # twice for cross-references
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "rapport.tex"], cwd=ROOT / "report", check=True,
                           stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()

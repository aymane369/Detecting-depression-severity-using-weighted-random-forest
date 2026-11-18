"""Central configuration: paths, seeds, experimental design constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Depression Oxidative Stress Mitochondrial.csv"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
TABLE_DIR = REPORT_DIR / "tables"

# ---------------------------------------------------------------- reproducibility
SEED = 42

# ---------------------------------------------------------------- protocol (follows the source article)
TEST_SIZE = 0.30          # 70 % train / 30 % hold-out test (Bader et al., 2024)
CV_FOLDS = 5              # 5-fold stratified CV inside the training set for the searches
N_ITER_RANDOM = 30        # candidates sampled by RandomizedSearchCV
N_TRIALS_OPTUNA = 30      # same budget for the Bayesian (TPE) search

# nested CV used for the statistical tests (outer loop)
NESTED_OUTER_FOLDS = 5
NESTED_OUTER_REPEATS = 2  # 5 x 2 = 10 paired outer scores per model
NESTED_INNER_ITER = 15    # RandomizedSearchCV candidates in each inner loop

# ---------------------------------------------------------------- variables
TARGET_COL = "PHQ-9-Score"
ID_COLS = ["Patient"]          # per-visit pseudonym, not a patient identifier
EXCLUDED_COLS = ["ScreenYear"]  # visit year: administrative, not a clinical predictor

BIOMARKERS = ["Humanin", "MOTSc", "P66"]
CLINICAL = [
    "ACE-Update",
    "OtherMeds-Depression",
    "Gender",
    "BirthAge",
    "DM-Status",
    "DM-MedUse",
    "CVD-Revised",
    "HT-Status",
    "HT-MedUse",
]
FEATURE_SETS = {
    "MDP": BIOMARKERS,            # biomarkers only (analogue of "oxidative stress only")
    "ALL": BIOMARKERS + CLINICAL,  # biomarkers + socio-demographic + health-related
}

# PHQ-9 severity bins used by the article: 0-4 / 5-13 / 14-20 / >=21
PHQ_BINS = [-1, 4, 13, 20, 99]
CLASS_NAMES_MULTI = ["No depression", "Mild", "Moderate", "Severe"]
CLASS_NAMES_BINARY = ["Not depressed", "Depressed"]

# experimental settings: (regime, task, feature set)
# 'full'  = all 2,621 rows (article-style protocol)
# 'clean' = rows whose three biomarker values are not shared with any other row (n=228)
SETTINGS = [
    ("full", "binary", "MDP"),
    ("full", "binary", "ALL"),
    ("full", "multi", "MDP"),
    ("full", "multi", "ALL"),
    ("clean", "binary", "MDP"),
    ("clean", "binary", "ALL"),
]


def setting_key(regime: str, task: str, fset: str) -> str:
    return f"{regime}-{task}-{fset}"

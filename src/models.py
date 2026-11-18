"""Model zoo, hyper-parameter search spaces and search factories.

Ten models are compared:
  * the six classifiers of the source article: LR, RF, KNN, SVM, NB, ANN;
  * the class-weighted variants of the article: WLR, WRF (class_weight='balanced',
    i.e. W_j = n / (K * n_j), Eq. 9 of the article) and RF-SMOTE;
  * one extra learner added by this project: WHGB (weighted histogram gradient boosting).

Every model is a pipeline ``scale -> [smote] -> clf``.  The scaler (and SMOTE) are fitted
inside each CV training fold only, so no information from the validation fold leaks.
"""
from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C

MODEL_NAMES = ["LR", "RF", "KNN", "SVM", "NB", "ANN", "WLR", "WRF", "RF-SMOTE", "WHGB"]
BASE_MODELS = ["LR", "RF", "KNN", "SVM", "NB", "ANN"]       # unweighted, article Tables 3-9
WEIGHTED_MODELS = ["WLR", "WRF", "RF-SMOTE", "WHGB"]         # article Table 13 (+ WHGB)


def build(name: str, seed: int = C.SEED) -> Pipeline:
    """Return an *untuned* pipeline carrying library-default hyper-parameters
    (except the ones fixed by the article: 350 epochs for the ANN, class weights)."""
    scaler = ("scale", StandardScaler())
    if name == "LR":
        clf = LogisticRegression(solver="saga", max_iter=5000, random_state=seed)
    elif name == "WLR":
        clf = LogisticRegression(solver="saga", max_iter=5000, class_weight="balanced", random_state=seed)
    elif name == "RF":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif name == "WRF":
        clf = RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1)
    elif name == "RF-SMOTE":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
        return ImbPipeline([scaler, ("smote", SMOTE(random_state=seed)), ("clf", clf)])
    elif name == "KNN":
        clf = KNeighborsClassifier()
    elif name == "SVM":
        # max_iter is a safety cap (default -1 = unlimited); typical RBF fits need << 10^6 iterations
        clf = SVC(probability=True, max_iter=1_000_000, random_state=seed)
    elif name == "NB":
        clf = GaussianNB()
    elif name == "ANN":
        clf = MLPClassifier(max_iter=350, random_state=seed)  # ReLU, 350 epochs (article)
    elif name == "WHGB":
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    else:
        raise KeyError(name)
    return Pipeline([scaler, ("clf", clf)])


# ------------------------------------------------------------------ GridSearchCV spaces
_RF_GRID = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [3, 5, 10, None],
    "clf__min_samples_leaf": [1, 3, 5, 10],
    "clf__max_features": ["sqrt", 0.5],
}
_LR_GRID = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l1", "l2"]}

GRIDS: dict[str, dict | list[dict]] = {
    "LR": _LR_GRID,
    "WLR": _LR_GRID,
    "RF": _RF_GRID,
    "WRF": _RF_GRID,
    "RF-SMOTE": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.5],
        "smote__k_neighbors": [3, 5],
    },
    "KNN": {
        "clf__n_neighbors": [1, 3, 5, 7, 11, 15, 25, 41],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    # RBF kernel only: a linear-kernel SVC with large C on overlapping classes can need hours
    # (libsvm has no default iteration cap), which is not a sensible search candidate here.
    "SVM": {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]},
    "NB": {"clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
    "ANN": {
        "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
        "clf__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "clf__learning_rate_init": [1e-3, 1e-2],
    },
    "WHGB": {
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [3, 5, None],
        "clf__max_leaf_nodes": [15, 31],
        "clf__l2_regularization": [0.0, 1.0, 10.0],
        "clf__max_iter": [100, 200],
    },
}

# ------------------------------------------------------------------ RandomizedSearchCV spaces
_RF_DIST = {
    "clf__n_estimators": randint(100, 501),
    "clf__max_depth": [3, 5, 8, 12, 16, None],
    "clf__min_samples_leaf": randint(1, 16),
    "clf__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
}
_LR_DIST = {"clf__C": loguniform(1e-3, 1e3), "clf__penalty": ["l1", "l2"]}

DISTRIBUTIONS: dict[str, dict] = {
    "LR": _LR_DIST,
    "WLR": _LR_DIST,
    "RF": _RF_DIST,
    "WRF": _RF_DIST,
    "RF-SMOTE": {**_RF_DIST, "smote__k_neighbors": randint(3, 8)},
    "KNN": {"clf__n_neighbors": randint(1, 61), "clf__weights": ["uniform", "distance"], "clf__p": [1, 2]},
    "SVM": {"clf__C": loguniform(1e-2, 1e3), "clf__gamma": loguniform(1e-4, 1e1), "clf__kernel": ["rbf"]},
    "NB": {"clf__var_smoothing": loguniform(1e-10, 1e-1)},
    "ANN": {
        "clf__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (64, 32, 16)],
        "clf__alpha": loguniform(1e-5, 1.0),
        "clf__learning_rate_init": loguniform(1e-4, 3e-2),
    },
    "WHGB": {
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__max_depth": [3, 4, 5, 6, None],
        "clf__max_leaf_nodes": randint(8, 64),
        "clf__l2_regularization": loguniform(1e-3, 10.0),
        "clf__max_iter": randint(50, 301),
    },
}


def grid_size(name: str) -> int:
    g = GRIDS[name]
    out = 0
    for sub in g if isinstance(g, list) else [g]:
        n = 1
        for v in sub.values():
            n *= len(v)
        out += n
    return out


def scoring_for(task: str) -> str:
    """Search objective: ROC-AUC (macro one-vs-rest for the multiclass task), as in the article."""
    return "roc_auc" if task == "binary" else "roc_auc_ovr"


def make_cv(seed: int = C.SEED, n_splits: int = C.CV_FOLDS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_search(name: str, method: str, task: str, *, n_iter: int = C.N_ITER_RANDOM,
                seed: int = C.SEED, n_jobs: int = -1, return_train_score: bool = True):
    """Build a GridSearchCV or RandomizedSearchCV for model *name* (stratified 5-fold CV)."""
    common = dict(scoring=scoring_for(task), cv=make_cv(seed), n_jobs=n_jobs, refit=True,
                  return_train_score=return_train_score, error_score="raise")
    est = build(name, seed)
    if method == "grid":
        return GridSearchCV(est, GRIDS[name], **common)
    if method == "random":
        return RandomizedSearchCV(est, DISTRIBUTIONS[name], n_iter=n_iter, random_state=seed, **common)
    raise ValueError(method)

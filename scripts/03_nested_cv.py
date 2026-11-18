"""Step 3 - nested cross-validation (paired scores for the statistical tests).

Outer loop : RepeatedStratifiedKFold, 5 folds x 2 repeats = 10 paired test scores per model.
Inner loop : RandomizedSearchCV (stratified 5-fold, 15 candidates) run on the outer-training part only.
For every (model, outer fold) we score
    - the model with default hyper-parameters   ("default")
    - the model tuned by the inner search       ("tuned")
on the *same* outer test fold, so before/after differences are paired.  Because the tuning is
repeated inside each outer training set, the "tuned" scores are not optimistically biased by the
selection of hyper-parameters (Cawley & Talbot, 2010).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold

from src import config as C
from src import data as D
from src import models as M
from src.metrics import compute_metrics

OUT = C.RESULTS_DIR / "nested"
OUT.mkdir(parents=True, exist_ok=True)


def one_job(key, task, model, fold, Xtr, ytr, Xte, yte, n_iter):
    rows = []
    # default hyper-parameters
    t0 = time.perf_counter()
    est = M.build(model).fit(Xtr, ytr)
    t_default = time.perf_counter() - t0
    rows.append({"setting": key, "model": model, "fold": fold, "variant": "default", "time_s": t_default,
                 "best_params": "", **compute_metrics(yte, est.predict_proba(Xte), task)})
    # tuned: inner randomized search (sequential inside the worker; parallelism is across jobs)
    t0 = time.perf_counter()
    s = M.make_search(model, "random", task, n_iter=n_iter, seed=C.SEED + fold, n_jobs=1,
                      return_train_score=False)
    s.fit(Xtr, ytr)
    t_tuned = time.perf_counter() - t0
    rows.append({"setting": key, "model": model, "fold": fold, "variant": "tuned", "time_s": t_tuned,
                 "best_params": str(s.best_params_), **compute_metrics(yte, s.predict_proba(Xte), task)})
    return rows


def run_setting(regime, task, fset, models, n_iter, outer_folds, outer_repeats):
    key = C.setting_key(regime, task, fset)
    X, y = D.get_xy(regime, task, fset)
    outer = RepeatedStratifiedKFold(n_splits=outer_folds, n_repeats=outer_repeats, random_state=C.SEED)
    jobs = []
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        for model in models:
            jobs.append((key, task, model, fold, X.iloc[tr], y[tr], X.iloc[te], y[te], n_iter))
    t0 = time.perf_counter()
    res = Parallel(n_jobs=-1, verbose=0)(delayed(one_job)(*j) for j in jobs)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(OUT / f"{key}.csv", index=False)
    piv = df.pivot_table(index="model", columns="variant", values="auc")
    print(f"=== {key} ({time.perf_counter() - t0:.0f}s)\n{piv.round(3).to_string()}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    models, n_iter, folds, reps = M.MODEL_NAMES, C.NESTED_INNER_ITER, C.NESTED_OUTER_FOLDS, C.NESTED_OUTER_REPEATS
    if args.quick:
        models, n_iter, folds, reps = ["LR", "WRF"], 2, 3, 1
        global OUT
        OUT = C.RESULTS_DIR / "nested_quick"
        OUT.mkdir(parents=True, exist_ok=True)
    for regime, task, fset in C.SETTINGS:
        if args.settings and C.setting_key(regime, task, fset) not in args.settings:
            continue
        run_setting(regime, task, fset, models, n_iter, folds, reps)


if __name__ == "__main__":
    main()

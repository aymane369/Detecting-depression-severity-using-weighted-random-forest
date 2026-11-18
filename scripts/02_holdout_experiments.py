"""Step 2 - hold-out experiments (article protocol).

For every setting (regime x task x feature set) and every model:
  1. default hyper-parameters            ("before optimisation")
  2. GridSearchCV   (stratified 5-fold, on the 70 % training set)
  3. RandomizedSearchCV (same CV, 30 candidates)
  4. Optuna TPE (Bayesian) search for WRF only, same budget as the random search
Each tuned model is refitted on the whole training set and evaluated once on the 30 % test set.

Outputs (results/holdout/<setting>_*)
  metrics.csv, best_params.joblib, proba.joblib (test-set probabilities), cv_<model>_<method>.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")  # inherited by joblib workers
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import cross_val_score

from src import config as C
from src import data as D
from src import models as M
from src.metrics import compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)
OUT = C.RESULTS_DIR / "holdout"
OUT.mkdir(parents=True, exist_ok=True)


def shrink_grids_for_smoke_test() -> None:
    """--quick: keep two values per hyper-parameter so the whole script runs in seconds."""
    for name, g in M.GRIDS.items():
        subs = g if isinstance(g, list) else [g]
        for sub in subs:
            for k in sub:
                sub[k] = sub[k][:2]


def optuna_search(name: str, Xtr, ytr, task: str, n_trials: int, seed: int):
    """Bayesian (TPE) search over the same space as RandomizedSearchCV, same CV and budget."""
    cv, scoring = M.make_cv(seed), M.scoring_for(task)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "clf__n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "clf__max_depth": trial.suggest_categorical("max_depth", [3, 5, 8, 12, 16, None]),
            "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "clf__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        }
        est = M.build(name, seed).set_params(**params)
        return cross_val_score(est, Xtr, ytr, cv=cv, scoring=scoring, n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.perf_counter() - t0
    best = {f"clf__{k}": v for k, v in study.best_params.items()}
    est = M.build(name, seed).set_params(**best).fit(Xtr, ytr)
    return est, best, study.best_value, elapsed, n_trials


def run_setting(regime: str, task: str, fset: str, models: list[str], n_iter: int, n_trials: int) -> None:
    key = C.setting_key(regime, task, fset)
    X, y = D.get_xy(regime, task, fset)
    Xtr, Xte, ytr, yte = D.holdout_split(X, y, regime, task)
    print(f"\n=== {key}: train={len(ytr)} test={len(yte)} classes(train)={np.bincount(ytr).tolist()} "
          f"classes(test)={np.bincount(yte).tolist()}", flush=True)

    records, probas, best_params = [], {"y_test": yte}, {}

    def record(model, method, est, proba, cv_auc, cv_train_auc, secs, n_fits, params):
        m = compute_metrics(yte, proba, task)
        records.append({"setting": key, "regime": regime, "task": task, "feature_set": fset, "model": model,
                        "method": method, **m, "cv_auc": cv_auc, "cv_train_auc": cv_train_auc,
                        "time_s": secs, "n_fits": n_fits})
        probas[f"{model}|{method}"] = proba
        best_params[f"{model}|{method}"] = params
        print(f"  {model:9s} {method:7s} AUC={m['auc']:.3f} F1w={m['f1']:.3f} F1m={m['f1_macro']:.3f} "
              f"cv={cv_auc:.3f} t={secs:.1f}s", flush=True)

    for name in models:
        # -- 1. default hyper-parameters
        est = M.build(name)
        t0 = time.perf_counter()
        est.fit(Xtr, ytr)
        secs = time.perf_counter() - t0
        cv_auc = cross_val_score(M.build(name), Xtr, ytr, cv=M.make_cv(), scoring=M.scoring_for(task), n_jobs=-1).mean()
        record(name, "default", est, est.predict_proba(Xte), cv_auc, np.nan, secs, 1, {})

        # -- 2./3. grid and random search
        for method in ("grid", "random"):
            s = M.make_search(name, method, task, n_iter=n_iter)
            t0 = time.perf_counter()
            s.fit(Xtr, ytr)
            secs = time.perf_counter() - t0
            cvr = pd.DataFrame(s.cv_results_)
            best_i = s.best_index_
            record(name, method, s.best_estimator_, s.best_estimator_.predict_proba(Xte), s.best_score_,
                   cvr.loc[best_i, "mean_train_score"], secs, len(cvr) * C.CV_FOLDS, s.best_params_)
            if name in ("RF", "WRF", "KNN"):  # kept for the hyper-parameter sensitivity analysis
                keep = [c for c in cvr.columns if c.startswith("param_") or c in
                        ("mean_test_score", "std_test_score", "mean_train_score", "rank_test_score")]
                cvr[keep].to_csv(OUT / f"{key}_cv_{name}_{method}.csv", index=False)

        # -- 4. Bayesian optimisation (Optuna TPE) for the weighted random forest only
        if name == "WRF":
            est, params, best_cv, secs, n_fits = optuna_search(name, Xtr, ytr, task, n_trials, C.SEED)
            record(name, "optuna", est, est.predict_proba(Xte), best_cv, np.nan, secs, n_fits * C.CV_FOLDS, params)

    pd.DataFrame(records).to_csv(OUT / f"{key}_metrics.csv", index=False)
    joblib.dump(best_params, OUT / f"{key}_best_params.joblib")
    joblib.dump(probas, OUT / f"{key}_proba.joblib")
    (OUT / f"{key}_best_params.json").write_text(json.dumps(
        {k: {p: (None if v is None else (list(v) if isinstance(v, tuple) else
                                         (float(v) if isinstance(v, (np.floating, float)) else
                                          (int(v) if isinstance(v, (np.integer, int)) else str(v)))))
             for p, v in d.items()} for k, d in best_params.items()}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="*", help="e.g. full-binary-ALL (default: all)")
    ap.add_argument("--quick", action="store_true", help="smoke test with tiny grids / two models")
    ap.add_argument("--out-dir", help="write outputs here instead of results/holdout (e.g. reproducibility check)")
    args = ap.parse_args()

    global OUT
    models, n_iter, n_trials = M.MODEL_NAMES, C.N_ITER_RANDOM, C.N_TRIALS_OPTUNA
    if args.quick:
        shrink_grids_for_smoke_test()
        models, n_iter, n_trials = ["LR", "WRF"], 3, 3
        OUT = C.RESULTS_DIR / "holdout_quick"
    if args.out_dir:
        OUT = Path(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    for regime, task, fset in C.SETTINGS:
        if args.settings and C.setting_key(regime, task, fset) not in args.settings:
            continue
        run_setting(regime, task, fset, models, n_iter, n_trials)


if __name__ == "__main__":
    main()

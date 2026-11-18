"""Step 5 - diagnostics and figures from the hold-out and nested-CV results.

Outputs
  results/holdout_all_metrics.csv, holdout_bootstrap_ci.csv, permutation_importance.csv, learning_curves.csv
  figures/before_after.*, confusion_matrices.*, roc_curves.*, learning_curves.*, hp_sensitivity.*,
          permutation_importance.*, nested_gain.*
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

from src import config as C
from src import data as D
from src import models as M
from src import plotting as P
from src.metrics import bootstrap_auc_ci

P.set_style()
HOLD = C.RESULTS_DIR / "holdout"

FEATURE_FR = {"Humanin": "Humanin", "MOTSc": "MOTS-c", "P66": "P66", "ACE-Update": "Haplotype ACE",
              "OtherMeds-Depression": "Antidépresseurs", "Gender": "Sexe", "BirthAge": "Âge",
              "DM-Status": "Diabète", "DM-MedUse": "Médic. diabète", "CVD-Revised": "Maladie cardiovasc.",
              "HT-Status": "Hypertension", "HT-MedUse": "Médic. hypertension"}


def setting_title(key: str) -> str:
    regime, task, fset = key.split("-")
    return (f"{'Jeu complet' if regime == 'full' else 'Sous-ens. propre'} · "
            f"{'binaire' if task == 'binary' else '4 classes'} · "
            f"{'biomarqueurs' if fset == 'MDP' else 'toutes variables'}")


def load_metrics() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(f) for f in sorted(HOLD.glob("*_metrics.csv"))], ignore_index=True)
    df.to_csv(C.RESULTS_DIR / "holdout_all_metrics.csv", index=False)
    return df


def best_method(metrics: pd.DataFrame, key: str, model: str) -> str:
    """Tuning method with the highest *cross-validated* AUC on the training set (never the test AUC)."""
    sub = metrics[(metrics.setting == key) & (metrics.model == model) & (metrics.method != "default")]
    return sub.sort_values("cv_auc", ascending=False).iloc[0]["method"]


def fit_tuned(key: str, model: str, method: str):
    """Rebuild a tuned pipeline from the stored best parameters and refit it on the training set."""
    regime, task, fset = key.split("-")
    X, y = D.get_xy(regime, task, fset)
    Xtr, Xte, ytr, yte = D.holdout_split(X, y, regime, task)
    params = joblib.load(HOLD / f"{key}_best_params.joblib")[f"{model}|{method}"]
    est = M.build(model).set_params(**params).fit(Xtr, ytr)
    return est, Xtr, ytr, Xte, yte


# ------------------------------------------------------------------------------ figures
def fig_before_after(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.2), sharex=True)
    cols = {"default": P.GREY, "grid": P.BLUE, "random": P.ORANGE}
    labs = {"default": "Défaut", "grid": "GridSearchCV", "random": "RandomizedSearchCV"}
    keys = [C.setting_key(*s) for s in C.SETTINGS]
    for ax, key in zip(axes.ravel(), keys):
        sub = metrics[metrics.setting == key]
        x = np.arange(len(M.MODEL_NAMES))
        for k, meth in enumerate(["default", "grid", "random"]):
            v = [sub[(sub.model == m) & (sub.method == meth)]["auc"].iloc[0] for m in M.MODEL_NAMES]
            ax.bar(x + (k - 1) * 0.27, v, 0.27, color=cols[meth], label=labs[meth])
        ax.set_xticks(x, M.MODEL_NAMES, rotation=60, ha="right", fontsize=7)
        ax.set_title(setting_title(key), fontsize=7.5)
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
    axes[0, 0].set_ylabel("AUC (test)")
    axes[1, 0].set_ylabel("AUC (test)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    P.save(fig, "before_after")


def fig_confusion(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 5.0))
    for r, key in enumerate(["full-binary-ALL", "full-multi-ALL"]):
        task = key.split("-")[1]
        pr = joblib.load(HOLD / f"{key}_proba.joblib")
        yte = pr["y_test"]
        meth = best_method(metrics, key, "WRF")
        for c, (lab, pk) in enumerate([("RF par défaut", "RF|default"), (f"WRF optimisé ({meth})", f"WRF|{meth}")]):
            cm = confusion_matrix(yte, pr[pk].argmax(1))
            norm = cm / cm.sum(1, keepdims=True)
            ax = axes[r, c]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            names = P.class_names_fr(task)
            ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.55 else "black")
            ax.set_title(f"{'Binaire' if task == 'binary' else '4 classes'} – {lab}", fontsize=7.5)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Réel")
            ax.grid(False)
    fig.tight_layout()
    P.save(fig, "confusion_matrices")


def fig_roc(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, key in zip(axes[:2], ["full-binary-ALL", "clean-binary-ALL"]):
        pr = joblib.load(HOLD / f"{key}_proba.joblib")
        yte = pr["y_test"]
        entries = [("RF défaut", "RF|default", P.GREY, "-"),
                   (f"RF optimisé", f"RF|{best_method(metrics, key, 'RF')}", P.BLUE, "-"),
                   ("WRF défaut", "WRF|default", P.ORANGE, "--"),
                   (f"WRF optimisé", f"WRF|{best_method(metrics, key, 'WRF')}", P.RED, "-"),
                   ("RF-SMOTE opt.", f"RF-SMOTE|{best_method(metrics, key, 'RF-SMOTE')}", P.GREEN, ":")]
        for lab, pk, col, ls in entries:
            fpr, tpr, _ = roc_curve(yte, pr[pk][:, 1])
            auc = metrics[(metrics.setting == key) & (metrics.model == pk.split("|")[0]) &
                          (metrics.method == pk.split("|")[1])]["auc"].iloc[0]
            ax.plot(fpr, tpr, color=col, ls=ls, lw=1.3, label=f"{lab} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
        ax.set_title(("(a) " if "full" in key else "(b) ") + setting_title(key), fontsize=7)
        ax.set_xlabel("Taux de faux positifs")
        ax.legend(fontsize=6, loc="lower right")
    axes[0].set_ylabel("Taux de vrais positifs")
    key = "full-multi-ALL"
    pr = joblib.load(HOLD / f"{key}_proba.joblib")
    meth = best_method(metrics, key, "WRF")
    yb = label_binarize(pr["y_test"], classes=range(4))
    ax = axes[2]
    for k in range(4):
        fpr, tpr, _ = roc_curve(yb[:, k], pr[f"WRF|{meth}"][:, k])
        ax.plot(fpr, tpr, color=P.SEVERITY_COLORS[k], lw=1.2, label=P.CLASS_FR_MULTI[k])
    auc = metrics[(metrics.setting == key) & (metrics.model == "WRF") & (metrics.method == meth)]["auc"].iloc[0]
    ax.plot([0, 1], [0, 1], color="k", lw=0.6, ls=":")
    ax.set_title(f"(c) WRF optimisé, 4 classes (AUC macro {auc:.2f})", fontsize=7)
    ax.set_xlabel("Taux de faux positifs")
    ax.legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    P.save(fig, "roc_curves")


def fig_learning_curves(metrics: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    rows = []
    for ax, key in zip(axes, ["full-binary-ALL", "full-multi-ALL", "clean-binary-ALL"]):
        regime, task, fset = key.split("-")
        meth = best_method(metrics, key, "WRF")
        _, Xtr, ytr, _, _ = fit_tuned(key, "WRF", meth)
        for lab, est, col in [("RF défaut", M.build("RF"), P.GREY),
                              (f"WRF optimisé", fit_tuned(key, "WRF", meth)[0], P.RED)]:
            sizes, tr, va = learning_curve(est, Xtr, ytr, cv=M.make_cv(), scoring=M.scoring_for(task),
                                           train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1, shuffle=True,
                                           random_state=C.SEED)
            ax.plot(sizes, va.mean(1), color=col, lw=1.4, label=f"{lab} – validation")
            ax.fill_between(sizes, va.mean(1) - va.std(1), va.mean(1) + va.std(1), color=col, alpha=0.15)
            ax.plot(sizes, tr.mean(1), color=col, lw=1, ls="--", label=f"{lab} – entraînement")
            rows += [{"setting": key, "model": lab, "train_size": int(s), "train_auc": a, "val_auc": b, "val_sd": c}
                     for s, a, b, c in zip(sizes, tr.mean(1), va.mean(1), va.std(1))]
        ax.set_title(setting_title(key), fontsize=7)
        ax.set_xlabel("Taille de l'échantillon d'entraînement")
        ax.set_ylim(0.4, 1.02)
    axes[0].set_ylabel("AUC")
    axes[0].legend(fontsize=5.5, loc="lower right")
    fig.tight_layout()
    P.save(fig, "learning_curves")
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "learning_curves.csv", index=False)
    return out


def fig_hp_sensitivity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for ax, key in zip(axes, ["full-binary-ALL", "clean-binary-ALL"]):
        cv = pd.read_csv(HOLD / f"{key}_cv_RF_grid.csv")
        cv["depth"] = cv["param_clf__max_depth"].fillna("None").astype(str).str.replace(".0", "", regex=False)
        piv = cv.pivot_table(index="depth", columns="param_clf__min_samples_leaf", values="mean_test_score")
        piv = piv.loc[["3", "5", "10", "None"]]
        im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns.astype(int))
        ax.set_yticks(range(piv.shape[0]), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if piv.values[i, j] < piv.values.mean() else "black")
        ax.set_xlabel("min_samples_leaf")
        ax.set_ylabel("max_depth")
        ax.set_title(("(a) " if "full" in key else "(b) ") + setting_title(key), fontsize=7.5)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, label="AUC (validation croisée)")
    fig.tight_layout()
    P.save(fig, "hp_sensitivity")


def fig_permutation_importance(metrics: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0))
    rows = []
    for ax, key in zip(axes, ["full-binary-ALL", "full-multi-ALL", "clean-binary-ALL"]):
        task = key.split("-")[1]
        meth = best_method(metrics, key, "WRF")
        est, _, _, Xte, yte = fit_tuned(key, "WRF", meth)
        pi = permutation_importance(est, Xte, yte, scoring=M.scoring_for(task), n_repeats=30,
                                    random_state=C.SEED, n_jobs=-1)
        order = np.argsort(pi.importances_mean)
        names = [FEATURE_FR[c] for c in Xte.columns]
        ax.barh([names[i] for i in order], pi.importances_mean[order], xerr=pi.importances_std[order],
                color=[P.ORANGE if Xte.columns[i] in C.BIOMARKERS else P.BLUE for i in order],
                error_kw={"lw": 0.7})
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(setting_title(key), fontsize=7)
        ax.set_xlabel("Baisse d'AUC après permutation")
        for i, c in enumerate(Xte.columns):
            rows.append({"setting": key, "feature": c, "importance_mean": pi.importances_mean[i],
                         "importance_std": pi.importances_std[i]})
    fig.tight_layout()
    P.save(fig, "permutation_importance")
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "permutation_importance.csv", index=False)
    return out


def fig_nested_gain() -> None:
    files = sorted((C.RESULTS_DIR / "nested").glob("*.csv"))
    if not files:
        return
    df = pd.concat([pd.read_csv(f) for f in files])
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.0), sharex=True)
    for ax, key in zip(axes.ravel(), [C.setting_key(*s) for s in C.SETTINGS]):
        g = df[df.setting == key]
        data = []
        for m in M.MODEL_NAMES:
            a = g[(g.model == m) & (g.variant == "default")].sort_values("fold")["auc"].to_numpy()
            b = g[(g.model == m) & (g.variant == "tuned")].sort_values("fold")["auc"].to_numpy()
            data.append(b - a)
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops={"color": "k"},
                        flierprops={"markersize": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor(P.SKY)
        ax.axhline(0, color=P.RED, lw=0.8, ls="--")
        ax.set_xticks(range(1, len(M.MODEL_NAMES) + 1), M.MODEL_NAMES, rotation=60, ha="right", fontsize=7)
        ax.set_title(setting_title(key), fontsize=7.5)
    axes[0, 0].set_ylabel("Gain d'AUC (optimisé − défaut)")
    axes[1, 0].set_ylabel("Gain d'AUC (optimisé − défaut)")
    fig.tight_layout()
    P.save(fig, "nested_gain")


def bootstrap_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in [C.setting_key(*s) for s in C.SETTINGS]:
        task = key.split("-")[1]
        pr = joblib.load(HOLD / f"{key}_proba.joblib")
        for model in ("RF", "WRF"):
            for meth in ("default", best_method(metrics, key, model)):
                lo, hi = bootstrap_auc_ci(pr["y_test"], pr[f"{model}|{meth}"], task)
                auc = metrics[(metrics.setting == key) & (metrics.model == model) & (metrics.method == meth)]["auc"].iloc[0]
                rows.append({"setting": key, "model": model, "method": meth, "auc": auc, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / "holdout_bootstrap_ci.csv", index=False)
    return out


def main() -> None:
    metrics = load_metrics()
    fig_before_after(metrics)
    fig_confusion(metrics)
    fig_roc(metrics)
    fig_hp_sensitivity()
    fig_learning_curves(metrics)
    fig_permutation_importance(metrics)
    fig_nested_gain()
    print(bootstrap_table(metrics).round(3).to_string())


if __name__ == "__main__":
    main()

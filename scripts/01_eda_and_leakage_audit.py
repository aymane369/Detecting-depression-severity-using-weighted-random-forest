"""Step 1 - exploratory analysis and data-leakage audit.

Outputs
  results/dataset_summary.csv      descriptive statistics (Table 1 analogue)
  results/leakage_audit.json       shared-value rates, outcome purity, connected components
  results/leakage_audit_auc.csv    Random-Forest AUC under several validation protocols
  figures/eda_distribution.*, eda_biomarkers.*, eda_correlation.*, leakage_audit.*
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src import config as C
from src import data as D
from src import plotting as P

P.set_style()
C.RESULTS_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------------- descriptive stats
def dataset_summary(raw: pd.DataFrame, enc: pd.DataFrame) -> None:
    rows = []
    for c in C.BIOMARKERS + ["BirthAge", C.TARGET_COL]:
        s = enc[c]
        rows.append({"variable": c, "type": "numeric", "min": s.min(), "max": s.max(),
                     "mean": s.mean(), "std": s.std(), "n_unique": s.nunique()})
    for c in ["ACE-Update", "OtherMeds-Depression", "Gender", "DM-Status", "DM-MedUse",
              "CVD-Revised", "HT-Status", "HT-MedUse"]:
        vc = raw[c].astype(str).value_counts()
        rows.append({"variable": c, "type": "categorical", "min": None, "max": None, "mean": None,
                     "std": None, "n_unique": raw[c].nunique(), "levels": json.dumps(vc.to_dict())})
    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "dataset_summary.csv", index=False)


def eda_figures(enc: pd.DataFrame) -> None:
    # (a) PHQ-9 distribution coloured by severity class + class counts
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [2.2, 1]})
    counts = enc[C.TARGET_COL].value_counts().sort_index()
    cls = pd.cut(counts.index, C.PHQ_BINS, labels=False)
    ax[0].bar(counts.index, counts.values, color=[P.SEVERITY_COLORS[k] for k in cls], width=0.85)
    ax[0].set_xlabel("Score PHQ-9")
    ax[0].set_ylabel("Effectif")
    ax[0].set_title("(a) Distribution du score PHQ-9")
    n4 = enc["y_multi"].value_counts().sort_index()
    bars = ax[1].bar(P.CLASS_FR_MULTI, n4.values, color=P.SEVERITY_COLORS)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("Effectif (échelle log)")
    ax[1].set_title("(b) Classes de sévérité")
    for b, v in zip(bars, n4.values):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v}\n({v / len(enc):.1%})", ha="center", fontsize=7)
    ax[1].set_ylim(10, 6000)
    ax[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    P.save(fig, "eda_distribution")

    # (b) biomarkers by severity class
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for a, c in zip(axes, C.BIOMARKERS):
        sns.boxplot(data=enc, x="y_multi", y=c, hue="y_multi", palette=P.SEVERITY_COLORS, ax=a,
                    fliersize=1.5, linewidth=0.8, legend=False)
        a.set_yscale("log")
        a.set_xticks(range(4), ["Abs.", "Lég.", "Mod.", "Sév."])
        a.set_xlabel("Sévérité")
        a.set_ylabel(c)
    fig.tight_layout()
    P.save(fig, "eda_biomarkers")

    # (c) Pearson correlation matrix (as in the article, Fig. 2)
    cols = C.BIOMARKERS + C.CLINICAL + [C.TARGET_COL]
    corr = enc[cols].corr()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 5.5}, cbar_kws={"shrink": 0.7}, ax=ax, linewidths=0.3)
    ax.set_title("Matrice de corrélation de Pearson")
    ax.grid(False)
    fig.tight_layout()
    P.save(fig, "eda_correlation")


# ----------------------------------------------------------------------------- leakage audit
def purity(df: pd.DataFrame, key: list[str] | str, label: str = "y_multi", labels=None) -> float:
    """Support-weighted share of the majority class inside groups of >=2 rows sharing *key*."""
    lab = df[label] if labels is None else pd.Series(labels, index=df.index)
    g = df.assign(_l=lab).groupby(key)
    sizes = g["_l"].transform("size")
    sub = df.assign(_l=lab)[sizes > 1]
    maj = sub.groupby(key)["_l"].agg(lambda s: s.value_counts().iloc[0])
    tot = sub.groupby(key).size()
    return float(maj.sum() / tot.sum())


def audit_values(raw_enc: pd.DataFrame) -> dict:
    out = {"n_rows": len(raw_enc)}
    rng = np.random.default_rng(C.SEED)
    for c in C.BIOMARKERS:
        sh = raw_enc.groupby(c)[c].transform("size") > 1
        perm = np.mean([purity(raw_enc, c, labels=rng.permutation(raw_enc["y_multi"].values)) for _ in range(20)])
        out[c] = {"n_unique": int(raw_enc[c].nunique()), "share_rows_shared": float(sh.mean()),
                  "purity_observed": purity(raw_enc, c), "purity_permuted": float(perm),
                  "max_multiplicity": int(raw_enc.groupby(c)[c].transform("size").max())}
    out["triplet"] = {
        "n_unique": int(raw_enc[C.BIOMARKERS].drop_duplicates().shape[0]),
        "share_rows_shared": float((raw_enc.groupby(C.BIOMARKERS)["y_multi"].transform("size") > 1).mean()),
        "purity_observed": purity(raw_enc, C.BIOMARKERS),
        "purity_permuted": float(np.mean([
            purity(raw_enc, C.BIOMARKERS, labels=rng.permutation(raw_enc["y_multi"].values)) for _ in range(20)])),
    }
    # connected components when rows sharing ANY biomarker value are linked
    parent = list(range(len(raw_enc)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c in C.BIOMARKERS:
        for _, idx in raw_enc.groupby(c).indices.items():
            for j in idx[1:]:
                parent[find(j)] = find(idx[0])
    comp = pd.Series([find(i) for i in range(len(raw_enc))]).value_counts()
    out["components"] = {"n_components": int(len(comp)), "largest": int(comp.iloc[0]),
                         "n_singletons": int((comp == 1).sum())}
    clean = ~D.shared_value_mask(raw_enc)
    out["clean_subset"] = {"n": int(clean.sum()), "prevalence_depressed": float(raw_enc.loc[clean, "y_binary"].mean()),
                           "class_counts": raw_enc.loc[clean, "y_multi"].value_counts().sort_index().to_dict()}
    # impossible-for-one-person check: biggest identical-triplet group mixes sexes and genotypes
    top = raw_enc.groupby(C.BIOMARKERS).size().idxmax()
    grp = raw_enc[(raw_enc[C.BIOMARKERS] == pd.Series(top, index=C.BIOMARKERS)).all(axis=1)]
    out["largest_triplet_group"] = {"size": int(len(grp)), "n_gender": int(grp["Gender"].nunique()),
                                    "n_ace": int(grp["ACE-Update"].nunique()),
                                    "age_min": int(grp["BirthAge"].min()), "age_max": int(grp["BirthAge"].max()),
                                    "phq_values": sorted(grp[C.TARGET_COL].unique().tolist())}
    return out


def _cv_scores(X, y, protocol: str, groups, seed: int) -> list[float]:
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    if protocol == "group":
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        splits = cv.split(X, y, groups)
    else:
        splits = StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)
    sc = []
    for tr, te in splits:
        rf.fit(X.iloc[tr], y[tr])
        sc.append(roc_auc_score(y[te], rf.predict_proba(X.iloc[te])[:, 1]))
    return sc


def audit_auc(raw_enc: pd.DataFrame, n_repeats: int = 5) -> pd.DataFrame:
    """Binary RF AUC under protocols of increasing strictness (default RF, 5 repeats x 5 folds)."""
    clean_mask = ~D.shared_value_mask(raw_enc)
    gid = raw_enc.groupby(C.BIOMARKERS).ngroup().to_numpy()
    first = ~raw_enc.duplicated(C.BIOMARKERS, keep="first")
    y_all = raw_enc["y_binary"].to_numpy()
    jobs = []
    for fset, cols in C.FEATURE_SETS.items():
        X = raw_enc[cols]
        defs = [
            ("random_all", X, y_all, None, "random"),
            ("group_triplet_all", X, y_all, gid, "group"),
            ("one_row_per_triplet", X[first.values], y_all[first.values], None, "random"),
            ("clean_subset", X[clean_mask.values].reset_index(drop=True), y_all[clean_mask.values], None, "random"),
            ("shuffled_all", X, y_all, None, "random"),
            ("shuffled_clean", X[clean_mask.values].reset_index(drop=True), y_all[clean_mask.values], None, "random"),
        ]
        for name, Xp, yp, g, proto in defs:
            Xp = Xp.reset_index(drop=True)
            for r in range(n_repeats):
                # label-permutation controls use a fresh permutation per repeat
                yr = np.random.default_rng(C.SEED + r).permutation(yp) if name.startswith("shuffled") else yp
                jobs.append((fset, name, len(yr), Xp, yr, proto, g, C.SEED + r))
    res = Parallel(n_jobs=-1)(delayed(_cv_scores)(Xp, yp, proto, g, seed) for _, _, _, Xp, yp, proto, g, seed in jobs)
    rows = []
    for (fset, name, n, *_), sc in zip(jobs, res):
        rows += [{"feature_set": fset, "protocol": name, "n_rows": n, "auc": s} for s in sc]
    df = pd.DataFrame(rows)
    return df


def audit_figure(audit: dict, auc_long: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1, 1.6]})
    names = C.BIOMARKERS
    x = np.arange(len(names))
    sh = [audit[c]["share_rows_shared"] * 100 for c in names]
    ax[0].bar(x, sh, color=P.BLUE, width=0.55)
    for xi, v in zip(x, sh):
        ax[0].text(xi, v + 1.5, f"{v:.0f} %", ha="center", fontsize=8)
    ax[0].set_xticks(x, names)
    ax[0].set_ylim(0, 105)
    ax[0].set_ylabel("Lignes dont la valeur est partagée (%)")
    ax[0].set_title("(a) Valeurs dupliquées")

    order = ["random_all", "group_triplet_all", "one_row_per_triplet", "clean_subset", "shuffled_all"]
    labels = ["Aléatoire\n(n=2621)", "Groupé par\ntriplet", "1 ligne par\ntriplet", "Sous-ens.\npropre (n=228)",
              "Étiquettes\npermutées"]
    w = 0.38
    for k, (fset, col) in enumerate([("MDP", P.ORANGE), ("ALL", P.BLUE)]):
        m = auc_long[auc_long.feature_set == fset].groupby("protocol")["auc"].agg(["mean", "std"]).loc[order]
        pos = np.arange(len(order)) + (k - 0.5) * w
        ax[1].bar(pos, m["mean"], w, yerr=m["std"], color=col, capsize=2, error_kw={"lw": 0.8},
                  label="Biomarqueurs seuls" if fset == "MDP" else "Toutes variables")
        for p_, v, s_ in zip(pos, m["mean"], m["std"]):
            ax[1].text(p_, v + s_ + 0.012, f"{v:.2f}", ha="center", fontsize=6.5)
    ax[1].axhline(0.5, color=P.GREY, ls="--", lw=0.8)
    ax[1].set_xticks(np.arange(len(order)), labels, fontsize=7)
    ax[1].set_ylim(0.3, 1.05)
    ax[1].set_ylabel("AUC (Random Forest par défaut)")
    ax[1].set_title("(b) AUC selon le protocole de validation")
    ax[1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    P.save(fig, "leakage_audit")


def main() -> None:
    raw = D.load_raw()
    enc = D.load_dataset("full")
    dataset_summary(raw, enc)
    eda_figures(enc)
    audit = audit_values(enc)
    auc_long = audit_auc(enc)
    auc_long.to_csv(C.RESULTS_DIR / "leakage_audit_auc.csv", index=False)
    (C.RESULTS_DIR / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    audit_figure(audit, auc_long)
    print(json.dumps(audit, indent=2))
    print(auc_long.groupby(["feature_set", "protocol"])["auc"].agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()

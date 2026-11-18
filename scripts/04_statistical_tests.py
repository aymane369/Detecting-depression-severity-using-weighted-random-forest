"""Step 4 - statistical tests on the paired nested-CV scores.

(a) Before/after tuning, per model     : paired t-test, Nadeau-Bengio corrected resampled t-test,
                                          Wilcoxon signed-rank (Holm-adjusted across models).
(b) Between models (article protocol)  : Friedman test + Conover post-hoc (reference model vs others).
(c) Feature sets (WRF)                 : all features vs biomarkers only.

Outputs: results/stats_tuning_gain.csv, stats_friedman.csv, stats_featureset.csv, nested_summary.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import config as C
from src import models as M

NESTED = C.RESULTS_DIR / "nested"
K = C.NESTED_OUTER_FOLDS

GROUPS = {  # model groups compared with Friedman + Conover; reference model first
    "base6": (["RF", "LR", "KNN", "SVM", "NB", "ANN"], "RF"),
    "weighted": (["WRF", "WLR", "RF-SMOTE"], "WRF"),
    "all10": (M.MODEL_NAMES, "WRF"),
}


def load_nested() -> pd.DataFrame:
    files = sorted(NESTED.glob("*.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def corrected_resampled_ttest(d: np.ndarray, k: int = K) -> tuple[float, float]:
    """Nadeau & Bengio (2003): variance inflated by (1/J + n_test/n_train) because CV folds share data."""
    j = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / j + 1.0 / (k - 1)) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=j - 1))


def tuning_gain(df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby("setting"):
        part = []
        for model in M.MODEL_NAMES:
            a = g[(g.model == model) & (g.variant == "default")].sort_values("fold")[metric].to_numpy()
            b = g[(g.model == model) & (g.variant == "tuned")].sort_values("fold")[metric].to_numpy()
            d = b - a
            t_nb, p_nb = corrected_resampled_ttest(d)
            try:
                p_w = stats.wilcoxon(d).pvalue if np.any(d != 0) else np.nan
            except ValueError:
                p_w = np.nan
            part.append({"setting": key, "model": model, "metric": metric, "n_pairs": len(d),
                         "mean_default": a.mean(), "mean_tuned": b.mean(), "gain": d.mean(), "sd_gain": d.std(ddof=1),
                         "p_ttest": stats.ttest_rel(b, a).pvalue, "p_ttest_corrected": p_nb, "p_wilcoxon": p_w})
        part = pd.DataFrame(part)
        ok = part.p_wilcoxon.notna()
        part["p_wilcoxon_holm"] = np.nan
        part.loc[ok, "p_wilcoxon_holm"] = multipletests(part.loc[ok, "p_wilcoxon"], method="holm")[1]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def friedman_conover(df: pd.DataFrame, metric: str = "auc") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby("setting"):
        for variant in ("default", "tuned"):
            for gname, (models, ref) in GROUPS.items():
                wide = g[(g.variant == variant) & g.model.isin(models)].pivot(index="fold", columns="model", values=metric)[models]
                chi2, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in models])
                post = sp.posthoc_conover_friedman(wide.to_numpy(), p_adjust="holm")  # Holm over all pairs
                post.index = post.columns = models
                ranks = wide.rank(axis=1, ascending=False).mean()
                rows.append({"setting": key, "variant": variant, "group": gname, "reference": ref, "other": "(all)",
                             "chi2": chi2, "p_friedman": p, "p_conover": np.nan, "mean_rank_ref": ranks[ref]})
                for other in models:
                    if other != ref:
                        rows.append({"setting": key, "variant": variant, "group": gname, "reference": ref,
                                     "other": other, "chi2": chi2, "p_friedman": p,
                                     "p_conover": post.loc[ref, other], "mean_rank_ref": ranks[ref]})
    return pd.DataFrame(rows)


def feature_set_test(df: pd.DataFrame, model: str = "WRF", metric: str = "auc") -> pd.DataFrame:
    rows = []
    for regime, task in sorted({(s.split("-")[0], s.split("-")[1]) for s in df.setting.unique()}):
        a = df[(df.setting == f"{regime}-{task}-ALL") & (df.model == model) & (df.variant == "tuned")].sort_values("fold")[metric].to_numpy()
        b = df[(df.setting == f"{regime}-{task}-MDP") & (df.model == model) & (df.variant == "tuned")].sort_values("fold")[metric].to_numpy()
        if len(a) == 0 or len(b) == 0:
            continue
        d = a - b
        rows.append({"regime": regime, "task": task, "model": model, "mean_ALL": a.mean(), "mean_MDP": b.mean(),
                     "gain": d.mean(), "p_wilcoxon": stats.wilcoxon(d).pvalue, "p_ttest_corrected": corrected_resampled_ttest(d)[1]})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_nested()
    summ = (df.groupby(["setting", "model", "variant"])[["auc", "auc_pr", "f1", "f1_macro", "balanced_acc", "accuracy", "time_s"]]
            .agg(["mean", "std"]))
    summ.columns = ["_".join(c) for c in summ.columns]
    summ.reset_index().to_csv(C.RESULTS_DIR / "nested_summary.csv", index=False)

    gain = tuning_gain(df, "auc")
    gain_f1 = tuning_gain(df, "f1_macro")
    pd.concat([gain, gain_f1]).to_csv(C.RESULTS_DIR / "stats_tuning_gain.csv", index=False)
    fr = friedman_conover(df, "auc")
    fr.to_csv(C.RESULTS_DIR / "stats_friedman.csv", index=False)
    fs = feature_set_test(df)
    fs.to_csv(C.RESULTS_DIR / "stats_featureset.csv", index=False)

    pd.set_option("display.width", 200)
    print(gain[["setting", "model", "mean_default", "mean_tuned", "gain", "p_ttest_corrected", "p_wilcoxon", "p_wilcoxon_holm"]].round(4).to_string())
    print(fr[(fr.other == "(all)")][["setting", "variant", "group", "chi2", "p_friedman"]].round(5).to_string())
    print(fs.round(4).to_string())


if __name__ == "__main__":
    main()

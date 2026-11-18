"""Step 6 - generate the LaTeX tables of the (French) report from the result CSV files.

Every number printed in the report tables comes from results/*.csv; nothing is typed by hand.
Output: report/tables/*.tex  (included with \\input in report/rapport.tex)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config as C
from src import models as M

R = C.RESULTS_DIR
T = C.TABLE_DIR
T.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------ formatting helpers
def f3(x, bold=False) -> str:
    if pd.isna(x):
        return "--"
    s = f"{x:.3f}".replace(".", ",")
    return f"\\textbf{{{s}}}" if bold else s


def fp(p) -> str:
    """p-value in French typography."""
    if pd.isna(p):
        return "--"
    if p < 0.001:
        return "$<\\!10^{-3}$"
    return f"{p:.3f}".replace(".", ",")


def fg(x) -> str:
    """signed gain."""
    if pd.isna(x):
        return "--"
    return f"{x:+.3f}".replace(".", ",").replace("-", "$-$")


def write(name: str, body: str) -> None:
    (T / f"{name}.tex").write_text(body, encoding="utf-8")


def tabular(colspec: str, header: list[str], rows: list[list[str]], midrules: list[int] | None = None) -> str:
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    # a header line that already ends with a row break or a rule must not get another row break
    ends_ok = re.compile(r"(\\\\|\\midrule|\\cmidrule(\([a-z]*\))?\{[\d-]+\})\s*$")
    out += [h if ends_ok.search(h) else h + " \\\\" for h in header]
    out.append("\\midrule")
    for i, r in enumerate(rows):
        out.append(" & ".join(r) + " \\\\")
        if midrules and i in midrules:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


SET_SHORT = {"full-binary-ALL": "Complet, binaire, toutes var.", "full-binary-MDP": "Complet, binaire, biomarq.",
             "full-multi-ALL": "Complet, 4 classes, toutes var.", "full-multi-MDP": "Complet, 4 classes, biomarq.",
             "clean-binary-ALL": "Propre, binaire, toutes var.", "clean-binary-MDP": "Propre, binaire, biomarq."}


def best_method(hold: pd.DataFrame, key: str, model: str) -> str:
    sub = hold[(hold.setting == key) & (hold.model == model) & (hold.method != "default")]
    return sub.sort_values("cv_auc", ascending=False).iloc[0]["method"]


# ------------------------------------------------------------------------------ tables
def tab_baseline(hold: pd.DataFrame) -> None:
    """Baseline (default hyper-parameters) of the six article models, all features (article Tables 5 and 9)."""
    rows = []
    for m in M.BASE_MODELS:
        r = []
        for key in ("full-binary-ALL", "full-multi-ALL"):
            s = hold[(hold.setting == key) & (hold.method == "default") & hold.model.isin(M.BASE_MODELS)]
            x = s[s.model == m].iloc[0]
            r += [f3(x.accuracy, x.accuracy == s.accuracy.max()), f3(x.f1, x.f1 == s.f1.max()),
                  f3(x.f1_macro, x.f1_macro == s.f1_macro.max()), f3(x.auc, x.auc == s.auc.max()),
                  f3(x.auc_pr, x.auc_pr == s.auc_pr.max())]
        rows.append([m] + r)
    header = ["& \\multicolumn{5}{c}{Binaire} & \\multicolumn{5}{c}{4 classes} \\\\ \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
              "Modèle & Acc. & F1$_p$ & F1$_m$ & AUC & AUC-PR & Acc. & F1$_p$ & F1$_m$ & AUC & AUC-PR"]
    write("tab_baseline", tabular("l" + "c" * 10, header, rows))


def tab_before_after(hold: pd.DataFrame) -> None:
    keys = ["full-binary-ALL", "full-multi-ALL", "clean-binary-ALL"]
    rows = []
    for m in M.MODEL_NAMES:
        r = [m]
        for key in keys:
            vals = [hold[(hold.setting == key) & (hold.model == m) & (hold.method == meth)].auc.iloc[0]
                    for meth in ("default", "grid", "random")]
            r += [f3(v, v == max(vals) and max(vals) > vals[0]) for v in vals]
        rows.append(r)
    header = ["& \\multicolumn{3}{c}{Complet, binaire} & \\multicolumn{3}{c}{Complet, 4 classes} & \\multicolumn{3}{c}{Propre, binaire} \\\\"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
              "Modèle & Déf. & Grid & Rand. & Déf. & Grid & Rand. & Déf. & Grid & Rand."]
    write("tab_before_after", tabular("l" + "ccc" * 3, header, rows, midrules=[5]))


def tab_weighted(hold: pd.DataFrame) -> None:
    """Weighted models (article Table 13 analogue): default vs tuned (method chosen by CV score)."""
    blocks = []
    for key in ("full-multi-ALL",):  # the binary counterpart is given by Table 'before/after' and the text
        rows = []
        for m in ["RF", "WLR", "WRF", "RF-SMOTE", "WHGB"]:
            for meth in ("default", best_method(hold, key, m)):
                x = hold[(hold.setting == key) & (hold.model == m) & (hold.method == meth)].iloc[0]
                lab = f"{m}" + ("" if meth == "default" else f" ({meth})")
                rows.append([lab, f3(x.accuracy), f3(x.f1), f3(x.f1_macro), f3(x.balanced_acc), f3(x.auc), f3(x.auc_pr)])
        wo = hold[(hold.setting == key) & (hold.model == "WRF") & (hold.method == "optuna")]
        if len(wo):
            x = wo.iloc[0]
            rows.append(["WRF (optuna)", f3(x.accuracy), f3(x.f1), f3(x.f1_macro), f3(x.balanced_acc), f3(x.auc), f3(x.auc_pr)])
        header = [f"\\multicolumn{{7}}{{l}}{{\\textit{{{SET_SHORT[key]}}}}} \\\\ \\midrule",
                  "Modèle & Acc. & F1$_p$ & F1$_m$ & Acc.$_{éq}$ & AUC & AUC-PR"]
        blocks.append(tabular("lcccccc", header, rows))
    write("tab_weighted", "\n\\medskip\n".join(blocks))


def tab_time(hold: pd.DataFrame) -> None:
    key = "full-binary-ALL"
    rows = []
    for m in M.MODEL_NAMES:
        s = hold[(hold.setting == key) & (hold.model == m)]
        g = s[s.method == "grid"].iloc[0]
        r_ = s[s.method == "random"].iloc[0]
        d = s[s.method == "default"].iloc[0]
        rows.append([m, f"{d.time_s:.2f}".replace(".", ","), f"{int(g.n_fits)}", f"{g.time_s:.0f}",
                     f"{int(r_.n_fits)}", f"{r_.time_s:.0f}", f3(d.cv_auc), f3(g.cv_auc), f3(r_.cv_auc)])
    wo = hold[(hold.setting == key) & (hold.model == "WRF") & (hold.method == "optuna")]
    header = ["Modèle & Déf. & \\multicolumn{2}{c}{Grid} & \\multicolumn{2}{c}{Random} & \\multicolumn{3}{c}{AUC-VC} \\\\"
              "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-9}",
              " & (s) & fits & s & fits & s & Déf. & Grid & Rand."]
    body = tabular("lcrrrrccc", header, rows)
    if len(wo):
        body += f"\n\n\\smallskip\\noindent\\footnotesize Optuna (TPE, WRF) : {int(wo.iloc[0].n_fits)} fits, {wo.iloc[0].time_s:.0f}\\,s."
    write("tab_time", body)


def tab_gain() -> None:
    g = pd.read_csv(R / "stats_tuning_gain.csv")
    g = g[g.metric == "auc"]
    keys = ["full-binary-ALL", "full-multi-ALL", "clean-binary-ALL"]
    rows = []
    for m in M.MODEL_NAMES:
        r = [m]
        for key in keys:
            x = g[(g.setting == key) & (g.model == m)].iloc[0]
            r += [f3(x.mean_default), fg(x.gain), fp(x.p_wilcoxon_holm), fp(x.p_ttest_corrected)]
        rows.append(r)
    header = ["& \\multicolumn{4}{c}{Complet, binaire} & \\multicolumn{4}{c}{Complet, 4 classes} & \\multicolumn{4}{c}{Propre, binaire} \\\\"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13}",
              "Modèle & AUC$_0$ & $\\Delta$ & $p_W$ & $p_{NB}$ & AUC$_0$ & $\\Delta$ & $p_W$ & $p_{NB}$ & AUC$_0$ & $\\Delta$ & $p_W$ & $p_{NB}$"]
    write("tab_gain", tabular("l" + "cccc" * 3, header, rows, midrules=[5]))


def tab_friedman() -> None:
    """One table: Friedman + Conover-Holm for the 6 base classifiers (RF vs others) and for the weighted models."""
    f = pd.read_csv(R / "stats_friedman.csv")
    keys = ["full-binary-ALL", "full-multi-ALL", "full-binary-MDP", "clean-binary-ALL"]
    rows = []
    for key in keys:
        for variant, vlab in (("default", "défaut"), ("tuned", "optimisé")):
            r = [SET_SHORT[key] if variant == "default" else "", vlab]
            g = f[(f.setting == key) & (f.variant == variant) & (f.group == "base6")]
            top = g[g.other == "(all)"].iloc[0]
            post = g[g.other != "(all)"].set_index("other")["p_conover"]
            r += [f"{top.chi2:.1f}".replace(".", ","), fp(top.p_friedman)] + [fp(post[m]) for m in ["LR", "KNN", "SVM", "NB", "ANN"]]
            w = f[(f.setting == key) & (f.variant == variant) & (f.group == "weighted")]
            wt = w[w.other == "(all)"].iloc[0]
            wp = w[w.other != "(all)"].set_index("other")["p_conover"]
            r += [f"{wt.chi2:.1f}".replace(".", ","), fp(wt.p_friedman), fp(wp["WLR"]), fp(wp["RF-SMOTE"])]
            rows.append(r)
    header = [r"& & \multicolumn{7}{c}{Six classifieurs de base (RF contre\,\dots)} & "
              r"\multicolumn{4}{c}{Modèles pondérés (WRF contre\,\dots)} \\"
              r"\cmidrule(lr){3-9}\cmidrule(lr){10-13}",
              r"Cadre & Variante & $\chi^2$ & $p_F$ & LR & KNN & SVM & NB & ANN & $\chi^2$ & $p_F$ & WLR & RF-SMOTE"]
    write("tab_friedman", tabular("llccccccccccc", header, rows, midrules=[1, 3, 5]))


def tab_leakage() -> None:
    a = pd.read_csv(R / "leakage_audit_auc.csv")
    labels = {"random_all": "Aléatoire ($n=2621$)",
              "group_triplet_all": "Groupé par triplet",
              "one_row_per_triplet": "1 ligne/triplet ($n=1571$)",
              "clean_subset": "Sous-ens. propre ($n=228$)",
              "shuffled_all": "Permutation ($n=2621$)",
              "shuffled_clean": "Permutation ($n=228$)"}
    rows = []
    for p, lab in labels.items():
        r = [lab]
        for fset in ("MDP", "ALL"):
            v = a[(a.protocol == p) & (a.feature_set == fset)].auc
            r.append(f"{f3(v.mean())} $\\pm$ {f3(v.std())}")
        rows.append(r)
    write("tab_leakage", tabular("lcc", ["Protocole & Biomarq. seuls & Toutes var."], rows, midrules=[3]))


def tab_bootstrap(hold: pd.DataFrame) -> None:
    b = pd.read_csv(R / "holdout_bootstrap_ci.csv")
    rows = []
    for key in [C.setting_key(*s) for s in C.SETTINGS]:
        r = [SET_SHORT[key]]
        for m in ("RF", "WRF"):
            s = b[(b.setting == key) & (b.model == m)]
            for _, x in s.iterrows():
                r.append(f"{f3(x.auc)} [{f3(x.ci_low)} ; {f3(x.ci_high)}]")
        rows.append(r)
    header = ["Cadre & RF défaut & RF optimisé & WRF défaut & WRF optimisé"]
    write("tab_bootstrap", tabular("lcccc", header, rows))


def tab_grids() -> None:
    def fmt_vals(v) -> str:
        s = ", ".join("None" if x is None else (f"{x:g}" if isinstance(x, float) else str(x)) for x in v)
        return "\\{" + s.replace("_", "\\_") + "\\}"

    def fmt_dist(v) -> str:
        if isinstance(v, list):
            return fmt_vals(v)
        a = getattr(v, "args", None)
        name = type(getattr(v, "dist", v)).__name__
        if hasattr(v, "dist"):
            name = v.dist.name
            a = v.args
        return f"{name}({', '.join(f'{x:g}' for x in a)})"

    rows = []
    for m in M.MODEL_NAMES:
        g = M.GRIDS[m]
        sub = g[0] if isinstance(g, list) else g
        grid = "; ".join(f"{k.split('__')[1].replace('_', chr(92) + '_')}={fmt_vals(v)}" for k, v in sub.items())
        d = M.DISTRIBUTIONS[m]
        dist = "; ".join(f"{k.split('__')[1].replace('_', chr(92) + '_')}={fmt_dist(v)}" for k, v in d.items())
        rows.append([m, str(M.grid_size(m)), f"\\scriptsize {grid}", f"\\scriptsize {dist}"])
    write("tab_grids", tabular("lrp{6.4cm}p{6.4cm}", ["Modèle & $|\\mathcal G|$ & Grille (GridSearchCV) & Distributions (RandomizedSearchCV)"], rows))


def tab_importance() -> None:
    p = pd.read_csv(R / "permutation_importance.csv")
    names = {"Humanin": "Humanin", "MOTSc": "MOTS-c", "P66": "P66", "ACE-Update": "Haplotype ACE",
             "OtherMeds-Depression": "Antidépresseurs", "Gender": "Sexe", "BirthAge": "Âge", "DM-Status": "Diabète",
             "DM-MedUse": "Médic. diabète", "CVD-Revised": "Mal. cardiovasc.", "HT-Status": "Hypertension",
             "HT-MedUse": "Médic. hypertension"}
    keys = ["full-binary-ALL", "full-multi-ALL", "clean-binary-ALL"]
    rows = []
    order = p[p.setting == "full-binary-ALL"].sort_values("importance_mean", ascending=False).feature
    for f in order:
        r = [names[f]]
        for key in keys:
            x = p[(p.setting == key) & (p.feature == f)].iloc[0]
            r.append(f"{f3(x.importance_mean)} $\\pm$ {f3(x.importance_std)}")
        rows.append(r)
    write("tab_importance", tabular("lccc", ["Variable & Complet, binaire & Complet, 4 cl. & Propre, binaire"], rows))


def tab_annex_holdout(hold: pd.DataFrame) -> None:
    """All models x methods for the biomarker-only settings (article Tables 3 and 7 analogue)."""
    keys = ["full-binary-MDP", "full-multi-MDP", "clean-binary-MDP"]
    rows = []
    for m in M.MODEL_NAMES:
        r = [m]
        for key in keys:
            vals = [hold[(hold.setting == key) & (hold.model == m) & (hold.method == meth)].auc.iloc[0]
                    for meth in ("default", "grid", "random")]
            r += [f3(v) for v in vals]
        rows.append(r)
    header = ["& \\multicolumn{3}{c}{Complet, binaire} & \\multicolumn{3}{c}{Complet, 4 classes} & \\multicolumn{3}{c}{Propre, binaire} \\\\"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
              "Modèle & Déf. & Grid & Rand. & Déf. & Grid & Rand. & Déf. & Grid & Rand."]
    write("tab_annex_holdout", tabular("l" + "ccc" * 3, header, rows, midrules=[5]))


def tab_featureset() -> None:
    f = pd.read_csv(R / "stats_featureset.csv")
    rows = [[f"{'Complet' if r.regime == 'full' else 'Propre'}, {'binaire' if r.task == 'binary' else '4 classes'}",
             f3(r.mean_MDP), f3(r.mean_ALL), fg(r.gain), fp(r.p_wilcoxon), fp(r.p_ttest_corrected)] for r in f.itertuples()]
    write("tab_featureset", tabular("lccccc", ["Cadre & AUC biomarq. & AUC toutes var. & $\\Delta$ & $p_W$ & $p_{NB}$"], rows))


def main() -> None:
    hold = pd.read_csv(R / "holdout_all_metrics.csv")
    tab_grids()
    tab_baseline(hold)
    tab_before_after(hold)
    tab_weighted(hold)
    tab_time(hold)
    tab_bootstrap(hold)
    tab_leakage()
    tab_annex_holdout(hold)
    if (R / "stats_tuning_gain.csv").exists():
        tab_gain()
        tab_friedman()
        tab_featureset()
    if (R / "permutation_importance.csv").exists():
        tab_importance()
    print("tables written to", T)


if __name__ == "__main__":
    main()

# Depression severity detection with weighted random forest — hyper-parameter optimisation pipeline

Project 14: a complete hyper-parameter optimisation pipeline (`GridSearchCV`, `RandomizedSearchCV`, plus an
Optuna/TPE Bayesian search) with stratified K-fold cross-validation, applied to the problem studied in
*Bader et al., "Detecting depression severity using weighted random forest and oxidative stress biomarkers",
Sci. Rep. 14:16328 (2024)*. The deliverable report (in French) follows `structureETconsignes.pdf` and lives in
`report/rapport.pdf`; everything else (code, comments, docs) is in English.

## What the project does

1. **Reproduces the article's protocol** on the supplied dataset: 70/30 stratified split, standardisation,
   six classifiers (LR, RF, KNN, SVM, NB, ANN), class-weighted variants (WLR, WRF, RF-SMOTE), binary and
   4-class tasks, biomarkers-only vs all features, permutation importance, Friedman + Conover tests.
2. **Improves it** with (A) explicit, comparable hyper-parameter searches (grid / random / Bayesian) and
   before-vs-after measurements incl. run time, (B) an extra weighted gradient-boosting model (WHGB),
   (C) nested cross-validation giving paired scores, bootstrap CIs, (D) paired statistical tests
   (Wilcoxon, Nadeau–Bengio corrected t, Friedman/Conover).
3. **Audits the data for leakage.** The supplied CSV contains many duplicated biomarker values that are
   almost perfectly aligned with the outcome class (see `results/leakage_audit.json`), which inflates
   random-split scores. The project therefore reports two regimes: `full` (all 2,621 rows, article-style)
   and `clean` (228 rows whose three biomarker values are unique; binary task only).

> The CSV's biomarkers are **Humanin, MOTSc, P66** (not the article's GSH/GSSG/8-OHdG set), so the article's
> *protocol* is reproduced, not its numbers.

## Main findings (details and numbers in `report/rapport.pdf`)

| | Result |
|---|---|
| Reproduced | article's protocol and qualitative ranking: RF best, LR/NB weakest, WRF >> WLR, WRF ≳ RF-SMOTE (only at α = 0.1) |
| Test AUC (full regime, all features) | RF 0.928 binary / 0.895 four classes; tuned WRF 0.925–0.928 / 0.915–0.927 (bootstrap 95 % CIs overlap) |
| Effect of tuning (nested CV, paired tests) | significant gains only for KNN, SVM, ANN (+0.04 to +0.12 AUC); none for RF/WRF (−0.007 to +0.009) |
| Grid vs Random vs Optuna | equivalent quality at comparable budget; random search needs fewer fits than the grid |
| Leakage audit | 67–85 % of biomarker values are shared between rows and almost perfectly aligned with the severity class (outcome purity 0.99 vs 0.74 by chance) |
| `clean` regime (228 rows, no shared values) | RF AUC 0.60 (biomarkers) / 0.72 (all features); tuning brings no gain (RF −0.03, p = 0.039) |

Interpretation: the high scores in the `full` regime most likely reflect leaked/duplicated values (a hypothesis —
the file has no documentation of how values were imputed), not a biological signal. Do not use the model clinically.

## Things to complete / verify before submitting

* Author block (name, affiliation, e-mail) at the top of `report/rapport.tex` (`\AuteurNom`, `\AuteurAffil`, `\AuteurMail`).
* The interpretation of the columns `Humanin`, `MOTSc`, `P66` as humanin / MOTS-c / p66Shc is inferred from the names.
* A few bibliography entries were typed from memory (not copied from the article's list); check them against the
  publishers' pages (`report/references.tex`).

## Layout

```
data/                      supplied CSV (copy)
src/                       config, data regimes, model zoo + search spaces, metrics, plotting helpers
scripts/01_eda_and_leakage_audit.py   EDA figures + leakage audit
scripts/02_holdout_experiments.py     default / grid / random / optuna on the 70/30 split
scripts/03_nested_cv.py               nested CV (5x2 outer, randomized-search inner) -> paired scores
scripts/04_statistical_tests.py       Wilcoxon, t, Nadeau-Bengio, Friedman + Conover
scripts/05_diagnostics_and_figures.py confusion matrices, ROC, learning curves, importance, bootstrap CIs
scripts/06_make_report_tables.py      LaTeX tables generated from results/*.csv
results/                   all numeric outputs (CSV / JSON / joblib) and run logs
figures/                   figures used in the report (PDF + PNG)
report/                    French report (LaTeX): rapport.tex, sections/, tables/, references.tex
run_all.py                 runs the whole pipeline
```

## Reproduce

```bash
pip install -r requirements.txt     # exact versions used (Python 3.13.2)
python run_all.py                   # ~1-2 h on a 12-thread CPU; or run the scripts one by one
cd report && pdflatex rapport.tex && pdflatex rapport.tex
```

All random seeds are fixed to 42 (splits, estimators, searches, Optuna). Pipelines fit the scaler and SMOTE
inside each CV fold. Quick smoke test: `python scripts/02_holdout_experiments.py --quick`.

## Protocol summary

| Item | Setting |
|---|---|
| Hold-out split | 70 / 30, stratified, seed 42 (test set used once per model) |
| Search CV | stratified 5-fold on the training part; objective = ROC-AUC (macro OvR for 4 classes) |
| Grid / Random / Optuna | full grid (see `src/models.py`) / 30 draws / 30 TPE trials (WRF only) |
| Nested CV | outer 5 folds × 2 repeats; inner randomized search, 15 candidates, 5-fold |
| Metrics | accuracy, weighted precision/recall/F1, macro-F1, balanced accuracy, ROC-AUC, PR-AUC |
| Tests | paired t, Nadeau–Bengio corrected t, Wilcoxon (Holm), Friedman + Conover |

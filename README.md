# Fraud Detection — End-to-End Classification Project

An end-to-end machine learning project that detects fraudulent financial
transactions, built as a hands-on foundation project in data science and
MLOps practices (experiment tracking, reproducible pipelines, Git workflow).

## Problem statement

Financial institutions process millions of transactions, and only a tiny
fraction are fraudulent (typically <5%). The goal is to build a classifier
that flags likely fraud while keeping false alarms low enough to be usable
in practice — a classic **imbalanced classification** problem where accuracy
is a misleading metric and the real tradeoff is *missed fraud vs. false
alarms*.

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
(Kaggle competition dataset) — real-world e-commerce transaction data with
transaction and identity tables, ~590k transactions, realistic missing data
and categorical features.

**To download it yourself:**
1. Create a free [Kaggle](https://www.kaggle.com/) account.
2. Go to `Account → Create New API Token` to download `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` (never commit this file — it's already
   gitignored here).
4. `kaggle competitions download -c ieee-fraud-detection -p data/raw/`
5. Unzip into `data/raw/` (already gitignored — raw data is never committed).

## Project roadmap

Each notebook below is both a project stage and a concept lesson, worked
through and committed one at a time:

- [x] `01_eda.ipynb` — data loading, schema validation, class imbalance,
      missing data, distributions, correlations
- [x] `02_statistical_analysis.ipynb` — hypothesis testing to justify which
      features actually relate to fraud
- [x] `03_feature_engineering.ipynb` — time-ordered split with a 30-day gap,
      entity (uid) construction, encoding, drift audit — all fitted on train only
- [ ] `04_imbalanced_learning.ipynb` — class weights, SMOTE/undersampling,
      threshold moving
- [ ] `05_baseline_modeling.ipynb` — logistic regression baseline →
      XGBoost/LightGBM, tracked with MLflow
- [ ] `06_model_evaluation.ipynb` — PR-AUC, ROC-AUC, confusion matrix,
      business-cost-aware threshold tuning
- [ ] `07_model_interpretability.ipynb` — SHAP values / feature importance
- [ ] `08_hyperparameter_tuning.ipynb` — Optuna/GridSearch with a CV
      strategy appropriate for imbalanced, time-ordered data
- [ ] `tests/` — unit tests (no data leakage, expected schema)
- [ ] Results write-up in this README

**Stretch goals (v2):** Dockerize the pipeline, serve the model behind a
FastAPI endpoint, add GitHub Actions CI to run tests on every push.

## Project structure

```
├── data/
│   ├── raw/          # gitignored — original Kaggle download
│   └── processed/    # gitignored — cleaned/feature-engineered data
├── notebooks/         # numbered, one per pipeline stage (see roadmap)
├── src/                # production-ready modules, promoted from notebooks
├── tests/              # unit tests
└── requirements.txt
```

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

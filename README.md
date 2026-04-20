# Week 13: ML Comparison Demo

Teaching demo for **QM 2023 — Statistics II: Data Analytics**, Week 13 (Machine Learning).

Covers the bias-variance tradeoff and a 4-model comparison (OLS vs. Ridge vs. Lasso vs. Random Forest) using real REIT factor return data.

---

## Prerequisites

- Python 3.10+
- pip

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Notebook

Open in VS Code or Jupyter and run cells top to bottom:

```bash
jupyter notebook ml_comparison_demo.ipynb
```

Or in VS Code: open the file and click **Run All**.

---

## What It Covers

- **Inference vs. prediction** — why OLS and ML answer different questions
- **Train/test split** — how to detect data leakage and measure true generalization
- **OLS baseline** — coefficient table, R² train vs. test
- **Ridge and Lasso** — how regularization works, cross-validated alpha selection, Lasso feature selection
- **Random Forest** — fitting 100 trees, diagnosing the overfitting signal visually
- **Feature importance comparison** — do all four models agree on the most important predictors?
- **Investment decision framework** — when OLS beats ML for real-world use

---

## Main Dependencies

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — all visualizations
- `scikit-learn` — OLS, Ridge, Lasso, Random Forest, GridSearchCV, StandardScaler
- `jupyter` — notebook execution

---

## Data

Uses `Data/factors_master_long_only.csv` from the repo root — 470 monthly observations (December 1986 – January 2026).

The notebook searches upward from its location to find the `Data/` folder automatically. No path configuration required if you run from inside the repo.

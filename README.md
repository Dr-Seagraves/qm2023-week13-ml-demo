# Week 13: ML Comparison Demo

Teaching demo for **QM 2023 — Statistics II: Data Analytics**, Week 13 (Machine Learning).

Covers a beginner-friendly introduction to machine learning and a 4-model comparison (OLS vs. Ridge vs. Lasso vs. Random Forest) using the **monthly REIT panel**.

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

- **What machine learning is** — plain-language walkthrough for students seeing ML for the first time
- **Prediction vs. explanation** — why ML is judged by held-out forecasting accuracy, not p-values
- **Target choice matters** — why next-month REIT returns are hard to predict but next-month volatility is more learnable
- **Time-based train/test split** — first 80% of months for training, last 20% for testing
- **Four-model comparison** — OLS, Ridge, Lasso, and Random Forest on the same monthly REIT prediction task
- **Interpretation checkpoints** — what each chart means and why more complex models do not automatically win
- **Saved figures** — every major visual is exported to `demos/figures/` for reuse in slides or notes

---

## Main Dependencies

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — all visualizations
- `seaborn` — styling and higher-level charts
- `scikit-learn` — OLS, Ridge, Lasso, Random Forest, GridSearchCV, StandardScaler
- `jupyter` — notebook execution

---

## Data

Uses `demos/data/REIT_sample_2004_2024.csv` — a local copy of the monthly REIT panel with firm characteristics, returns, and risk measures.

The notebook now prefers the local `demos/data/` copy and falls back to the repo-level `Data/` folder if needed. No path configuration required if you run from inside the repo.

## Output Figures

Running the notebook saves intuitive visuals to `demos/figures/`, including:

- ML workflow diagram
- return-vs-volatility signal comparison
- monthly REIT market backdrop charts
- feature correlation heatmap
- model comparison chart
- actual-vs-predicted chart for the best model
- feature-importance chart

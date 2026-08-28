# Credit Scoring — Home Credit Default Risk

A machine-learning pipeline that predicts the probability of a client defaulting on a loan, built on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle dataset.

---

## Username

```
bdaanoun_01EDU_oujda_08_2026
cbenlafk_01EDU_oujda_08_2026
```

---

## Project Overview

The project covers the full ML lifecycle:

| Stage | Description |
|---|---|
| **Feature Engineering** | Merges the main application table with bureau, previous application, and instalment data; builds aggregate features and handles missing values |
| **Preprocessing** | Splits data into train/test sets, imputes remaining nulls, and serialises processed CSVs |
| **Training** | Trains an XGBoost binary classifier with early stopping; saves the model + preprocessor as a `.pkl` file |
| **Prediction** | Evaluates the model (ROC-AUC) on the held-out test set and generates a Kaggle submission CSV |
| **Explainability** | Produces SHAP-based per-client explanations and global feature-importance plots |

---

## Project Structure

```
credit-scoring/
├── data/                        # Raw and processed data (not tracked by git)
├── feature_engineering/         # Dataset assembly and EDA notebooks
│   ├── build_dataset.py
│   ├── application_train.py
│   ├── bureau.py
│   ├── previous_application.py
│   ├── installments.py
│   └── EDA_*.ipynb
├── scripts/                     # Core pipeline scripts
│   ├── preprocess.py            # Step 1 – preprocess & split
│   ├── train.py                 # Step 2 – train model
│   ├── predict.py               # Step 3 – evaluate & generate submission
│   ├── one_client.py            # SHAP explanation for a single client
│   ├── explain_all_clients.py   # SHAP explanations for all clients
│   ├── feature_importance.py    # Global feature importance plot
│   ├── generate_L_curves.py     # Learning curves
│   ├── kaggle_predict.py        # Kaggle test-set predictions
│   ├── helpers.py               # Shared utilities
│   └── save_to_pdf.py           # Export reports to PDF
├── dashboard/                   # (WIP) Interactive dashboard
├── results/                     # Model artefacts and predictions (not tracked)
├── requirements.txt
└── readme.md
```

---

## Requirements

- Python 3.8+
- Install dependencies:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
numpy
pandas
scikit_learn
matplotlib
xgboost==2.1.4
shap==0.49.1
```

---

## How to Run

All scripts must be executed from the `scripts/` directory.

### 1 — Preprocess the data

Builds the full feature dataset and saves processed train/test CSVs to `data/`.

```bash
cd scripts
python preprocess.py
```

### 2 — Train the model

Trains the XGBoost classifier and saves the model to `results/model/xgboost.pkl`.

```bash
python train.py
```

### 3 — Evaluate and generate predictions

Prints ROC-AUC on the test set and writes `results/prediction.csv` (Kaggle submission format).

```bash
python predict.py
```

### 4 — (Optional) Explain a single client

```bash
python one_client.py
```

### 5 — (Optional) Explain all clients

```bash
python explain_all_clients.py
```

---

## Results

The trained model outputs:

- `results/model/xgboost.pkl` — serialised model + preprocessor
- `results/model/feature_importance.png` — top features by gain
- `results/model/learning_curve.png` — train vs. validation AUC over boosting rounds
- `results/prediction.csv` — Kaggle submission file

---

## Data

Download the raw CSVs from the [Kaggle competition page](https://www.kaggle.com/c/home-credit-default-risk/data) and place them in the `data/` directory before running the pipeline.

Key files expected:
- `data/application_train.csv`
- `data/application_test.csv`
- `data/bureau.csv`
- `data/previous_application.csv`
- `data/installments_payments.csv`

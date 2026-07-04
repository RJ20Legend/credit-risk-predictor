# Credit Risk / Loan Default Prediction

End-to-end ML project predicting borrower default probability using the [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) dataset.

## Stack
- **ML**: XGBoost, LightGBM, Logistic Regression
- **Imbalance**: `scale_pos_weight`, class weights
- **API**: FastAPI + Pydantic
- **Frontend**: Vanilla HTML/JS
- **Deployment**: Render / Railway (free tier)

## Project Structure
```
loan/
├── data/               # CSVs (gitignored)
├── src/
│   ├── config.py       # all paths, constants, hyperparams
│   ├── preprocess.py   # cleaning + imputation
│   ├── features.py     # feature engineering (13 features)
│   ├── train.py        # model training + evaluation
│   └── predict.py      # inference pipeline (used by API)
├── api/
│   ├── main.py         # FastAPI app
│   └── schema.py       # Pydantic request/response models
├── frontend/
│   └── index.html      # single-page UI
└── models/             # saved artifacts (gitignored)
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (from project root)
python -m src.train

# 3. Start the API
uvicorn api.main:app --reload

# 4. Open frontend
# Open frontend/index.html in your browser
```

## Key ML Decisions
- **Imbalanced data** (~6.7% default rate): handled via `scale_pos_weight` for tree models, `class_weight='balanced'` for logistic regression
- **Feature engineering**: 13 engineered features including `DelinquencyScore`, `EverDelinquent`, `UtilizationSquared`, `DisposableIncome`, `YoungHighUtil`
- **Evaluation metric**: ROC-AUC and PR-AUC (PR-AUC is more informative for imbalanced classes)
- **Imputation**: MonthlyIncome imputed by median grouped on `NumberOfOpenCreditLinesAndLoans` (not flat median)

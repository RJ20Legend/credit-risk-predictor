"""
main.py — FastAPI application.

Start with:
    uvicorn api.main:app --reload

Endpoints:
    GET  /         → health check
    POST /predict  → score a single borrower
    GET  /docs     → auto-generated Swagger UI
"""

from fastapi            import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schema         import LoanRequest, LoanResponse
from src.predict        import load_artifacts, predict

# ── Load model artifacts once at startup ────────────────────────────────
try:
    MODEL, FEATURE_NAMES, STATE = load_artifacts()
except FileNotFoundError:
    raise RuntimeError(
        "Model artifacts not found. Run `python src/train.py` first."
    )

# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Risk API",
    description="Predicts the probability that a borrower will default within 2 years.",
    version="1.0.0",
)

# Allow the HTML frontend (served locally or on any domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Credit Risk API is running."}


@app.post("/predict", response_model=LoanResponse, tags=["Prediction"])
def predict_default(request: LoanRequest):
    """
    Score a borrower and return default probability.

    - **probability**: float between 0 and 1
    - **prediction**: 1 = likely default, 0 = likely no default
    - **risk_label**: "High Risk" or "Low Risk"
    """
    try:
        # Convert Pydantic model to dict, handling field aliases
        input_data = {
            "RevolvingUtilizationOfUnsecuredLines": request.RevolvingUtilizationOfUnsecuredLines,
            "age":                                  request.age,
            "NumberOfTime30-59DaysPastDueNotWorse": request.NumberOfTime30_59DaysPastDueNotWorse,
            "DebtRatio":                            request.DebtRatio,
            "MonthlyIncome":                        request.MonthlyIncome,
            "NumberOfOpenCreditLinesAndLoans":      request.NumberOfOpenCreditLinesAndLoans,
            "NumberOfTimes90DaysLate":              request.NumberOfTimes90DaysLate,
            "NumberRealEstateLoansOrLines":         request.NumberRealEstateLoansOrLines,
            "NumberOfTime60-89DaysPastDueNotWorse": request.NumberOfTime60_89DaysPastDueNotWorse,
            "NumberOfDependents":                   request.NumberOfDependents,
        }

        result = predict(input_data, MODEL, FEATURE_NAMES, STATE)
        return LoanResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

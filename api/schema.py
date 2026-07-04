"""
schema.py — Pydantic models for request validation and response shape.

Keeping this separate from main.py means the API logic stays clean
and these models can be reused/tested independently.
"""

from pydantic import BaseModel, Field, field_validator


class LoanRequest(BaseModel):
    """Raw borrower features — exactly what the frontend form sends."""

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0, le=1,
        description="Total balance on credit cards / personal lines of credit divided by sum of credit limits. Between 0 and 1.",
        example=0.45,
    )
    age: int = Field(
        ..., ge=18, le=109,
        description="Age of borrower in years.",
        example=45,
    )
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        ..., ge=0,
        description="Number of times borrower has been 30-59 days past due but no worse.",
        alias="NumberOfTime30-59DaysPastDueNotWorse",
        example=0,
    )
    DebtRatio: float = Field(
        ..., ge=0,
        description="Monthly debt payments / monthly gross income.",
        example=0.35,
    )
    MonthlyIncome: float = Field(
        ..., ge=0,
        description="Monthly income in USD.",
        example=5000,
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(
        ..., ge=0,
        description="Number of open loans and lines of credit.",
        example=8,
    )
    NumberOfTimes90DaysLate: int = Field(
        ..., ge=0,
        description="Number of times borrower has been 90 days or more past due.",
        example=0,
    )
    NumberRealEstateLoansOrLines: int = Field(
        ..., ge=0,
        description="Number of mortgage and real estate loans.",
        example=1,
    )
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        ..., ge=0,
        description="Number of times borrower has been 60-89 days past due but no worse.",
        alias="NumberOfTime60-89DaysPastDueNotWorse",
        example=0,
    )
    NumberOfDependents: float = Field(
        ..., ge=0,
        description="Number of dependents in family (spouse, children etc.).",
        example=1,
    )

    model_config = {"populate_by_name": True}


class LoanResponse(BaseModel):
    """What the API returns after scoring a request."""
    probability: float   = Field(..., description="Predicted probability of default (0-1).")
    prediction:  int     = Field(..., description="Binary prediction: 1 = default, 0 = no default.")
    risk_label:  str     = Field(..., description="Human-readable risk label.")

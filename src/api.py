from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="DCA AI Management API",
    description="AI-powered Debt Collection Agency Management using XGBoost",
    version="1.0.0"
)

# Load the trained XGBoost model
try:
    model = joblib.load('models/dca_recovery_model.pkl')
    model_loaded = True
except FileNotFoundError:
    model = None
    model_loaded = False
except Exception as e:
    model = None
    model_loaded = False

class AccountData(BaseModel):
    amount_overdue: float = Field(..., gt=0, description="Amount overdue in dollars")
    days_overdue: int = Field(..., gt=0, le=1000, description="Days since payment was due")
    customer_age: int = Field(..., ge=18, le=100, description="Customer age in years")
    payment_history_score: float = Field(..., ge=0, le=1, description="Payment history score (0-1)")
    contact_attempts: int = Field(..., ge=0, le=50, description="Number of contact attempts made")

    @field_validator('days_overdue')
    def validate_days_overdue(cls, v):
        if v == 0:
            raise ValueError('days_overdue cannot be zero')
        return v

class BatchAccountData(BaseModel):
    accounts: List[AccountData] = Field(..., max_length=100, description="List of accounts to predict")

@app.get("/")
async def root():
    return {
        "message": "DCA AI Management API",
        "status": "running",
        "model_loaded": model_loaded,
        "model_type": "XGBoost" if model_loaded else "None"
    }

@app.get("/model_info")
async def model_info():
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    return {
        "model_type": "XGBoost",
        "features": [
            "amount_overdue",
            "days_overdue",
            "customer_age",
            "payment_history_score",
            "contact_attempts",
            "amount_per_day",
            "urgency_score"
        ],
        "target": "recovery_probability",
        "performance": {
            "f1_score": 0.64,
            "accuracy": 0.64
        }
    }

def prepare_features(account: AccountData):
    """Prepare features for prediction."""
    amount_per_day = account.amount_overdue / account.days_overdue
    urgency_score = account.days_overdue * account.amount_overdue / 1000

    return np.array([[account.amount_overdue, account.days_overdue, account.customer_age,
                     account.payment_history_score, account.contact_attempts, amount_per_day, urgency_score]])

@app.post("/predict_recovery")
async def predict_recovery(account: AccountData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        features = prepare_features(account)

        # Predict probability
        probability = model.predict_proba(features)[0][1]

        # Predict class
        prediction = model.predict(features)[0]

        # Calculate risk level
        risk_level = "High" if probability < 0.3 else "Medium" if probability < 0.7 else "Low"

        return {
            "recovery_probability": float(probability),
            "predicted_recovery": bool(prediction),
            "risk_level": risk_level,
            "prioritization_score": float(features[0][6]),
            "recommendation": "High priority collection" if risk_level == "High" else
                            "Standard collection process" if risk_level == "Medium" else
                            "Low priority monitoring"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(batch_data: BatchAccountData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        results = []
        for account in batch_data.accounts:
            features = prepare_features(account)
            probability = model.predict_proba(features)[0][1]
            prediction = model.predict(features)[0]
            risk_level = "High" if probability < 0.3 else "Medium" if probability < 0.7 else "Low"

            results.append({
                "account_data": account.dict(),
                "recovery_probability": float(probability),
                "predicted_recovery": bool(prediction),
                "risk_level": risk_level,
                "prioritization_score": float(features[0][6])
            })

        # Sort by prioritization score (highest first)
        results.sort(key=lambda x: x["prioritization_score"], reverse=True)

        return {
            "total_accounts": len(results),
            "predictions": results,
            "summary": {
                "high_risk_count": sum(1 for r in results if r["risk_level"] == "High"),
                "medium_risk_count": sum(1 for r in results if r["risk_level"] == "Medium"),
                "low_risk_count": sum(1 for r in results if r["risk_level"] == "Low"),
                "expected_recoveries": sum(1 for r in results if r["predicted_recovery"])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
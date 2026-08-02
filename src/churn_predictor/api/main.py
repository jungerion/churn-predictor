"""FastAPI app serving the churn model.

Run with:  uv run uvicorn churn_predictor.api.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from churn_predictor.models.predict import predict_one
from churn_predictor.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Churn Predictor API",
    description="Predicts whether a telecom customer will churn.",
    version="0.1.0",
)


class CustomerFeatures(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(examples=[0])
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(examples=[12])
    PhoneService: str = Field(examples=["Yes"])
    MultipleLines: str = Field(examples=["No"])
    InternetService: str = Field(examples=["Fiber optic"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["Yes"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["Yes"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(examples=[70.35])
    TotalCharges: float = Field(examples=[845.5])


class PredictionResponse(BaseModel):
    churn_prediction: bool
    churn_probability: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        result = predict_one(customer.model_dump())
        return result
    except FileNotFoundError as e:
        logger.error(str(e))
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
from fastapi import FastAPI

from app.schemas import LoanRequest, PredictionResponse
from app.predict import predict_loan

app = FastAPI(
    title="RISKLOCK API",
    description="Governed Loan Risk Prediction Service",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: LoanRequest):
    # Use Pydantic v2 API to avoid deprecation warnings
    return predict_loan(request.model_dump())

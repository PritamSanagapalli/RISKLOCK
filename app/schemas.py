from pydantic import BaseModel, ConfigDict, Field
from typing import Union


class LoanRequest(BaseModel):
    loan_amnt: float = Field(..., gt=0)
    annual_inc: float = Field(..., gt=0)
    int_rate: Union[str, float]
    emp_length: Union[str, float]
    dti: float = Field(..., ge=0)
    earliest_cr_line: str
    term: str
    home_ownership: str


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    probability_default: float
    risk_level: str
    top_risk_factors: list[str]
    model_version: str
    data_quality_score: float | None

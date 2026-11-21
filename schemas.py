from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    student_id: Optional[str] = Field(default=None)
    features: Union[Dict[str, Any], List[float]] = Field(...)


class StressPrediction(BaseModel):
    label: str
    level: int
    message: str
    confidence: float
    probabilities: Dict[str, float]


class TransformedMetric(BaseModel):
    name: str
    form_field: str
    raw_value: Any
    processed_value: float
    normalized_value: float
    empuje: Optional[float] = None
    is_top_factor: bool = False
    factor_text: Optional[str] = None
    status: str


class PredictionResponse(BaseModel):
    student_id: Optional[str] = None
    source: str
    model_used: str
    prediction: StressPrediction
    top_factors: List[TransformedMetric]
    transformed_data: List[TransformedMetric]
    summary_vector: List[float]
    disclaimer: str
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
import time

from src.models.predictor import Predictor
from src.api.logger import APILogger
from src.features.feature_pipeline import FeaturePipeline
import joblib

app = FastAPI()

# Load artifacts
pipeline = joblib.load("feature_pipeline.pkl")
predictor = Predictor("model.pkl")

class ShopperInput(BaseModel):
    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    VisitorType: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    Weekend: bool


@app.post("/predict")
def predict_api(data: ShopperInput):
    # Convert input to dict for logging
    input_data = data.dict()

    # Convert to df
    df = pd.DataFrame([input_data])

    start = time.time()

    # Apply feature pipeline
    processed = pipeline.transform(df)

    # Model prediction
    pred, prob = predictor.predict(processed)

    latency = (time.time() - start) * 1000

    # Log request-response
    APILogger.log_request_response(
        input_data=input_data,
        output_data={"prediction": int(pred[0]), "probability": float(prob[0])},
        latency_ms=latency
    )

    return {
        "prediction": int(pred[0]),
        "probability": float(prob[0]),
        "latency_ms": latency
    }

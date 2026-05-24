"""
app.py
------
FastAPI real-time inference endpoint for Portugal Housing Price Prediction.

Run locally:
    uvicorn app:app --reload --port 8080

Test:
    curl -X POST http://localhost:8080/predict \\
      -H "Content-Type: application/json" \\
      -d '{"total_area": 100, "parking": 1, "construction_year": 2010,
           "total_rooms": 4, "living_area": 80, "number_of_bathrooms": 2,
           "district": "Lisboa", "city": "Lisboa", "town": "Arroios",
           "type": "Apartment", "energy_certificate": "B", "elevator": true}'
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Portugal Housing Price Prediction API",
    description="Predict the asking price of a Portuguese real estate listing.",
    version="1.0.0",
)

# CORS middleware — allows browser-based clients to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
MODEL_PATH = Path(__file__).parent / "model.pkl"
model = joblib.load(MODEL_PATH)


class PropertyFeatures(BaseModel):
    total_area: float = 100.0
    parking: float = 1.0
    construction_year: float = 2010.0
    total_rooms: float = 4.0
    living_area: float = 80.0
    number_of_bathrooms: float = 2.0
    district: str = "Lisboa"
    city: str = "Lisboa"
    town: str = "Arroios"
    type: str = "Apartment"
    energy_certificate: str = "B"
    elevator: bool = True


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "EUR"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PropertyFeatures):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    return PredictionResponse(
        predicted_price=round(float(prediction), 2),
        currency="EUR",
    )

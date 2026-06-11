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

import math
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

app = FastAPI(
    title="Portugal Housing Price Prediction API",
    description="Predict the asking price of a Portuguese real estate listing.",
    version="1.1.0",
)

# CORS is opt-in, never a wildcard: browsers only need it for cross-origin
# JavaScript calls, which this API does not serve by default. To enable it for
# a known frontend, set ALLOWED_ORIGINS to a comma-separated origin list.
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if _origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

# Load model on startup
MODEL_PATH = Path(__file__).parent / "model.pkl"
model = joblib.load(MODEL_PATH)

# Closed vocabularies from the training data. Unknown districts or certificate
# levels are rejected here; city and town stay free-text because the encoder
# pools unseen values into its infrequent bucket safely.
DISTRICTS = (
    "Aveiro", "Beja", "Braga", "Bragança", "Castelo Branco", "Coimbra",
    "Évora", "Faro", "Guarda", "Ilha Terceira", "Ilha da Madeira",
    "Ilha das Flores", "Ilha de Porto Santo", "Ilha de Santa Maria",
    "Ilha de São Miguel", "Ilha do Faial", "Leiria", "Lisboa", "Portalegre",
    "Porto", "Santarém", "Setúbal", "Viana do Castelo", "Vila Real", "Viseu",
    "Z - Fora de Portugal",
)
PROPERTY_TYPES = ("Apartment", "House", "Duplex", "Studio", "Mansion", "Manor")
ENERGY_CERTIFICATES = ("A+", "A", "B", "B-", "C", "D", "E", "F", "G", "NC",
                       "No Certificate", "Not available")

_CURRENT_YEAR = datetime.now().year


class PropertyFeatures(BaseModel):
    """One residential listing. Bounds mirror the training-data sanity checks."""

    total_area: float = Field(100.0, ge=16, le=100_000,
                              description="Total area in m2 (>= 16, the minimum dwelling size)")
    living_area: float = Field(80.0, ge=0, le=100_000,
                               description="Livable area in m2; cannot exceed total_area")
    parking: float = Field(1.0, ge=0, le=50)
    construction_year: float = Field(2010.0, ge=1500, le=_CURRENT_YEAR + 5)
    total_rooms: float = Field(4.0, ge=0, le=100)
    number_of_bathrooms: float = Field(2.0, ge=0, le=50)
    district: Literal[DISTRICTS] = "Lisboa"
    city: str = Field("Lisboa", min_length=1, max_length=100)
    town: str = Field("", max_length=120)
    type: Literal[PROPERTY_TYPES] = "Apartment"
    energy_certificate: Literal[ENERGY_CERTIFICATES] = "B"
    elevator: bool = True

    @field_validator("total_area", "living_area", "parking", "construction_year",
                     "total_rooms", "number_of_bathrooms")
    @classmethod
    def reject_non_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("value must be a finite number")
        return v

    @model_validator(mode="after")
    def living_within_total(self):
        if self.living_area > self.total_area:
            raise ValueError("living_area cannot exceed total_area")
        return self


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "EUR"


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PropertyFeatures):
    input_df = pd.DataFrame([data.model_dump()])
    try:
        prediction = float(model.predict(input_df)[0])
    except Exception as exc:  # surface model failures as a clean API error
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    if not math.isfinite(prediction) or prediction <= 0:
        raise HTTPException(status_code=500, detail="model returned an invalid price")
    return PredictionResponse(predicted_price=round(prediction, 2), currency="EUR")

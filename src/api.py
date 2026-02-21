"""
FastAPI prediction service for the Grid Risk Platform.

Usage
-----
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Endpoints
    POST /predict    — score a single outage record
    GET  /health     — liveness check
    GET  /model-info — model version and feature list
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import ARTIFACTS_DIR, FEATURE_NAMES_FILE, METRICS_FILE, MODEL_VERSION
from src.predict import GridRiskPredictor

logger = logging.getLogger(__name__)

_predictor: Optional[GridRiskPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    try:
        _predictor = GridRiskPredictor()
        logger.info("Model loaded — version %s", MODEL_VERSION)
    except FileNotFoundError as e:
        logger.error("Artifacts missing — run training pipeline first: %s", e)
    yield
    _predictor = None


app = FastAPI(
    title="Grid Risk & Reliability API",
    version=MODEL_VERSION,
    description="Predict probability of high-impact power outage events.",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────

class OutageInput(BaseModel):
    """Input features for a single outage risk prediction.

    All fields are optional; the preprocessor handles missing values.
    Provide as many as available for best accuracy.
    """
    ANOMALY_LEVEL: Optional[float] = Field(None, description="Oceanic Niño Index anomaly level")
    DEMAND_LOSS_MW: Optional[float] = Field(None, ge=0, description="Demand loss in MW")
    RES_PRICE: Optional[float] = Field(None, ge=0, description="Residential electricity price (cents/kWh)")
    COM_PRICE: Optional[float] = Field(None, ge=0, description="Commercial electricity price")
    IND_PRICE: Optional[float] = Field(None, ge=0, description="Industrial electricity price")
    TOTAL_PRICE: Optional[float] = Field(None, ge=0, description="Average total electricity price")
    TOTAL_SALES: Optional[float] = Field(None, ge=0, description="Total electricity sales (MWh)")
    TOTAL_CUSTOMERS: Optional[float] = Field(None, ge=0, description="Total utility customers")
    PC_REALGSP_STATE: Optional[float] = Field(None, description="Per capita real GSP of the state")
    PC_REALGSP_REL: Optional[float] = Field(None, description="Relative per capita real GSP")
    PC_REALGSP_CHANGE: Optional[float] = Field(None, description="% change in per capita real GSP")
    UTIL_REALGSP: Optional[float] = Field(None, description="Real GSP contributed by utility sector")
    UTIL_CONTRI: Optional[float] = Field(None, description="Utility sector contribution (%)")
    POPULATION: Optional[float] = Field(None, ge=0, description="State population")
    POPPCT_URBAN: Optional[float] = Field(None, ge=0, le=100, description="Urban population %")
    POPDEN_URBAN: Optional[float] = Field(None, ge=0, description="Urban population density")
    POPDEN_RURAL: Optional[float] = Field(None, ge=0, description="Rural population density")
    AREAPCT_URBAN: Optional[float] = Field(None, ge=0, le=100, description="Urban area %")
    PCT_LAND: Optional[float] = Field(None, ge=0, le=100, description="Land area %")
    PCT_WATER_TOT: Optional[float] = Field(None, ge=0, le=100, description="Water area %")
    CLIMATE_REGION: Optional[str] = Field(None, description="U.S. climate region")
    CLIMATE_CATEGORY: Optional[str] = Field(None, description="Climate episode category (warm/cold/normal)")
    CAUSE_CATEGORY: Optional[str] = Field(None, description="Outage cause category")
    NERC_REGION: Optional[str] = Field(None, description="NERC reliability region")
    MONTH: Optional[int] = Field(None, ge=1, le=12, description="Month of event (1-12)")
    RES_SALES: Optional[float] = Field(None, ge=0, description="Residential electricity sales (MWh)")

    class Config:
        json_schema_extra = {
            "example": {
                "ANOMALY_LEVEL": -0.3,
                "DEMAND_LOSS_MW": 250.0,
                "RES_PRICE": 11.6,
                "COM_PRICE": 9.5,
                "IND_PRICE": 6.7,
                "TOTAL_PRICE": 9.3,
                "TOTAL_CUSTOMERS": 2500000,
                "POPULATION": 5800000,
                "POPPCT_URBAN": 73.0,
                "CLIMATE_REGION": "East North Central",
                "CAUSE_CATEGORY": "severe weather",
                "NERC_REGION": "RFC",
                "MONTH": 7,
            }
        }


class PredictionResponse(BaseModel):
    probability: float = Field(description="P(high-impact outage)")
    prediction: int = Field(description="Binary label (0 or 1)")
    risk_tier: str = Field(description="LOW / MODERATE / HIGH / CRITICAL")
    model_version: str


# ── Helpers ───────────────────────────────────────────────────────────────

def _input_to_record(inp: OutageInput) -> Dict[str, Any]:
    """Convert Pydantic model to the dict format expected by the predictor."""
    raw = inp.model_dump(exclude_none=True)
    # Map underscored API field names back to dotted dataset column names
    mapped: Dict[str, Any] = {}
    for key, val in raw.items():
        col_name = key.replace("_", ".")
        mapped[col_name] = val
    return mapped


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    status = "ok" if _predictor is not None else "model_not_loaded"
    return {"status": status, "version": MODEL_VERSION}


@app.get("/model-info")
async def model_info() -> Dict[str, Any]:
    if _predictor is None:
        raise HTTPException(503, "Model not loaded — run training pipeline first.")
    metrics_path = ARTIFACTS_DIR / METRICS_FILE
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "version": MODEL_VERSION,
        "features": _predictor.feature_names,
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(inp: OutageInput) -> PredictionResponse:
    if _predictor is None:
        raise HTTPException(503, "Model not loaded — run training pipeline first.")

    try:
        record = _input_to_record(inp)
        result = _predictor.predict_single(record)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(422, f"Prediction error: {e}")

    return PredictionResponse(
        probability=round(result["probability"], 4),
        prediction=result["prediction"],
        risk_tier=result["risk_tier"],
        model_version=MODEL_VERSION,
    )

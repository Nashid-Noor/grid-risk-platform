"""
Gradio interface for the Grid Risk & Reliability Platform.

Usage
-----
    python -m src.app                     # launches on localhost:7860
    GEMINI_API_KEY=... python -m src.app   # enables plain-language explanations

Features
--------
  • Input form with key outage-related features
  • Risk score + tier badge
  • SHAP top contributing factors table
  • Optional Gemini-powered plain-language summary
"""
from __future__ import annotations

import json
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (avoids loading model on import)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_predictor():
    from src.predict import GridRiskPredictor
    return GridRiskPredictor()


@lru_cache(maxsize=1)
def _get_feature_names() -> List[str]:
    from src.config import ARTIFACTS_DIR, FEATURE_NAMES_FILE
    with open(ARTIFACTS_DIR / FEATURE_NAMES_FILE) as f:
        return json.load(f)


def _get_gemini_model():
    """Return a Gemini GenerativeModel or None if API key is absent."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        logger.warning("Gemini init failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

TIER_COLORS = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MODERATE": "🟡",
    "LOW": "🟢",
}


def _build_record(
    anomaly_level: float,
    demand_loss_mw: float,
    res_price: float,
    com_price: float,
    ind_price: float,
    total_price: float,
    total_sales: float,
    total_customers: float,
    population: float,
    poppct_urban: float,
    popden_urban: float,
    popden_rural: float,
    climate_region: str,
    climate_category: str,
    cause_category: str,
    nerc_region: str,
    month: int,
) -> Dict[str, Any]:
    """Pack UI inputs into the dict format expected by the predictor."""
    return {
        "ANOMALY.LEVEL": anomaly_level,
        "DEMAND.LOSS.MW": demand_loss_mw,
        "RES.PRICE": res_price,
        "COM.PRICE": com_price,
        "IND.PRICE": ind_price,
        "TOTAL.PRICE": total_price,
        "TOTAL.SALES": total_sales,
        "TOTAL.CUSTOMERS": total_customers,
        "POPULATION": population,
        "POPPCT_URBAN": poppct_urban,
        "POPDEN_URBAN": popden_urban,
        "POPDEN_RURAL": popden_rural,
        "CLIMATE.REGION": climate_region,
        "CLIMATE.CATEGORY": climate_category,
        "CAUSE.CATEGORY": cause_category,
        "NERC.REGION": nerc_region,
        "MONTH": month,
    }


def predict_risk(
    anomaly_level, demand_loss_mw, res_price, com_price, ind_price,
    total_price, total_sales, total_customers, population, poppct_urban,
    popden_urban, popden_rural, climate_region, climate_category,
    cause_category, nerc_region, month,
) -> Tuple[str, str]:
    """Run prediction and return (risk_summary, shap_table_markdown)."""
    predictor = _get_predictor()
    record = _build_record(
        anomaly_level, demand_loss_mw, res_price, com_price, ind_price,
        total_price, total_sales, total_customers, population, poppct_urban,
        popden_urban, popden_rural, climate_region, climate_category,
        cause_category, nerc_region, int(month),
    )

    result = predictor.predict_single(record)
    prob = result["probability"]
    tier = result["risk_tier"]
    icon = TIER_COLORS.get(tier, "⚪")

    # SHAP explanation
    from src.explain import explain_prediction
    df = pd.DataFrame([record])
    from src.features import engineer_features
    df = engineer_features(df)
    
    # Ensure all columns expected by the preprocessor are present (fill with NaN if missing)
    expected_cols = getattr(predictor.preprocessor, "feature_names_in_", [])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    X = predictor.preprocessor.transform(df)
    shap_factors = explain_prediction(
        X, model=predictor.model, feature_names=predictor.feature_names, top_k=8
    )

    # Format outputs
    summary = (
        f"## {icon} Risk Tier: **{tier}**\n\n"
        f"**Probability of high-impact outage:** {prob:.1%}\n\n"
        f"Threshold: ≥50% → positive prediction"
    )

    table_rows = ["| Feature | SHAP Value | Direction |", "|---|---|---|"]
    for f in shap_factors:
        table_rows.append(f"| {f['feature']} | {f['shap_value']:+.4f} | {f['direction']} |")
    shap_table = "\n".join(table_rows)

    return summary, shap_table


def generate_gemini_explanation(
    anomaly_level, demand_loss_mw, res_price, com_price, ind_price,
    total_price, total_sales, total_customers, population, poppct_urban,
    popden_urban, popden_rural, climate_region, climate_category,
    cause_category, nerc_region, month,
) -> str:
    """Call Gemini to produce a plain-language explanation of the risk."""
    model = _get_gemini_model()
    if model is None:
        return (
            "⚠️ Gemini API key not configured.\n\n"
            "Set the `GEMINI_API_KEY` environment variable and restart the app."
        )

    # Run the prediction first to get the numbers
    predictor = _get_predictor()
    record = _build_record(
        anomaly_level, demand_loss_mw, res_price, com_price, ind_price,
        total_price, total_sales, total_customers, population, poppct_urban,
        popden_urban, popden_rural, climate_region, climate_category,
        cause_category, nerc_region, int(month),
    )
    result = predictor.predict_single(record)

    # Get SHAP factors
    from src.explain import explain_prediction
    from src.features import engineer_features
    df = pd.DataFrame([record])
    df = engineer_features(df)
    
    # Ensure all columns expected by the preprocessor are present (fill with NaN if missing)
    expected_cols = getattr(predictor.preprocessor, "feature_names_in_", [])
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    X = predictor.preprocessor.transform(df)
    shap_factors = explain_prediction(
        X, model=predictor.model, feature_names=predictor.feature_names, top_k=5
    )

    prompt = f"""You are a grid reliability analyst explaining an AI risk prediction to a
non-technical utility operations manager. Be concise (3-4 sentences max).

Prediction: {result['probability']:.1%} probability of high-impact outage → {result['risk_tier']} tier.

Top contributing factors:
{json.dumps(shap_factors, indent=2)}

Input context:
- Climate region: {climate_region}
- Cause category: {cause_category}
- Month: {month}
- Population: {population:,.0f}
- Anomaly level (ONI): {anomaly_level}

Explain what is driving this risk score and what the operations team should watch for.
Do NOT mention SHAP, ML models, or technical jargon."""

    try:
        response = model.generate_content(prompt)
        return f"### 💡 AI Explanation\n\n{response.text}"
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

CLIMATE_REGIONS = [
    "East North Central", "Central", "Northeast", "Northwest",
    "South", "Southeast", "Southwest", "West", "West North Central",
]
CLIMATE_CATEGORIES = ["normal", "warm", "cold"]
CAUSE_CATEGORIES = [
    "severe weather", "intentional attack", "system operability disruption",
    "public appeal", "equipment failure", "fuel supply emergency", "islanding",
]
NERC_REGIONS = ["RFC", "SERC", "WECC", "TRE", "NPCC", "MRO", "SPP", "FRCC", "ECAR", "HECO"]


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Grid Risk & Reliability Platform",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# ⚡ Grid Risk & Reliability Platform\n"
            "Predict the probability of a high-impact power outage event. "
            "All inputs are optional — the model handles missing values."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Scenario Inputs")
                anomaly_level = gr.Number(label="Anomaly Level (ONI)", value=-0.3)
                demand_loss_mw = gr.Number(label="Demand Loss (MW)", value=250.0)
                month = gr.Slider(1, 12, step=1, value=7, label="Month")

                with gr.Row():
                    res_price = gr.Number(label="Res. Price", value=11.6)
                    com_price = gr.Number(label="Com. Price", value=9.5)
                    ind_price = gr.Number(label="Ind. Price", value=6.7)
                    total_price = gr.Number(label="Total Price", value=9.3)

                total_sales = gr.Number(label="Total Sales (MWh)", value=6.5e7)
                total_customers = gr.Number(label="Total Customers", value=2.5e6)
                population = gr.Number(label="State Population", value=5.8e6)
                poppct_urban = gr.Number(label="Urban Pop %", value=73.0)
                popden_urban = gr.Number(label="Urban Density", value=2200.0)
                popden_rural = gr.Number(label="Rural Density", value=18.0)

                climate_region = gr.Dropdown(CLIMATE_REGIONS, label="Climate Region", value="East North Central")
                climate_category = gr.Dropdown(CLIMATE_CATEGORIES, label="Climate Category", value="normal")
                cause_category = gr.Dropdown(CAUSE_CATEGORIES, label="Cause Category", value="severe weather")
                nerc_region = gr.Dropdown(NERC_REGIONS, label="NERC Region", value="RFC")

            with gr.Column(scale=1):
                gr.Markdown("### Prediction Results")
                risk_output = gr.Markdown(label="Risk Score")
                shap_output = gr.Markdown(label="Top Risk Factors (SHAP)")

                predict_btn = gr.Button("🔍 Predict Risk", variant="primary", size="lg")

                gr.Markdown("---")
                gr.Markdown("### Plain-Language Explanation (Gemini)")
                gemini_output = gr.Markdown(label="AI Explanation")
                gemini_btn = gr.Button("💬 Generate Explanation", variant="secondary")

        all_inputs = [
            anomaly_level, demand_loss_mw, res_price, com_price, ind_price,
            total_price, total_sales, total_customers, population, poppct_urban,
            popden_urban, popden_rural, climate_region, climate_category,
            cause_category, nerc_region, month,
        ]

        predict_btn.click(fn=predict_risk, inputs=all_inputs, outputs=[risk_output, shap_output])
        gemini_btn.click(fn=generate_gemini_explanation, inputs=all_inputs, outputs=gemini_output)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)

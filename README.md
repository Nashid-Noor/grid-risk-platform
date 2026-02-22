---
title: Grid Risk Platform
emoji: 🌍
colorFrom: red
colorTo: pink
sdk: gradio
sdk_version: 6.6.0
app_file: app.py
pinned: false
---

# ⚡ AI-Powered Grid Risk & Reliability Platform

**Live Demo:** [Play with the interactive Grid Risk Platform here!](https://huggingface.co/spaces/nashid16/grid-risk-platform?logs=container)


Production-structured outage risk prediction system for electricity distribution operations. Predicts the probability of high-impact power outage events with explainability, monitoring, and governance built in.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                              │
│  Kaggle "Major Power Outage" CSV → data.py (load/clean/label) │
│  → features.py (engineer + sklearn ColumnTransformer)          │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│                      MODEL LAYER                               │
│  train.py: Logistic Regression baseline + XGBoost final        │
│  → 5-fold stratified CV, PR-AUC / ROC-AUC / Brier             │
│  → artifacts/ (joblib models, preprocessor, metrics.json)      │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│                    INFERENCE LAYER                              │
│  predict.py: GridRiskPredictor (load artifacts, score, drift)  │
│  explain.py: SHAP TreeExplainer (global + local)               │
└────────┬──────────────────────┬────────────────────────────────┘
         │                      │
┌────────▼────────┐   ┌────────▼────────────────────────────────┐
│  FastAPI (api.py) │   │   Gradio UI (app.py)                   │
│  POST /predict    │   │   Input form → risk score + SHAP table │
│  GET /health      │   │   Optional: Gemini plain-language       │
│  GET /model-info  │   │   explanation via LLM                   │
└──────────────────┘   └─────────────────────────────────────────┘
         │                      │
┌────────▼──────────────────────▼────────────────────────────────┐
│                   MONITORING LAYER                              │
│  monitor.py: Evidently drift reports + retrain trigger          │
│  predict.py: per-request z-score drift check                   │
│  → monitoring_reports/ (HTML + JSON, timestamped)              │
└────────────────────────────────────────────────────────────────┘
         │
┌────────▼───────────────────────────────────────────────────────┐
│                   GOVERNANCE LAYER                              │
│  governance/model_card.md       — capabilities & limitations   │
│  governance/risk_assessment.md  — failure modes & mitigations  │
│  governance/monitoring_plan.md  — drift detection & retraining │
│  governance/approval_checklist.md — deployment sign-off        │
└────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
grid-risk-platform/
├── run_pipeline.py              # End-to-end training + explainability
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                # Paths, constants, feature lists
│   ├── data.py                  # Load, clean, target definition
│   ├── features.py              # Engineering + sklearn preprocessor
│   ├── train.py                 # LR baseline + XGBoost + CV + metrics
│   ├── predict.py               # Inference + z-score drift detection
│   ├── explain.py               # SHAP global & local explanations
│   ├── monitor.py               # Evidently drift reports + retrain trigger
│   ├── api.py                   # FastAPI prediction endpoint
│   └── app.py                   # Gradio UI
├── artifacts/                   # Serialised models, metrics, preprocessor
├── monitoring_reports/          # Evidently HTML/JSON reports
├── governance/
│   ├── model_card.md
│   ├── risk_assessment.md
│   ├── monitoring_plan.md
│   └── approval_checklist.md
└── data/                        # Place outage_data.csv here
```

---

## Dataset

**Source:** [Major Power Outage Risks in the U.S.](https://www.kaggle.com/datasets/autunno/major-power-outages-us) (Purdue / DOE)

Approximately 1,500 major outage events across the continental US (2000–2016). Each record includes cause, duration, customers affected, demand loss, climate conditions, economic indicators, and demographic context.

**No synthetic data is used.** All features are derived from columns present in the original dataset.

### Assumptions

1. **Target variable:** "High-impact" = customers affected ≥ 75th percentile AND duration ≥ 75th percentile. This captures the upper-severity tail. The threshold is configurable in `config.py`.
2. **Leakage prevention:** `CUSTOMERS.AFFECTED` and `OUTAGE.DURATION` define the target and are excluded from features. The model must learn risk from structural signals alone.
3. **Weather proxy:** `ANOMALY.LEVEL` (Oceanic Niño Index) serves as the climate signal. No external weather API is required.
4. **Cause category at inference:** `CAUSE.CATEGORY` may not be known pre-event. The model handles it as missing when omitted.
5. **Train/test split:** Random stratified (80/20). A time-based split is recommended for production validation but not enforced in v1 given the small dataset.
6. **CSV format:** The raw Kaggle file may have header rows to skip. Adjust `SKIP_ROWS_IN_RAW` in `config.py` if needed (default: 6 for the Excel export format).

---

## Business KPI Framing

This system maps to three operational KPIs that grid operators track:

### 1. SAIDI Reduction (System Average Interruption Duration Index)

High-impact events disproportionately drive SAIDI. By identifying elevated-risk windows in advance, operations centres can pre-position crews and materials, reducing restoration time. A 10% improvement in preparedness response for correctly flagged events could reduce SAIDI contribution from major events by 5–15%.

### 2. Resource Pre-Positioning Efficiency

False positive cost is measurable: crew overtime, equipment staging, unused mutual aid requests. The model's calibrated probabilities allow graduated response — full mobilisation for CRITICAL tier, partial readiness for HIGH, watchlist for MODERATE — rather than binary go/no-go.

### 3. Regulatory Performance (NERC/state PUC)

Demonstrating data-driven risk management strengthens performance-based rate case filings and NERC reliability assessments. The governance package (model card, risk assessment, monitoring plan) provides auditable evidence of systematic risk management.

**Leading indicator:** Fraction of actual high-impact events that were flagged as HIGH or CRITICAL tier ≥24h before the event.  
**Lagging indicator:** SAIDI and CAIDI trends for events that received model-assisted preparation vs. those that did not.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
# Place the Kaggle CSV at data/outage_data.csv
python run_pipeline.py --data data/outage_data.csv
```

This produces all artifacts in `artifacts/` and a SHAP summary plot.

### 3. Launch the API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Test with:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ANOMALY_LEVEL": -0.3, "DEMAND_LOSS_MW": 250, "CAUSE_CATEGORY": "severe weather", "MONTH": 7, "POPULATION": 5800000, "CLIMATE_REGION": "East North Central", "NERC_REGION": "RFC"}'
```

Interactive docs at `http://localhost:8000/docs` (Swagger UI).

### 4. Launch the Gradio UI

```bash
python -m src.app
# Open http://localhost:7860

# With Gemini explanations:
GEMINI_API_KEY=your-key-here python -m src.app
```

### 5. Run drift monitoring

```python
from src.data import get_dataset
from src.monitor import DriftMonitor

training_data = get_dataset("data/outage_data.csv")
production_batch = get_dataset("data/new_batch.csv")

monitor = DriftMonitor(training_data, production_batch)
report_path = monitor.generate_report()
print(f"Retrain recommended: {monitor.should_retrain()}")
```

---

## Monitoring Strategy

The system implements three complementary monitoring layers:

**Per-request (real-time):** Z-score drift detection in `predict.py` compares each incoming record's feature values against training-time mean/std. Flags `warn` (z ≥ 2.0) or `alert` (z ≥ 3.5) per feature.

**Per-batch (daily):** Evidently `DataDriftPreset` in `monitor.py` runs KS-test (numeric) and chi-squared (categorical) comparisons. Generates HTML + JSON reports archived in `monitoring_reports/`.

**Retrain trigger:** If >30% of tracked features show distributional drift, `should_retrain()` returns True. In a production scheduler, this gates a retraining DAG.

**Ground truth loop:** When actual outage outcomes are available (delayed), join predictions to outcomes and compute ROC-AUC/PR-AUC against the training baseline. Alert if PR-AUC drops below 80% of the test-set value.

See `governance/monitoring_plan.md` for full operational procedures.

---

## Deployment: Hugging Face Spaces

The Gradio UI is directly deployable to Hugging Face Spaces.

### Steps

1. Create a new Space on huggingface.co (SDK: Gradio).

2. Push the repository:
```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/grid-risk-platform
git push hf main
```

3. Ensure `artifacts/` contains trained model files. Either commit artifacts to the repo (acceptable for models <500MB) or use `huggingface_hub` to download at startup.

4. Set secrets in the Space settings:
   - `GEMINI_API_KEY` (optional, for plain-language explanations)

5. The Space entry point should be `src/app.py`. Add a root-level launcher if needed:
```bash
echo 'from src.app import build_app; app = build_app(); app.launch()' > app.py
```

6. `requirements.txt` is auto-installed by HF Spaces.

### Dockerfile alternative

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["python", "-m", "src.app"]
```

---

## Migration Path: Azure ML

For enterprise deployment on Azure ML, the following mapping applies:

| Current Component | Azure ML Equivalent |
|---|---|
| `run_pipeline.py` | Azure ML Pipeline (command steps) |
| `artifacts/` | Azure ML Model Registry (versioned model assets) |
| `src/train.py` | Training component registered as Azure ML Command |
| `src/api.py` | Azure ML Managed Online Endpoint |
| `src/monitor.py` | Azure ML Data Drift Monitor or scheduled Evidently job |
| `monitoring_reports/` | Azure Blob Storage + Azure Monitor alerts |
| `governance/` | Model Registry metadata + Responsible AI dashboard |

### Migration steps

1. **Register the model** in Azure ML Model Registry with version metadata.
2. **Create a managed endpoint:** Wrap `src/predict.py` in a scoring script with `init()` and `run()`. Deploy on CPU compute (Standard_DS2_v2 is sufficient).
3. **Schedule monitoring:** Register a Data Drift Monitor against the training dataset, or run the Evidently pipeline as a recurring job.
4. **CI/CD:** GitHub Actions or Azure DevOps pipeline: lint → test → train → evaluate → register → deploy. Gate production on the approval checklist.

---

## Governance

All governance documentation is in the `governance/` directory:

| Document | Purpose |
|---|---|
| `model_card.md` | Model capabilities, limitations, bias considerations, intended use |
| `risk_assessment.md` | Failure modes, data privacy, operational risks, regulatory notes |
| `monitoring_plan.md` | Drift detection procedures, retraining policy, rollback, alerting |
| `approval_checklist.md` | Pre-deployment verification with sign-off matrix and versioning policy |

---

## Phase Roadmap

- [x] **Phase 1** — Core ML pipeline (data, features, training, explainability)
- [x] **Phase 2** — Deployment, monitoring, governance (API, UI, Evidently, docs)
- [ ] **Phase 3** — MLOps hardening (CI/CD, model registry, A/B testing, temporal CV)

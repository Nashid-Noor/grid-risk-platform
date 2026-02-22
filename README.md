#  AI-Powered Grid Risk & Reliability Platform

Production-structured outage risk prediction system for electricity distribution operations. Predicts the probability of high-impact power outage events with explainability, monitoring, and governance built in.

---
**Live Demo:** ([https://huggingface.co/spaces/nashid16/grid-risk-platform?logs=container](https://huggingface.co/spaces/nashid16/grid-risk-platform))

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

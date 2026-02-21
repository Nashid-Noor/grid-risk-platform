# Monitoring Plan — Grid Risk Prediction System

**Version:** 1.0  
**Effective date:** 2025-02  
**Review cadence:** Quarterly

---

## 1. Monitoring Objectives

1. Detect data drift before model accuracy degrades visibly.
2. Track prediction distribution to catch silent failures (model always predicting one class).
3. Maintain calibration quality so probability outputs remain trustworthy.
4. Provide an auditable trail of model performance over time.

## 2. Monitoring Layers

### Layer 1 — Input Data Quality (every batch)

| Check | Method | Threshold | Action |
|---|---|---|---|
| Missing value rate | Per-column null % | >50% on any feature | Alert + log |
| Schema validation | Pydantic model in FastAPI | Type mismatch | Reject request (422) |
| Range violations | Min/max from training ref | Beyond 3× training range | Warn in logs |
| Volume anomaly | Request count per hour | <10% or >500% of baseline | Alert |

### Layer 2 — Feature Drift (daily or per-batch)

**Primary tool:** Evidently `DataDriftPreset`

The `DriftMonitor` class in `src/monitor.py` compares a reference dataset (training data) against incoming production data using statistical tests (KS-test for numeric features, chi-squared for categorical features).

**Workflow:**
1. Accumulate a day's worth of inference requests into a batch DataFrame.
2. Instantiate `DriftMonitor(reference_df=training_data, current_df=batch)`.
3. Call `monitor.generate_report()` → saves HTML + JSON to `monitoring_reports/`.
4. Check `monitor.should_retrain()` → returns `True` if >30% of features have drifted.

**Fallback:** The simpler z-score drift detector in `src/predict.py` runs per-request with no dependencies beyond numpy and provides immediate per-feature warnings.

### Layer 3 — Prediction Distribution (daily)

| Metric | Computation | Alert condition |
|---|---|---|
| Mean predicted probability | Average P(high_impact) over batch | Shift >0.15 from training mean |
| Positive rate | Fraction of predictions ≥ 0.5 | >2× or <0.5× training positive rate |
| Entropy of predictions | Shannon entropy of probability distribution | Drop >30% (model becoming overconfident) |

These are computed from logged predictions and compared against training-time baselines stored in `artifacts/drift_reference.joblib`.

### Layer 4 — Ground Truth Feedback (when available)

When actual outage outcomes become available (typically weeks or months after the event):

1. Join predictions to outcomes by event identifier.
2. Compute ROC-AUC, PR-AUC, and Brier score on the labelled batch.
3. Compare against the test-set metrics in `artifacts/metrics.json`.
4. If PR-AUC drops below 80% of the test-set value, trigger retraining review.

This loop is necessarily delayed. Layers 1–3 provide early warning in the interim.

## 3. Retraining Policy

### Automatic Trigger

The `retrain_trigger()` function in `src/monitor.py` returns `True` when Evidently detects distributional drift in ≥30% of tracked features. In a scheduled deployment, this would gate a retraining DAG (Airflow, Prefect, etc.).

### Scheduled Retraining

Even without drift detection, the model should be retrained:
- **Quarterly** with any newly available outage data.
- **Immediately** after any major grid topology change (utility merger, new transmission line, generation retirement).

### Retraining Procedure

1. Acquire updated dataset (new outage events appended to historical data).
2. Run `python run_pipeline.py --data data/updated_outage_data.csv`.
3. Compare new metrics against previous version in `artifacts/metrics.json`.
4. If PR-AUC improves or holds steady: promote new artifacts.
5. If PR-AUC degrades: investigate root cause before promoting. See rollback procedure below.

## 4. Rollback Procedure

Model artifacts are versioned by directory:

```
artifacts/
  v1.0.0/
    xgb_final.joblib
    preprocessor.joblib
    metrics.json
    ...
  v1.1.0/
    ...
```

**To rollback:**
1. Update `MODEL_VERSION` in `src/config.py` to the previous version string.
2. Point `ARTIFACTS_DIR` to the previous version's directory (or copy artifacts back).
3. Restart the API/UI service.
4. Validate with a smoke-test batch to confirm the previous model loads and scores correctly.

In a container deployment, rollback means redeploying the previous container image tag.

## 5. Alerting

| Severity | Condition | Channel | Response SLA |
|---|---|---|---|
| INFO | Drift detected in 1–2 features | Log file | Review at next standup |
| WARN | Drift in 3+ features or prediction distribution shift | Email/Slack | Investigate within 24h |
| CRITICAL | Retrain trigger fired or model fails to load | PagerDuty / on-call | Investigate within 2h |

## 6. Monitoring Report Archive

All Evidently HTML/JSON reports are saved to `monitoring_reports/` with timestamps. Retain for at least 12 months for audit purposes.

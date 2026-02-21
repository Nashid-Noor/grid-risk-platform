# Risk Assessment — Grid Outage Risk Prediction System

**Document version:** 1.0  
**Assessment date:** 2025-02  
**Assessor:** ML Engineering Lead  
**Review cadence:** Quarterly, or upon any model retraining

---

## 1. System Risk Classification

**Overall risk level: MEDIUM**

This system provides advisory risk scores to human operators. It does not control physical infrastructure or make autonomous decisions. However, over-reliance on model outputs could lead to misallocation of restoration resources, making governance essential.

## 2. Failure Modes

### 2.1 False Negatives (missed high-impact events)

**Impact:** An actual severe outage is scored as low-risk, leading to under-preparation.

**Likelihood:** Moderate. The model is trained on ~1,500 events and has limited ability to generalise to novel failure modes (e.g., coordinated cyber-physical attacks, cascading failures from new renewable penetration patterns).

**Mitigation:**
- PR-AUC is the primary metric precisely because it penalises missed positives under class imbalance.
- The decision threshold (default 0.5) can be lowered to increase recall at the cost of more false alarms. A threshold of 0.3 should be evaluated for safety-critical deployments.
- Physics-based reliability models (N-1 analysis) remain the primary defense layer; this ML system is supplementary.

### 2.2 False Positives (over-prediction of risk)

**Impact:** Resources are pre-positioned for events that don't materialise, increasing operational cost.

**Likelihood:** Moderate-high during climate anomaly periods (elevated ONI values).

**Mitigation:**
- Operations teams should calibrate their response based on probability bands (LOW/MODERATE/HIGH/CRITICAL), not binary labels.
- Brier score and calibration curves are tracked to ensure predicted probabilities are reliable.

### 2.3 Data Drift

**Impact:** Model accuracy degrades silently as the grid evolves (new generation mix, infrastructure upgrades, regulatory changes, climate shifts).

**Likelihood:** High over any 2+ year window.

**Mitigation:**
- Evidently-based drift monitoring runs on each inference batch.
- Z-score reference statistics are saved from training and compared at inference.
- Automated retrain trigger fires when >30% of features show distributional shift.
- See `monitoring_plan.md` for operational procedures.

### 2.4 Target Definition Instability

**Impact:** The 75th-percentile thresholds are computed from training data. If the severity distribution shifts (e.g., more extreme events due to climate change), the target definition no longer captures the intended risk tier.

**Mitigation:**
- Thresholds should be reviewed annually against updated data.
- Consider absolute thresholds (e.g., >100k customers AND >24h) instead of relative percentiles for production deployments.

### 2.5 Feature Leakage at Inference

**Impact:** `CAUSE.CATEGORY` is a strong predictor but is often unknown at the time a risk assessment is needed (pre-event).

**Mitigation:**
- The UI and API accept `CAUSE.CATEGORY` as optional. When omitted, the preprocessor encodes it as `MISSING` and the model relies on other signals.
- Documentation makes clear that pre-event predictions without cause information will have wider uncertainty.

## 3. Data Privacy

### 3.1 PII Assessment

The Major Power Outage dataset contains **no personally identifiable information**. Records are aggregated at the state/utility level. No customer names, addresses, or account numbers are present.

### 3.2 Sensitive Data

- State-level demographic data (population, urban/rural density) is present. While not PII, it could enable geographic profiling.
- Economic indicators (`PC.REALGSP.STATE`, utility pricing) are publicly available from government sources.

### 3.3 Data Handling

- The raw dataset is stored locally and not transmitted to external services.
- The optional Gemini API integration sends only aggregated scenario parameters and SHAP summaries — no raw data, no customer information.
- API responses include only the risk score and feature contributions, not the underlying training data.

## 4. Operational Risks

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| Model artifacts corrupted or deleted | Medium | Low | Versioned artifact storage; retrain pipeline is fully automated |
| API endpoint goes down | Low | Medium | Health check endpoint; container orchestration with restart policy |
| Operator over-trusts model | High | Medium | Clear UI labeling as "advisory"; mandatory human sign-off in SOPs |
| Model used outside intended scope | High | Low | Model card specifies intended use; API documentation includes limitations |
| Adversarial manipulation of inputs | Medium | Low | Input validation in FastAPI schema; anomaly detection on incoming data |

## 5. Regulatory Considerations

- NERC CIP standards may apply if the system is connected to bulk electric system operations. In its current advisory-only capacity, it is unlikely to fall under CIP scope, but this should be confirmed with compliance.
- If deployed in the EU, the AI Act's risk classification for critical infrastructure would apply. The system would likely be classified as "high-risk" and require conformity assessment.
- State PUC reporting requirements should be reviewed to ensure model-assisted decisions are documented.

## 6. Approval

| Role | Name | Date | Sign-off |
|---|---|---|---|
| ML Engineering Lead | ____________ | ______ | ☐ |
| Grid Operations Manager | ____________ | ______ | ☐ |
| Compliance Officer | ____________ | ______ | ☐ |
| Data Privacy Officer | ____________ | ______ | ☐ |

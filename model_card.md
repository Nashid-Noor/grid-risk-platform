# Model Card — Grid Outage High-Impact Risk Classifier

**Version:** 1.0.0  
**Owner:** Grid Analytics Team  
**Last updated:** 2025-02  
**Status:** Staging — pending operational validation

---

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost binary classifier (gradient-boosted trees) |
| Baseline | Logistic Regression with balanced class weights |
| Framework | XGBoost 2.x via scikit-learn API |
| Training data | Major Power Outage Dataset (US), Purdue/DOE, ~1,500 events (2000–2016) |
| Target | Binary: high-impact outage (customers affected ≥ 75th pctl AND duration ≥ 75th pctl) |
| Primary metric | PR-AUC (chosen over ROC-AUC due to class imbalance) |
| Input features | 20 numeric + 5 categorical (after engineering); leakage columns excluded |
| Inference latency | <50ms per prediction (CPU, single record) |

## Intended Use

**Primary use case:** Decision-support tool for grid operations centres to prioritise outage preparedness and resource pre-positioning. The model scores incoming scenarios (weather conditions, region, demand patterns) and flags elevated-risk windows.

**Not intended for:**
- Autonomous switching or load-shedding decisions without human review.
- Replacement of physics-based grid reliability models (N-1 contingency analysis).
- Real-time protective relay logic or SCADA integration.
- Regulatory compliance reporting as a sole data source.

**End users:** Operations managers, reliability engineers, dispatch coordinators. All predictions are advisory; operational decisions remain with licensed personnel.

## Training and Evaluation

### Data

The dataset covers major outage events reported to DOE across the continental US. Each record represents a single outage event with associated economic, demographic, climate, and cause-category attributes.

**Preprocessing:**
- Rows missing both `CUSTOMERS.AFFECTED` and `OUTAGE.DURATION` are dropped.
- Numeric columns imputed with median; categoricals filled with a sentinel value.
- Features standardised (z-score) for the logistic baseline; XGBoost uses raw scale.

**Target construction:** An event qualifies as "high-impact" when it falls in the top quartile on BOTH affected customers and outage duration. Both columns are removed from features to prevent leakage.

### Evaluation

| Metric | Logistic Baseline | XGBoost Final |
|---|---|---|
| ROC-AUC (test) | — | — |
| PR-AUC (test) | — | — |
| Brier score | — | — |
| CV ROC-AUC (5-fold) | — | — |

*(Fill from `artifacts/metrics.json` after training.)*

**Calibration:** Brier score is tracked. If max bin gap exceeds 0.10, Platt scaling is recommended before deployment.

## Bias and Fairness

### Known limitations

- **Geographic bias:** The dataset is skewed towards states with mandatory reporting. Rural and tribal regions are underrepresented, which likely means the model underestimates risk in those areas.
- **Temporal coverage:** Data spans 2000–2016. Grid infrastructure, generation mix, and climate patterns have shifted materially since then. The model should not be assumed accurate for post-2016 conditions without retraining on updated data.
- **Socioeconomic proxy variables:** Features like `POPPCT_URBAN`, `PC.REALGSP.STATE`, and electricity pricing encode socioeconomic structure. The model may implicitly associate low-income or rural regions with different risk levels, which requires careful interpretation.
- **Cause-category leakage risk:** `CAUSE.CATEGORY` is known post-hoc for historical events. At inference time, the operator must either supply an expected cause or leave it as missing. This gap between training and deployment conditions should be communicated clearly.

### Mitigation steps

1. Feature importance is monitored via SHAP to detect if protected-class proxies dominate predictions.
2. Subgroup performance analysis should be conducted across NERC regions before production deployment.
3. Model outputs are presented as probabilities with explicit uncertainty, not deterministic labels.

## Limitations

- The model has no access to real-time weather, load, or topology data. `ANOMALY.LEVEL` (ONI) is a coarse climate proxy.
- Approximately 1,500 training examples is a small dataset by ML standards. Predictions carry meaningful uncertainty.
- The 75th-percentile threshold is a policy choice, not a physical constant. Operators should adjust in `config.py` if their risk appetite differs.
- Cross-validation uses random splits, not temporal splits. A time-aware evaluation would better reflect deployment conditions.

## Ethical Considerations

- The model must not be used to justify reducing service investment in historically underserved communities.
- Risk scores should inform resource allocation, not determine which communities receive slower restoration response.
- All predictions should be reviewed by a qualified engineer before influencing operational decisions.

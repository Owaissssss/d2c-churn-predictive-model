# Formal Model Card: Random Forest Churn Scoring Pipeline

## 1. Intended Scope
* **Primary Use Case**: Identifies high-risk customer profiles likely to churn over a rolling 60-day window to support proactive retention actions.
* **Target Audience**: Product Analytics teams, CRM Automation engineers, and Performance Marketing managers.
* **Out of Scope**: Not for use in restricting baseline web services, altering product pricing unfairly, or blacklisting accounts from support desks.

## 2. Training Features & Dataset Specifications
* **Data Origin**: 2,400 unique customer rows combined with historical transactional records up to the snapshot cutoff date of `2025-09-30` to prevent data leakage.
* **Categorical Encoding**: Full multi-tier dummy variable coding applied to location bands, preferred products, and historical acquisition channels.

## 3. Performance Metrics (Validation Holdout)
* **Operational Classification Threshold**: `0.40`
* **Model Evaluation Scores**:
  * Accuracy: `80.95%`
  * Precision: `77.48%`
  * Recall: `79.59%`
  * F1-Score: `78.52%`
  * ROC-AUC: `0.8783`

## 4. Ethical Framing & Monitoring Needs
* **Demographic Fair Treatment**: Missing attributes (such as skin profile tokens or loyalty records) are handled as explicit missing features rather than dropping rows, preventing performance drops on underserved segments.
* **Monitoring Triggers**: Requires automated pipeline execution and retraining every 30 days to protect against feature drift as market conditions evolve.
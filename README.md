# Vodafone Customer Churn Prediction

## Project Overview

This project leverages advanced machine learning to predict customer churn for Vodafone subscribers. The primary goal is to identify customers at risk of leaving, enabling proactive retention strategies and quantifiable business impact.

**Key Objectives:**
- Build a robust churn prediction model (CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression)
- Handle class imbalance (6.3% churn rate)
- Deliver actionable business insights and ROI analysis
- Enable data-driven retention campaigns


## Project Workflow

1. **Data Preparation & Exploration**
   - Load and inspect a large-scale telecom dataset (150,000+ customers, 800+ features)
   - Assess data quality, handle missing values and optimize memory usage

2. **Feature Engineering & Selection**
   - Remove highly correlated and low-importance features
   - Reduce dimensionality from 817 to ~315 features using business logic and model-based importance

3. **Model Training & Optimization**
   - Train multiple models (CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression)
   - Address class imbalance with balanced weights and stratified splits
   - Optimize hyperparameters and decision thresholds for business value

4. **Evaluation & Insights**
   - Evaluate models using ROC AUC, F1, precision, recall and confusion matrix
   - Analyze feature importance and translate technical drivers into business terms
   - Visualize key results (feature importance, ROC curve, confusion matrix)

5. **Business Recommendations & ROI Analysis**
   - Quantify the financial impact of retention campaigns at different response rates
   - Provide actionable recommendations for marketing and customer success teams


## Key Results

- **ROC AUC:** 0.88 (excellent for imbalanced data)
- **F1 Score:** 0.49 (good balance for business action)
- **Churners correctly identified:** 4,900+ (on test set)
- **Top churn drivers:** Local market share, customer activity, device type, package status

### Business Impact & ROI

- Retention campaign is profitable if > 7% of targeted churners are retained
- At 10% retention, campaign ROI = 40% ($83,980 net gain)
- Retaining a customer is up to 6x cheaper than acquiring a new one

---

## Anomaly Detection Approach (Advanced Experiments)

A dedicated folder, `anomaly_detection_approach`, contains experiments with unsupervised and semi-supervised anomaly detection for churn prediction:

- **Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM** were tested as standalone churn detectors. All performed poorly, with low F1 and recall, missing many churners and/or making many false alarms. Isolation Forest was the best among them, but still limited.
- **Adding the Isolation Forest anomaly score as a feature to CatBoost** increased recall (found more churners), but at the cost of a significant increase in false positives (lower precision and F1). The original CatBoost model (without iso_score) achieved higher precision and F1 on unseen data, while CatBoost+iso_score found more churners but with a higher false positive rate (up to 15%).
- **Conclusion:** The best approach depends on business needs:
  - If you want to catch as many churners as possible (high recall), CatBoost+iso_score may be preferred.
  - If you want to minimize false positives and maintain high precision/F1, the original CatBoost model may be better.

**Recommendation:** Use anomaly detection scores as features in supervised models only if your business can tolerate more false positives in exchange for higher recall. Otherwise, the original CatBoost model may be preferable for its higher precision and F1.

image.png
# Anomaly Detection Approach for Churn Prediction

This folder contains experiments and results for unsupervised and semi-supervised anomaly detection methods applied to Vodafone churn prediction data. The goal is to identify churners as anomalies/outliers in the customer base and to assess the value of these methods for business use.

## Contents

- **01_isolation_forest.ipynb**
  - Applies Isolation Forest to the churn dataset, tuning thresholds and visualizing anomaly scores.
  - **Conclusion:** The iso_score (anomaly score) provides almost no signal to distinguish churners from non-churners. Churners' scores overlap almost entirely with non-churners, and the model's F1 and recall are low.

- **02_anomaly_detection_several_models.ipynb**
  - Compares Isolation Forest, Local Outlier Factor (LOF), and One-Class SVM for churn detection.
  - **Conclusion:** Isolation Forest performs best among the three, but all models have low ROC AUC, F1, and recall. LOF and One-Class SVM perform close to random guessing and miss most churners.

- **03_catboost_enforced_by_iso.ipynb**
  - Uses the Isolation Forest anomaly score as an additional feature for a supervised CatBoost model.
  - **Conclusion:** Adding the iso_score feature to CatBoost increases the model's ability to find more churners (higher recall), but this comes at the cost of a significant increase in false positives (lower precision). In practice, the original CatBoost model (without iso_score) achieves higher precision and F1 on unseen data, while the CatBoost+iso_score model finds more churners but with a higher false positive rate (up to 15%). The choice between these models depends on business priorities: maximizing true churn detection (recall) vs. minimizing false alarms (precision).

- **anomaly_detection_report.pdf**
  - Visual and tabular summaries of the results and conclusions.

- **isolation_forest_anomaly_score_by_class.png, confusion_matrix_iso_basic.png**
  - Plots visualizing the distribution of anomaly scores and confusion matrices for model evaluation.

## Overall Conclusion

- **Unsupervised anomaly detection methods (Isolation Forest, LOF, One-Class SVM) are not effective for direct churn prediction in this dataset.**
  - All models have low F1 and recall, missing many churners and/or making many false alarms.
  - Isolation Forest is the best among them, but still limited.
- **Adding anomaly scores as features to a supervised model (CatBoost) increases recall but may reduce precision and F1, leading to more false positives.**
- **The best approach depends on business needs:**
  - If you want to catch as many churners as possible (high recall), CatBoost+iso_score may be preferred.
  - If you want to minimize false positives and maintain high precision/F1, the original CatBoost model may be better.

**Recommendation:** Use anomaly detection scores as features in supervised models only if your business can tolerate more false positives in exchange for higher recall. Otherwise, the original CatBoost model may be preferable for its higher precision and F1. 
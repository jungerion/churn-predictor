## Results
Random Forest achieved accuracy 0.807, precision 0.682, recall 0.511, F1 0.584, and
ROC-AUC 0.841 on the held-out test set. Logistic Regression performed comparably
(accuracy 0.806, F1 0.604, ROC-AUC 0.842), suggesting the relationship between
these features and churn is largely captured by a simpler linear model.

## What I'd improve
- Class imbalance handling (~27% churn rate) — recall of 0.51 means the model
  misses about half of actual churners; techniques like class weighting or
  SMOTE could help catch more of them, at the cost of more false positives.
- Hyperparameter tuning via cross-validation rather than fixed values in config.
- Feature importance / SHAP analysis to explain *why* the model flags a
  customer, not just that it does — important for a real business use case.
- CI pipeline (GitHub Actions) to run tests automatically on every push.
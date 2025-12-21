1.Data cleaning including missing values, outliers and multi-collinearity.
Missing values: No null values were found in the dataset. Rows where amount = 0 were dropped to ensure valid transactions.
Outliers: Large number of outliers detected (e.g., amount = 338k, oldbalanceOrg = 1.1M). Instead of removing all, engineered features were created (ratios, errors) to handle extreme values without discarding valuable fraud signals.
Multicollinearity: Highly correlated features were removed (newbalanceOrig, newbalanceDest, day, err_org). This avoided redundant information in modeling.

2.Describe your fraud detection model in elaboration.
Pipeline:
Preprocessing (drop IDs, one-hot encoding transaction type).
Feature engineering: delta_org, delta_dest, err_org, err_dest, ratios (amt_to_oldbal, amt_to_destold), time features (day, hour), merchant flag.
Feature selection: applied Lasso (L1), which reduced features to two strongest predictors → delta_org, amt_to_destold.
Train-test split with stratification, scaling with StandardScaler.
Models trained: Logistic Regression, Random Forest, XGBoost.
Best model: Random Forest, chosen for best trade-off between precision & recall.

3.How did you select variables to be included in the model?
Started with original + engineered features.
Removed constant and highly correlated variables.
Dropped weakly correlated features with fraud.
Applied Lasso regression, which shrank coefficients and retained only delta_org (68% importance) and amt_to_destold (32% importance).
Final model trained only on these two features.

4.Demonstrate the performance of the model by using best set of tools.
To demonstrate model performance on this highly imbalanced fraud dataset, Precision–Recall AUC, F1-score, and confusion matrices were used as primary evaluation tools. Although Logistic Regression and XGBoost achieved high ROC-AUC values, Random Forest outperformed them in PR-AUC (0.53) and F1-score (0.58), while detecting the highest number of fraud cases. Therefore, Random Forest is the most suitable model for deployment.

5.What are the key factors that predict fraudulent customer?
From Lasso selection + Random Forest importance + SHAP analysis, the two most predictive factors are: delta_org=difference between old balance and new balance of origin account (fraud often creates inconsistencies). amt_to_destold=ratio of transaction amount to old destination balance (suspicious when transfer is disproportionately large). delta_org contributes 68% and amt_to_destold 32% to fraud prediction.

6.Do these factors make sense? If yes, How? If not, How not?
Yes, they align with fraud patterns: Fraud often leaves mismatched balances (delta_org). Fraud transactions are large relative to account balances (amt_to_destold). These are consistent with domain knowledge: fraudsters exploit cash-out/transfer anomalies.

7.What kind of prevention should be adopted while company update its infrastructure?
Real-time monitoring of flagged transactions using Random Forest scores + rules.
Block multiple rapid transactions.
Device/IP fingerprinting for unusual access.
Continuous model retraining and drift monitoring.

8.Assuming these actions have been implemented, how would you determine if they work?
Track fraud KPIs:
   Reduction in fraud losses ($ value).
   Increase in recall (fraud caught) without drastic drop in precision.
   False positive rate (customer friction).
Re-check feature distributions, fraud patterns, recalibrate thresholds.
Success = lower fraud losses, stable customer experience, and adaptable system.

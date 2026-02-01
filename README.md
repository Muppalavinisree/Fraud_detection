##Fraud Detection Using Machine Learning
#Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning techniques.
The goal is to build an interpretable and high-performing fraud detection model on a highly imbalanced dataset, while preserving real fraud signals and minimizing false positives.

#Dataset & Problem Statement
Source: Kaggle (synthetic financial transaction data)
Challenge:
  Severe class imbalance
  Presence of extreme outliers
  Risk of multicollinearity
Objective:
  Accurately identify fraudulent transactions
  Prioritize recall and precision over raw accuracy

#Data Cleaning & Preprocessing
  Missing Values
  No null values were present in the dataset
  Transactions with amount = 0 were removed to ensure data validity

#Outliers
Significant outliers were detected (e.g., transaction amounts up to 338K, balances exceeding 1M)
Instead of removing all outliers, feature engineering was applied to preserve fraud-related signals:
   Balance deltas
   Ratio-based features
This approach avoids losing rare but meaningful fraud patterns

#Multicollinearity
Highly correlated variables were identified and removed:
   1)newbalanceOrig
   2)newbalanceDest
   3)day
   4)err_org
This reduced redundancy and improved model stability

#Feature Engineering
Created domain-driven features to capture fraud behavior:
delta_org: Difference between old and new origin balance
delta_dest: Difference between old and new destination balance
err_org, err_dest: Transaction inconsistencies
Ratio features:
   amt_to_oldbal
   amt_to_destold
Time-based features:
   Transaction hour, day
Merchant flag for transaction type

#Feature Selection
Started with original + engineered features
Removed:
 1)Constant variables
 2)Highly correlated features
 3)Weakly correlated predictors
Applied Lasso Regression (L1 Regularization):
  Reduced feature space to two strongest predictors:
  -delta_org → 68% importance
  -amt_to_destold → 32% importance
Final models were trained using only these two features for interpretability and robustness

#Model Pipeline
   Dropped identifiers and non-informative columns
   One-hot encoded transaction types
   Train-test split with stratification
   Feature scaling using StandardScaler

#Models trained:
  Logistic Regression
  Random Forest
  XGBoost

#Model Evaluation
 Given the imbalanced nature of fraud detection, the following metrics were prioritized:
      Precision–Recall AUC
      F1-score
      Confusion Matrix

#Results Summary
Model	                 PR-AUC	          F1-score	 Observation
Logistic Regression	  High ROC-AUC	    Moderate	 Missed many fraud cases
XGBoost	              High ROC-AUC    	Good	    Balanced but less recall
Random Forest	        0.53	            0.58	    Best fraud detection performance

Random Forest was selected due to its superior balance between precision and recall and highest fraud capture rate.

#Key Fraud Predictors
Based on Lasso selection, Random Forest importance, and SHAP analysis, the most influential features are:
->delta_org (68%)
   1)Inconsistencies between old and new balances
   2)Common indicator of fraudulent manipulation
->amt_to_destold (32%)
   1)Large transaction amounts relative to destination balance
   2)Signals abnormal transfer behavior
These factors align strongly with real-world fraud patterns.

#Do These Factors Make Sense?
Yes.
These predictors align with known financial fraud patterns such as balance inconsistencies and unusually large transfers.

#Fraud Prevention Recommendations
1)Real-time transaction monitoring using Random Forest risk scores
2)Velocity checks for multiple rapid transactions
3)Device and IP fingerprinting
4)Continuous model retraining and data drift monitoring

#Measuring Success After Deployment
Effectiveness can be evaluated using:
     Reduction in fraud-related financial losses
     Improved recall without significant precision drop
     Stable false positive rate to avoid customer friction
     Monitoring feature distribution shifts and recalibrating thresholds
Success = Lower fraud loss + stable customer experience + adaptive system

#Tools & Technologies
Python
Pandas, NumPy
scikit-learn
XGBoost
SHAP
Matplotlib, Seaborn

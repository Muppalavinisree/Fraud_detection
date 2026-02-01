# Fraud Detection Using Machine Learning

## Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning techniques.  
The objective is to build an interpretable and high-performing fraud detection model on a highly imbalanced dataset, while preserving rare fraud signals and minimizing false positives.

---

## Dataset & Problem Statement
- Source: Kaggle (synthetic financial transaction dataset)
- Key challenges:
  - Severe class imbalance
  - Presence of extreme outliers
  - Risk of multicollinearity
- Goal:
  - Accurately identify fraudulent transactions
  - Optimize precision and recall rather than overall accuracy

---

## Data Cleaning & Preprocessing

### Missing Values
- No null values were found in the dataset
- Transactions with `amount = 0` were removed to ensure valid financial activity

### Outliers
- Significant outliers were detected (e.g., transaction amounts up to 338K, balances exceeding 1.1M)
- Instead of removing all outliers, feature engineering was applied to preserve fraud-related signals:
  - Balance difference features
  - Ratio-based features
- This approach avoids discarding rare but valuable fraud patterns

### Multicollinearity
- Highly correlated variables were removed to reduce redundancy:
  - `newbalanceOrig`, `newbalanceDest`, `day`, `err_org`
- Improved model stability and interpretability

---

## Feature Engineering
Domain-driven features were created to capture fraud behavior:
- `delta_org`: Difference between old and new origin balance
- `delta_dest`: Difference between old and new destination balance
- `err_org`, `err_dest`: Transaction inconsistencies
- Ratio features:
  - `amt_to_oldbal`
  - `amt_to_destold`
- Time-based features:
  - Transaction hour and day
- Merchant flag derived from transaction type

---

## Feature Selection
- Started with original and engineered features
- Removed constant, redundant, and weakly correlated variables
- Applied Lasso Regression (L1 regularization) for feature selection:
  - Retained two strongest predictors:
    - `delta_org` (68% importance)
    - `amt_to_destold` (32% importance)
- Final models were trained using only these two features for robustness and interpretability

---

## Modeling Pipeline
1. Dropped identifiers and non-informative columns  
2. One-hot encoded transaction types  
3. Performed stratified train-test split  
4. Applied feature scaling using StandardScaler  
5. Trained the following models:
   - Logistic Regression  
   - Random Forest  
   - XGBoost  

---

## Model Evaluation
Due to the imbalanced nature of the dataset, the following metrics were prioritized:
- Precision–Recall AUC  
- F1-score  
- Confusion matrix  

### Performance Summary
Model                   PR-AUC           F1-score      Remarks 

Logistic Regression    High ROC-AUC     Moderate      Missed many fraud cases 
XGBoost                High ROC-AUC      Good         Balanced performance 
Random Forest           0.53             0.58         Best fraud detection capability 

Random Forest was selected as the final model due to its superior balance between precision and recall.

---

## Key Fraud Predictors
Based on Lasso selection, Random Forest feature importance, and SHAP analysis, the most influential predictors are:
- `delta_org` (68%): Inconsistencies between old and new balances
- `amt_to_destold` (32%): Unusually large transfers relative to destination balance

These predictors align with known financial fraud patterns.

---

## Fraud Prevention Recommendations
- Real-time transaction monitoring using Random Forest risk scores
- Velocity checks for rapid repeated transactions
- Device and IP fingerprinting
- Continuous model retraining and data drift monitoring

---

## Measuring Post-Deployment Effectiveness
Model effectiveness can be evaluated through:
- Reduction in fraud-related financial losses
- Improved fraud recall without significant precision drop
- Controlled false positive rate to reduce customer friction
- Monitoring feature distribution shifts and recalibrating thresholds

---

## Tools & Technologies
- Python  
- Pandas, NumPy  
- scikit-learn  
- XGBoost  
- SHAP  
- Matplotlib, Seaborn  

---

## Key Takeaways
- Handles real-world imbalanced fraud data
- Focuses on both interpretability and performance
- Demonstrates a complete end-to-end machine learning pipeline
- Strong alignment with industry fraud detection practices



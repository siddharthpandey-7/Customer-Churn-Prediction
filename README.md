# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview
Customer churn is a major challenge for subscription-based businesses like telecom companies. This project builds an end-to-end machine learning pipeline to predict whether a customer is likely to churn, using customer demographics, service usage, and billing information.

The focus of this project is correct ML engineering practices, including proper preprocessing, class imbalance handling, and leakage-free model training.

## 🎯 Problem Statement
The goal is to predict customer churn (Yes/No) so that businesses can proactively retain high-risk customers and reduce revenue loss.

Churn prediction is a binary classification problem with imbalanced classes, making careful evaluation and modeling essential.

## 📂 Dataset Description
* Dataset Size: 7,043 customers
* Features: 21 input features + 1 target variable
* Target Variable: `Churn`
   * `0` → Stayed
   * `1` → Churned

### 🔹 Key Feature Types
* Categorical: gender, SeniorCitizen, Contract, PaymentMethod, InternetService, etc.
* Numerical: tenure, MonthlyCharges, TotalCharges

The dataset is moderately imbalanced, with approximately 26% churned customers.

## 📈 Exploratory Data Analysis (EDA) – Key Insights
* Customers with short tenure are more likely to churn.
* Month-to-month contracts have significantly higher churn rates.
* Customers with higher monthly charges tend to churn more.
* Long-term contracts (1-year, 2-year) show strong customer retention.

These insights align well with real-world telecom business behavior.

## ⚙️ Machine Learning Pipeline
A fully automated scikit-learn pipeline was used to ensure clean, reproducible, and leakage-free training.

### Pipeline Steps:
1. Preprocessing
   * Numerical features → `StandardScaler`
   * Categorical features → `OneHotEncoder (handle_unknown='ignore')`
2. Class Imbalance Handling
   * `SMOTE` applied inside the pipeline
3. Model Training
   * Logistic Regression (baseline)
   * XGBoost (final model)

This approach prevents data leakage and ensures consistent preprocessing during inference.

## 🤖 Models Used
| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline model |
| XGBoost Classifier | Final selected model |

XGBoost was selected due to its better performance and ability to capture non-linear relationships.

## 📊 Model Evaluation
Evaluation was performed on a held-out test set using appropriate metrics for imbalanced classification.

### 🔹 Metrics Used
* ROC-AUC Score
* Precision, Recall, F1-score
* Accuracy

### 🔹 Final Performance (XGBoost)
* ROC-AUC: ~0.82
* Accuracy: ~0.78
* Strong balance between precision and recall for churn prediction

ROC-AUC was prioritized because it measures the model's ability to rank churners higher than non-churners, which is crucial in business scenarios.

## 🧪 Sample Prediction
The trained pipeline can directly make predictions on new customer data:

```
Prediction: Churn
Probability: 0.89
```

This output can be used to trigger retention strategies such as targeted offers or customer outreach.

## 🛠️ Tech Stack
* Programming Language: Python
* Libraries:
   * pandas, numpy
   * matplotlib, seaborn
   * scikit-learn
   * imbalanced-learn (SMOTE)
   * xgboost
* Model Persistence: joblib

## 📌 Key Highlights
* End-to-end pipeline-based ML project
* Proper handling of categorical features
* SMOTE used correctly inside pipeline
* Leakage-free preprocessing and evaluation
* Production-ready saved model
* Business-aligned evaluation metrics

## 🚀 Future Improvements
* Add model explainability using SHAP
* Deploy the model using FastAPI
* Perform cost-based threshold tuning
* Monitor model performance in production

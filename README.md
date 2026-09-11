# 📊 Customer Churn Prediction using Machine Learning

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses such as telecom companies. This project develops an end-to-end machine learning solution to predict whether a customer is likely to churn based on demographic, service, contract, and billing information.

The project covers data preprocessing, class imbalance handling, model training and evaluation, model persistence, and deployment through a Flask web application.

---

## 🎯 Problem Statement

The goal is to predict whether a customer will **Stay** or **Churn**, helping businesses identify customers who may be at higher risk of leaving and take proactive retention measures.

This is a binary classification problem with an imbalanced target distribution.

---

## 📂 Dataset

The dataset contains **7,043 customer records** and 22 original columns. The `Unnamed: 0` and `customerID` columns are removed before modeling, leaving **19 input features** and the `Churn` target.

### Dataset Summary

- **Records:** 7,043
- **Original columns:** 22
- **Input features:** 19
- **Target:** `Churn`
- **Stayed:** 5,174
- **Churned:** 1,869
- **Churn rate:** ~26.5%

### Target Encoding

- `0` → Stayed
- `1` → Churned

### Feature Types

**Numerical**
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

**Categorical**
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

---

## 📈 Exploratory Data Analysis

EDA was performed to understand customer characteristics and their relationship with churn.

### Key Insights

- Customers with shorter tenure show a higher tendency to churn.
- Higher monthly charges are associated with higher churn.
- Month-to-month contracts show substantially more churn than longer-term contracts.
- Longer-term contracts are associated with stronger customer retention in the dataset.
- Tenure shows a negative correlation with churn, while monthly charges show a positive correlation.

---

## ⚙️ Machine Learning Pipeline

A reproducible pipeline was developed using **scikit-learn** and **imbalanced-learn**.

### Pipeline

```text
Input Data
    │
    ▼
ColumnTransformer
    │
    ├── Numerical Features
    │      ├── Median Imputation
    │      └── StandardScaler
    │
    └── Categorical Features
           ├── Most-Frequent Imputation
           └── OneHotEncoder
                    │
                    ▼
                  SMOTE
                    │
                    ▼
                  Model
```

### Preprocessing

- Missing numerical values are handled using median imputation.
- Missing categorical values are handled using most-frequent imputation.
- Numerical features are standardized using StandardScaler.
- Categorical features are encoded using OneHotEncoder.
- `handle_unknown='ignore'` allows the pipeline to handle unseen categorical values during prediction.

### Class Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) is applied within the training pipeline after the train-test split. This ensures that synthetic samples are generated only during model training and helps prevent information from the held-out test set from influencing the resampling process.

---

## 🤖 Models Used

Two classification models were trained and evaluated:

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline model |
| XGBoost Classifier | Model used in the Flask application |

The models were compared using multiple evaluation metrics rather than relying on a single score.

---

## 📊 Model Evaluation

Both models were evaluated on a held-out test set containing 1,409 customers.

### Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Model Comparison

| Model | Accuracy | ROC-AUC | Churn Precision | Churn Recall | Churn F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.74 | 0.8399 | 0.51 | 0.80 | 0.62 |
| XGBoost | 0.78 | 0.8164 | 0.59 | 0.61 | 0.60 |

### Results

The models show different strengths:

- Logistic Regression achieved higher ROC-AUC and churn recall.
- XGBoost achieved higher overall accuracy and churn precision.
- The XGBoost pipeline was selected for the Flask application.

This demonstrates the importance of evaluating classification models using multiple metrics, particularly when working with imbalanced data.

---

## 🧪 Example Prediction

The trained XGBoost pipeline was saved using Joblib and used to make predictions on new customer data.

### Example Output

```
Customer is likely to CHURN
Probability: 0.89
```

The example customer received a predicted churn probability of approximately 0.89.

---

## 🌐 Web Application

A Flask-based web application was developed to provide an easy-to-use interface for churn prediction.

The application collects a selected set of customer details:

- Gender
- Senior Citizen status
- Tenure
- Monthly Charges
- Total Charges
- Contract type
- Tech Support
- Payment Method

To keep the form simple and user-friendly, the remaining model features are assigned predefined values by the application before prediction.

> **Note:** Predictions from the simplified interface are based partly on these predefined feature values.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Matplotlib
- Seaborn
- Joblib
- Flask
- HTML/CSS

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── templates/
│   └── index.html
│
├── .gitignore
├── app.py
├── churn_prediction_pipeline.joblib
├── Customer_Churn_Prediction.ipynb
├── README.md
├── requirements.txt
└── telco.csv
```

### File Description

| File | Description |
|---|---|
| `Customer_Churn_Prediction.ipynb` | Data analysis, preprocessing, model training, and evaluation |
| `app.py` | Flask application for churn prediction |
| `templates/index.html` | Web interface |
| `churn_prediction_pipeline.joblib` | Saved XGBoost prediction pipeline |
| `telco.csv` | Dataset |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Ignored files and directories |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Customer-Churn-Prediction
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

Open the local address provided by Flask in your browser and enter the customer details to generate a churn prediction.

---

## 💾 Model Persistence

The trained XGBoost pipeline is saved using Joblib:

```python
joblib.dump(xgb_pipeline, "churn_prediction_pipeline.joblib")
```

The saved pipeline can be loaded by the Flask application to perform predictions without retraining the model.

---

## 📌 Key Highlights

- End-to-end customer churn prediction solution
- 19 input features
- Data cleaning and missing-value handling
- Numerical scaling and categorical encoding
- SMOTE-based class imbalance handling
- Stratified train-test split
- Leakage-aware training pipeline
- Logistic Regression baseline
- XGBoost model
- Multi-metric model evaluation
- Saved ML pipeline using Joblib
- Flask-based prediction application
- Simplified user-friendly input interface

---

## 🚀 Future Improvements

- Add SHAP-based model explainability
- Perform business-focused threshold tuning
- Improve prediction explanations in the web application
- Explore hyperparameter tuning
- Deploy using a production WSGI server
- Add model monitoring and performance tracking

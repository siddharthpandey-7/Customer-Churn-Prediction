# 📊 Customer Churn Prediction using Machine Learning

## 📌 Overview

This project develops an end-to-end machine learning solution to predict whether a customer is likely to **Churn** or **Stay** based on demographic, service, contract, and billing information.

The project includes data preprocessing, exploratory data analysis, class imbalance handling with SMOTE, model training and evaluation, model persistence, and a Flask-based web application for making predictions.

## 🎯 Problem Statement

Customer churn is an important challenge for subscription-based businesses. The objective of this project is to predict customers who are likely to churn, helping businesses identify higher-risk customers and take proactive retention measures.

This is a binary classification problem with an imbalanced target distribution.

## 📂 Dataset

The dataset contains **7,043 customer records** and **22 original columns**. The `Unnamed: 0` and `customerID` columns are removed before modeling, leaving **19 input features** and the `Churn` target.

| Attribute | Value |
|---|---:|
| Records | 7,043 |
| Original columns | 22 |
| Input features | 19 |
| Stayed | 5,174 |
| Churned | 1,869 |
| Churn rate | ~26.5% |

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

## 📈 Exploratory Data Analysis

EDA was performed to understand customer characteristics and their relationship with churn.

### Key Insights

- Customers with shorter tenure tend to have a higher tendency to churn.
- Higher monthly charges are associated with higher churn.
- Month-to-month contracts show substantially more churn than longer-term contracts.
- Longer-term contracts are associated with stronger retention in the dataset.
- Tenure shows a negative correlation with churn, while monthly charges show a positive correlation.

## ⚙️ Machine Learning Approach

An end-to-end pipeline was built using **scikit-learn** and **imbalanced-learn**.

### Train-Test Split

An **80/20 stratified train-test split** was used to preserve the class distribution between the training and test sets.

### Preprocessing

**Numerical features**

- Median imputation for missing values
- Standardization using `StandardScaler`

**Categorical features**

- Most-frequent imputation for missing values
- One-hot encoding using `OneHotEncoder`
- First category dropped using `drop='first'`
- `handle_unknown='ignore'` for unseen categories

### Class Imbalance

**SMOTE (Synthetic Minority Over-sampling Technique)** is applied within the training pipeline after the train-test split. This ensures that synthetic samples are generated only during model training and that the held-out test set is not resampled.

### Pipeline

```text
Input Data
    │
    ▼
ColumnTransformer
    │
    ├── Numerical → Imputation → StandardScaler
    │
    └── Categorical → Imputation → OneHotEncoder
                              │
                              ▼
                            SMOTE
                              │
                              ▼
                            Model
```

## 🤖 Models

Two classification models were trained and evaluated:

| Model | Role |
|---|---|
| Logistic Regression | Baseline model |
| XGBoost Classifier | Model used in the Flask application |

## 📊 Model Evaluation

The models were evaluated on a held-out test set of 1,409 customers using accuracy, precision, recall, F1-score, and ROC-AUC.

| Model | Accuracy | ROC-AUC | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.74 | 0.8399 | 0.51 | 0.80 | 0.62 |
| XGBoost | 0.78 | 0.8164 | 0.59 | 0.61 | 0.60 |

### Results

- Logistic Regression achieved higher ROC-AUC and churn recall.
- XGBoost achieved higher overall accuracy and churn precision.
- The XGBoost pipeline was selected for the Flask application.

This comparison highlights the importance of evaluating classification models using multiple metrics, particularly for imbalanced datasets.

## 💾 Model Persistence

The trained XGBoost pipeline is saved using Joblib:

```python
joblib.dump(xgb_pipeline, "churn_prediction_pipeline.joblib")
```

The saved pipeline can be loaded by the Flask application to make predictions without retraining the model.

## 🌐 Flask Web Application

A Flask-based web application provides a simple interface for generating churn predictions.

The application collects:

- Gender
- Senior Citizen status
- Tenure
- Monthly Charges
- Total Charges
- Contract type
- Tech Support
- Payment Method

To keep the interface simple and user-friendly, the remaining model features are assigned predefined values by the application before prediction.

> **Note:** Predictions from the simplified interface are partly based on these predefined feature values.

### Example Output

```
Customer is likely to CHURN
Probability: 0.89
```

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Joblib
- Flask
- Matplotlib
- Seaborn
- Jupyter Notebook
- HTML/CSS

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

| File | Description |
|---|---|
| `Customer_Churn_Prediction.ipynb` | Data analysis, preprocessing, model training, and evaluation |
| `app.py` | Flask application for churn prediction |
| `templates/index.html` | Web application interface |
| `churn_prediction_pipeline.joblib` | Saved XGBoost prediction pipeline |
| `telco.csv` | Dataset |
| `requirements.txt` | Project dependencies |
| `.gitignore` | Ignored files and directories |

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

## ▶️ Run the Application

```bash
python app.py
```

Open the local URL shown by Flask in your browser and enter the customer details to generate a churn prediction.

## 🔮 Future Improvements

- Add SHAP-based model explainability.
- Perform business-focused prediction threshold tuning.
- Improve prediction explanations in the web application.
- Explore hyperparameter tuning.
- Deploy using a production WSGI server.
- Add model monitoring and performance tracking.
- 

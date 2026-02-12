from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained ML pipeline
model = joblib.load("churn_prediction_pipeline.joblib")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        data = {
            'gender': request.form['gender'],
            'SeniorCitizen': request.form['SeniorCitizen'],
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': int(request.form['tenure']),
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': request.form['Contract'],
            'PaperlessBilling': 'Yes',
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        df = pd.DataFrame([data])

        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        prediction = "Churn" if pred == 1 else "Stay"
        probability = round(prob, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

# REQUIRED for Render (free deployment)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

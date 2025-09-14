from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load trained pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_pipeline.joblib")
model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return "✅ Customer Churn Prediction API is running!"

# Original column names used during training
expected_features = [
    'gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
    'StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
    'tenure','MonthlyCharges','TotalCharges'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        # Fill missing columns with default values
        for col in expected_features:
            if col not in df.columns:
                df[col] = 0

        # Convert numeric columns to float
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Predict using the pipeline
        prediction = model.predict(df)[0]

        return jsonify({"churn_prediction": int(prediction)})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

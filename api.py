# api.py
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

def feature_engineering(df):
    df["inflow_outflow_ratio"] = df["mpesa_inflow_freq"] / (df["mpesa_outflow_freq"] + 1)
    df["airtime_per_topup"] = df["avg_airtime"] / (df["airtime_topup_count"] + 1)
    df["late_bills_ratio"] = df["utility_bills_paid_late"] / (df["utility_bills_total"] + 1)
    return df

# Load your trained model
model = joblib.load("final_credit_model.joblib")

FEATURES = [
    "mpesa_txn_count", "avg_mpesa_amount", "mpesa_inflow_freq", "mpesa_outflow_freq",
    "airtime_topup_count", "avg_airtime", "utility_bills_total", "utility_bills_paid_late",
    "inflow_outflow_ratio", "airtime_per_topup", "late_bills_ratio"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = feature_engineering(df)
    X = df[FEATURES]
    proba = model.predict_proba(X)[0, 1]
    pred = int(model.predict(X)[0])
    return jsonify({
        "probability_good": float(proba),
        "probability_bad": float(1 - proba),
        "prediction": pred
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

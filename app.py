"""
app.py
Flask web app for the Personalised Predictive Healthcare demo.

Routes:
  GET  /            -> HTML form
  POST /predict      -> runs the ML model + GenAI explainer, returns JSON
  GET  /health        -> simple healthcheck (used by Render)
"""
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

from genai_explainer import generate_explanation

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "risk_model.joblib")
_bundle = joblib.load(MODEL_PATH)
MODEL = _bundle["model"]
FEATURES = _bundle["features"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        patient = {f: float(data[f]) for f in FEATURES}
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid or missing input: {e}"}), 400

    X = np.array([[patient[f] for f in FEATURES]])
    probability = float(MODEL.predict_proba(X)[0, 1])

    importances = MODEL.feature_importances_
    factors = sorted(zip(FEATURES, importances), key=lambda x: -x[1])

    explanation, source = generate_explanation(probability, factors, patient)

    return jsonify({
        "risk_probability": round(probability, 4),
        "risk_level": "High" if probability >= 0.66 else ("Moderate" if probability >= 0.33 else "Low"),
        "top_factors": [f[0] for f in factors[:3]],
        "explanation": explanation,
        "explanation_source": source,  # "llm" or "template" -- shows which path was used
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)

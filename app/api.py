"""
=============================================================
  DISEASE PREDICTOR - WEB API (Flask)
=============================================================
  Provides a REST API for the disease predictor.

  Install Flask:
      pip install flask

  Run:
      python app/api.py

  Endpoints:
      GET  /symptoms          → list of all valid symptoms
      GET  /diseases          → list of all detectable diseases
      POST /predict           → predict disease from symptoms

  POST /predict body (JSON):
      { "symptoms": ["fever", "headache", "nausea"] }

  Response:
      {
        "disease": "Malaria",
        "confidence": 97.3,
        "description": "...",
        "precautions": [...],
        "medications": "...",
        "diet": "...",
        "workout": [...]
      }
=============================================================
"""

import pickle
import zipfile
import pandas as pd
import numpy as np
import os
import sys

from flask import Flask, request, jsonify, send_from_directory

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/ui")
def serve_ui():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")

# ── Load everything at startup ────────────────────────────────
print("Loading model artifacts...", end="", flush=True)
try:
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        MODEL = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        LE = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
        FEATURE_COLS = pickle.load(f)
    print(" ✅")
except FileNotFoundError:
    print("\n❌ Models not found. Run 'python train_model.py' first.")
    sys.exit(1)

DESC    = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
PREC    = pd.read_csv(os.path.join(DATA_DIR, "precautions_df.csv"))
MEDS    = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
DIETS   = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
WORKOUT = pd.read_csv(os.path.join(DATA_DIR, "workout_df.csv"))

COL_MAP = {c.strip().lower(): i for i, c in enumerate(FEATURE_COLS)}

def norm(s):
    return str(s).strip().lower()

def get_info(disease_name):
    dn = norm(disease_name)
    description, precautions, medications, diet, workout = "", [], "", "", []

    row = DESC[DESC['Disease'].apply(norm) == dn]
    description = row.iloc[0]['Description'] if not row.empty else "Not available."

    row = PREC[PREC['Disease'].apply(norm) == dn]
    if not row.empty:
        precautions = [str(row.iloc[0][f'Precaution_{i}']) for i in range(1, 5)
                       if pd.notna(row.iloc[0].get(f'Precaution_{i}'))]

    row = MEDS[MEDS['Disease'].apply(norm) == dn]
    medications = row.iloc[0]['Medication'] if not row.empty else "Not available."

    row = DIETS[DIETS['Disease'].apply(norm) == dn]
    diet = row.iloc[0]['Diet'] if not row.empty else "Not available."

    row = WORKOUT[WORKOUT['disease'].apply(norm) == dn]
    workout = row['workout'].tolist() if not row.empty else []

    return description, precautions, medications, diet, workout


# ── API Routes ────────────────────────────────────────────────

@app.route("/symptoms", methods=["GET"])
def get_symptoms():
    return jsonify({"symptoms": sorted(FEATURE_COLS), "count": len(FEATURE_COLS)})


@app.route("/diseases", methods=["GET"])
def get_diseases():
    return jsonify({"diseases": sorted(LE.classes_.tolist()), "count": len(LE.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "symptoms" not in data:
        return jsonify({"error": "Please provide 'symptoms' as a list."}), 400

    symptoms_input = data["symptoms"]
    if not isinstance(symptoms_input, list) or len(symptoms_input) == 0:
        return jsonify({"error": "'symptoms' must be a non-empty list."}), 400

    # Build input vector
    input_vector = np.zeros(len(FEATURE_COLS))
    matched = []
    unmatched = []

    for sym in symptoms_input:
        key = sym.strip().lower().replace(" ", "_")
        if key in COL_MAP:
            input_vector[COL_MAP[key]] = 1
            matched.append(key)
        else:
            unmatched.append(sym)

    if not matched:
        return jsonify({
            "error": "No valid symptoms recognized.",
            "unrecognized": unmatched,
            "hint": "Call GET /symptoms to see valid symptom names."
        }), 400

    X = input_vector.reshape(1, -1)
    pred_encoded = MODEL.predict(X)[0]
    disease = LE.inverse_transform([pred_encoded])[0]

    confidence = 0.0
    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        confidence = round(float(proba[pred_encoded]) * 100, 2)

    description, precautions, medications, diet, workout = get_info(disease)

    return jsonify({
        "disease": disease,
        "confidence_pct": confidence,
        "matched_symptoms": matched,
        "unrecognized_symptoms": unmatched,
        "description": description,
        "precautions": precautions,
        "medications": medications,
        "diet": diet,
        "workout": workout,
        "disclaimer": "This is an AI prediction tool for educational purposes only. Consult a qualified doctor."
    })


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Disease Predictor API",
        "endpoints": {
            "GET /symptoms": "List all valid symptoms",
            "GET /diseases": "List all detectable diseases",
            "POST /predict": "Predict disease. Body: {'symptoms': ['fever','headache']}"
        }
    })


if __name__ == "__main__":
    print("Starting Disease Predictor API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)

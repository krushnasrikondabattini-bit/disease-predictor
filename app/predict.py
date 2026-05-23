"""
=============================================================
  DISEASE PREDICTOR - COMMAND LINE APP
=============================================================
  Predicts disease from symptoms and shows:
    - Predicted Disease
    - Disease Description
    - Precautions
    - Medications
    - Recommended Diet
    - Workout Suggestions

  Usage:
      python app/predict.py
=============================================================
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load saved model artifacts ────────────────────────────────
def load_artifacts():
    try:
        with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
            le = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
            feature_columns = pickle.load(f)
        return model, le, feature_columns
    except FileNotFoundError:
        print("❌ Model not found! Please run 'python train_model.py' first.")
        sys.exit(1)

# ── Load reference CSVs ────────────────────────────────────────
def load_reference_data():
    desc    = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
    prec    = pd.read_csv(os.path.join(DATA_DIR, "precautions_df.csv"))
    meds    = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
    diets   = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
    workout = pd.read_csv(os.path.join(DATA_DIR, "workout_df.csv"))
    return desc, prec, meds, diets, workout

# ── Helper: lookup info for a disease ─────────────────────────
def get_disease_info(disease_name, desc, prec, meds, diets, workout):
    # Normalize disease name for matching
    def norm(s):
        return str(s).strip().lower()
    dn = norm(disease_name)

    description = "Not available."
    row = desc[desc['Disease'].apply(norm) == dn]
    if not row.empty:
        description = row.iloc[0]['Description']

    precautions = []
    row = prec[prec['Disease'].apply(norm) == dn]
    if not row.empty:
        precautions = [str(row.iloc[0][f'Precaution_{i}']) for i in range(1, 5)
                       if pd.notna(row.iloc[0][f'Precaution_{i}'])]

    medications = "Not available."
    row = meds[meds['Disease'].apply(norm) == dn]
    if not row.empty:
        medications = row.iloc[0]['Medication']

    diet = "Not available."
    row = diets[diets['Disease'].apply(norm) == dn]
    if not row.empty:
        diet = row.iloc[0]['Diet']

    workouts = []
    row = workout[workout['disease'].apply(norm) == dn]
    if not row.empty:
        workouts = row['workout'].tolist()

    return description, precautions, medications, diet, workouts

# ── Predict disease from symptom list ─────────────────────────
def predict_disease(symptoms_input, model, le, feature_columns):
    # Build binary feature vector
    input_vector = np.zeros(len(feature_columns))
    matched = []
    unmatched = []

    clean_symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_input]
    col_map = {c.strip().lower(): i for i, c in enumerate(feature_columns)}

    for sym in clean_symptoms:
        if sym in col_map:
            input_vector[col_map[sym]] = 1
            matched.append(sym)
        else:
            unmatched.append(sym)

    if not matched:
        return None, [], 0.0

    X = input_vector.reshape(1, -1)
    pred_encoded = model.predict(X)[0]
    disease = le.inverse_transform([pred_encoded])[0]

    # Confidence score (if model supports it)
    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = proba[pred_encoded] * 100

    return disease, unmatched, confidence

# ── Pretty print section ──────────────────────────────────────
def print_section(title, content, bullet=False):
    print(f"\n  {'─'*50}")
    print(f"  📌 {title}")
    print(f"  {'─'*50}")
    if isinstance(content, list):
        if bullet:
            for item in content:
                print(f"     • {item}")
        else:
            for item in content:
                print(f"     {item}")
    else:
        print(f"     {content}")

# ── Main interactive loop ──────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  🏥  DISEASE PREDICTOR  ")
    print("="*60)
    print("  Enter your symptoms and get a disease prediction with")
    print("  precautions, medications, diet, and workout tips.")
    print("="*60)

    # Load model
    print("\n  Loading model...", end="", flush=True)
    model, le, feature_columns = load_artifacts()
    print(" ✅")

    # Load reference data
    desc, prec, meds, diets, workout = load_reference_data()

    print(f"\n  Available symptoms: {len(feature_columns)}")
    print(f"  Detectable diseases: {len(le.classes_)}")
    print("\n  TIP: Enter symptom names separated by commas.")
    print("       Example: fever, headache, nausea, fatigue")
    print("  Type 'symptoms' to see all available symptoms.")
    print("  Type 'quit' to exit.\n")

    while True:
        print("─" * 60)
        user_input = input("  Enter symptoms: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("  Goodbye! Stay healthy! 💪")
            break
        if user_input.lower() == "symptoms":
            print("\n  All available symptoms:")
            for i, sym in enumerate(sorted(feature_columns), 1):
                print(f"    {i:3}. {sym}")
            continue

        # Parse symptoms
        symptoms_list = [s.strip() for s in user_input.split(",") if s.strip()]

        # Predict
        disease, unmatched, confidence = predict_disease(
            symptoms_list, model, le, feature_columns
        )

        if disease is None:
            print("\n  ⚠️  No valid symptoms recognized. Please check spelling.")
            print("     Type 'symptoms' to see the full list.")
            continue

        if unmatched:
            print(f"\n  ⚠️  Unrecognized symptoms (ignored): {', '.join(unmatched)}")

        # Get full info
        description, precautions, medications, diet, workouts = get_disease_info(
            disease, desc, prec, meds, diets, workout
        )

        # Display results
        print(f"\n{'='*60}")
        conf_str = f"  (Confidence: {confidence:.1f}%)" if confidence > 0 else ""
        print(f"  🔬  PREDICTED DISEASE: {disease.upper()}{conf_str}")
        print(f"{'='*60}")

        print_section("Description", description)
        print_section("Precautions", precautions if precautions else ["No precautions data available."], bullet=True)
        print_section("Medications", medications)
        print_section("Recommended Diet", diet)
        print_section("Workout / Lifestyle Tips", workouts if workouts else ["No workout data available."], bullet=True)

        print(f"\n  {'─'*50}")
        print("  ⚠️  DISCLAIMER: This is an AI prediction tool for")
        print("     educational purposes only. Please consult a")
        print("     qualified doctor for medical advice.")
        print(f"  {'─'*50}\n")

if __name__ == "__main__":
    main()

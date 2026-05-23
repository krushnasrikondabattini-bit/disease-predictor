"""
=============================================================
  DISEASE PREDICTION - DATASET EXPANDER
=============================================================
  Expands your existing dataset with 25 new diseases,
  39 new symptom columns, and updates all supporting CSVs.

  Run this ONCE before retraining:
      python expand_dataset.py
      python train_model.py

  What changes:
      data/Training.csv         ← 41 → 66 diseases, 132 → 171 symptoms
      data/description.csv      ← +25 disease descriptions
      data/precautions_df.csv   ← +25 disease precautions
      data/medications.csv      ← +25 disease medications
      data/diets.csv            ← +25 disease diets
      data/workout_df.csv       ← +25 disease workout tips
      data/Symptom-severity.csv ← +39 new symptom severities
=============================================================
"""

import zipfile
import pandas as pd
import numpy as np
import os

np.random.seed(42)

DATA_DIR = "data"
N_SAMPLES = 120   # samples per new disease (matches original balance)

# ──────────────────────────────────────────────────────────────
# STEP 1: LOAD EXISTING Training.csv
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  DISEASE DATASET EXPANDER")
print("=" * 60)
print("\n[1/7] Loading existing Training.csv...")

TRAIN_PATH = os.path.join(DATA_DIR, "Training.csv")
try:
    with zipfile.ZipFile(TRAIN_PATH, 'r') as z:
        with z.open('Training.csv') as f:
            df_existing = pd.read_csv(f)
    print("    Loaded from nested-zip format.")
except Exception:
    df_existing = pd.read_csv(TRAIN_PATH)
    print("    Loaded as plain CSV.")

# Strip whitespace from column names (the original has some dirty names)
df_existing.columns = [c.strip() for c in df_existing.columns]

existing_symptom_cols = list(df_existing.columns[:-1])   # all except 'prognosis'
print(f"    Existing rows     : {len(df_existing)}")
print(f"    Existing diseases : {df_existing['prognosis'].nunique()}")
print(f"    Existing symptoms : {len(existing_symptom_cols)}")

# ──────────────────────────────────────────────────────────────
# STEP 2: NEW SYMPTOM COLUMNS TO ADD
# ──────────────────────────────────────────────────────────────
print("\n[2/7] Defining new symptom columns...")

NEW_SYMPTOMS = [
    "loss_of_taste",
    "oxygen_saturation_drop",
    "blood_in_urine",
    "flank_pain",
    "hair_loss",
    "cold_intolerance",
    "facial_pain_pressure",
    "post_nasal_drip",
    "bad_breath",
    "swollen_tonsils",
    "ear_pain",
    "discharge_from_ear",
    "hearing_loss",
    "eye_discharge",
    "eye_pain",
    "crusting_of_eyelids",
    "sensitivity_to_light",
    "burning_sensation_in_skin",
    "nerve_pain",
    "one_sided_rash",
    "knee_swelling",
    "tophi",
    "bloating",
    "mucus_in_stool",
    "sudden_severe_headache",
    "widespread_pain",
    "tender_points",
    "fatty_stool",
    "irregular_periods",
    "pelvic_pain",
    "facial_hair_growth",
    "butterfly_rash",
    "photosensitivity_rash",
    "numbness_in_feet",
    "tremors",
    "stiff_muscles",
    "memory_loss",
    "seizures",
    "confusion",
]

# Filter to only truly new ones
NEW_SYMPTOMS = [s for s in NEW_SYMPTOMS if s not in existing_symptom_cols]
all_symptom_cols = existing_symptom_cols + NEW_SYMPTOMS

# Add new columns (all 0) to existing data
for sym in NEW_SYMPTOMS:
    df_existing[sym] = 0

print(f"    New symptoms added : {len(NEW_SYMPTOMS)}")
print(f"    Total symptoms now : {len(all_symptom_cols)}")

# ──────────────────────────────────────────────────────────────
# STEP 3: NEW DISEASE SYMPTOM PROFILES
# ──────────────────────────────────────────────────────────────
# Each entry maps symptom_name → probability of being 1 in a training sample.
# Higher = that symptom appears more consistently for this disease.
# Uses both existing column names AND new column names.
# ──────────────────────────────────────────────────────────────
print("\n[3/7] Defining new disease profiles...")

NEW_DISEASES = {

    "COVID-19": {
        "fatigue": 0.92, "cough": 0.88, "headache": 0.82,
        "loss_of_smell": 0.80, "loss_of_taste": 0.78,
        "breathlessness": 0.65, "muscle_pain": 0.72,
        "throat_irritation": 0.70, "mild_fever": 0.58,
        "high_fever": 0.42, "runny_nose": 0.55,
        "chills": 0.52, "nausea": 0.40,
        "diarrhoea": 0.36, "oxygen_saturation_drop": 0.32,
        "chest_pain": 0.28, "restlessness": 0.38,
        "body_ache": 0.00,   # placeholder kept for clarity
    },

    "Appendicitis": {
        "abdominal_pain": 0.95, "nausea": 0.88,
        "vomiting": 0.75, "high_fever": 0.68,
        "loss_of_appetite": 0.85, "belly_pain": 0.90,
        "restlessness": 0.62, "constipation": 0.38,
        "diarrhoea": 0.28, "bloating": 0.42,
        "malaise": 0.52, "sweating": 0.48,
    },

    "Kidney Stones": {
        "back_pain": 0.92, "flank_pain": 0.90,
        "burning_micturition": 0.80, "blood_in_urine": 0.75,
        "nausea": 0.78, "vomiting": 0.65,
        "sweating": 0.60, "restlessness": 0.70,
        "mild_fever": 0.42, "abdominal_pain": 0.52,
        "bladder_discomfort": 0.60,
    },

    "Iron Deficiency Anemia": {
        "fatigue": 0.95, "lethargy": 0.88,
        "breathlessness": 0.80, "headache": 0.72,
        "cold_hands_and_feets": 0.70, "hair_loss": 0.72,
        "cold_intolerance": 0.65, "fast_heart_rate": 0.58,
        "weight_loss": 0.48, "malaise": 0.55,
        "yellowish_skin": 0.28, "lack_of_concentration": 0.58,
        "dizziness": 0.60, "weakness_in_limbs": 0.55,
    },

    "Anxiety Disorder": {
        "anxiety": 0.95, "restlessness": 0.90,
        "palpitations": 0.82, "sweating": 0.70,
        "headache": 0.75, "fatigue": 0.68,
        "muscle_pain": 0.58, "breathlessness": 0.62,
        "chest_pain": 0.55, "irritability": 0.75,
        "nausea": 0.42, "mood_swings": 0.52,
        "lack_of_concentration": 0.60, "dizziness": 0.48,
    },

    "Clinical Depression": {
        "depression": 0.95, "fatigue": 0.90,
        "lethargy": 0.85, "weight_loss": 0.68,
        "mood_swings": 0.82, "loss_of_appetite": 0.75,
        "irritability": 0.70, "anxiety": 0.62,
        "lack_of_concentration": 0.80, "restlessness": 0.58,
        "headache": 0.52, "malaise": 0.48,
        "confusion": 0.35,
    },

    "Chronic Sinusitis": {
        "congestion": 0.92, "sinus_pressure": 0.88,
        "headache": 0.85, "runny_nose": 0.82,
        "facial_pain_pressure": 0.80, "post_nasal_drip": 0.78,
        "cough": 0.68, "throat_irritation": 0.62,
        "bad_breath": 0.52, "mild_fever": 0.42,
        "fatigue": 0.58, "loss_of_smell": 0.48,
        "malaise": 0.40,
    },

    "Tonsillitis": {
        "throat_irritation": 0.95, "swollen_tonsils": 0.92,
        "high_fever": 0.85, "headache": 0.70,
        "fatigue": 0.72, "patches_in_throat": 0.80,
        "swelled_lymph_nodes": 0.75, "bad_breath": 0.62,
        "nausea": 0.48, "malaise": 0.58,
        "chills": 0.42, "loss_of_appetite": 0.52,
    },

    "Ear Infection": {
        "ear_pain": 0.95, "headache": 0.72,
        "discharge_from_ear": 0.70, "hearing_loss": 0.68,
        "high_fever": 0.62, "fatigue": 0.68,
        "irritability": 0.58, "nausea": 0.42,
        "loss_of_balance": 0.38, "malaise": 0.52,
        "chills": 0.32, "dizziness": 0.45,
    },

    "Conjunctivitis": {
        "redness_of_eyes": 0.95, "watering_from_eyes": 0.90,
        "eye_discharge": 0.88, "itching": 0.82,
        "eye_pain": 0.70, "crusting_of_eyelids": 0.75,
        "sensitivity_to_light": 0.62,
        "blurred_and_distorted_vision": 0.52,
        "mild_fever": 0.32,
    },

    "Shingles": {
        "skin_rash": 0.95, "one_sided_rash": 0.90,
        "nerve_pain": 0.88, "burning_sensation_in_skin": 0.85,
        "blister": 0.82, "itching": 0.78,
        "fatigue": 0.68, "high_fever": 0.52,
        "headache": 0.58, "muscle_pain": 0.52,
        "sensitivity_to_light": 0.42,
    },

    "Measles": {
        "red_spots_over_body": 0.92, "high_fever": 0.90,
        "cough": 0.88, "runny_nose": 0.82,
        "redness_of_eyes": 0.78, "fatigue": 0.72,
        "loss_of_appetite": 0.68, "swelled_lymph_nodes": 0.62,
        "watering_from_eyes": 0.70, "malaise": 0.65,
        "sensitivity_to_light": 0.48, "headache": 0.55,
    },

    "Gout": {
        "joint_pain": 0.95, "knee_swelling": 0.88,
        "tophi": 0.62, "painful_walking": 0.82,
        "swelling_joints": 0.75, "mild_fever": 0.52,
        "back_pain": 0.48, "restlessness": 0.58,
        "fatigue": 0.52, "chills": 0.32,
        "muscle_pain": 0.38, "knee_pain": 0.72,
    },

    "Irritable Bowel Syndrome": {
        "abdominal_pain": 0.92, "bloating": 0.88,
        "mucus_in_stool": 0.75, "constipation": 0.70,
        "diarrhoea": 0.68, "belly_pain": 0.82,
        "fatigue": 0.62, "nausea": 0.55,
        "mood_swings": 0.42, "anxiety": 0.48,
        "loss_of_appetite": 0.38, "passage_of_gases": 0.72,
    },

    "Pancreatitis": {
        "abdominal_pain": 0.95, "nausea": 0.88,
        "vomiting": 0.82, "high_fever": 0.70,
        "sweating": 0.62, "back_pain": 0.75,
        "belly_pain": 0.85, "loss_of_appetite": 0.80,
        "dehydration": 0.58, "yellowish_skin": 0.42,
        "restlessness": 0.52, "malaise": 0.58,
    },

    "Meningitis": {
        "sudden_severe_headache": 0.92, "stiff_neck": 0.90,
        "high_fever": 0.90, "vomiting": 0.78,
        "sensitivity_to_light": 0.82, "altered_sensorium": 0.62,
        "red_spots_over_body": 0.52, "fatigue": 0.68,
        "chills": 0.58, "malaise": 0.62,
        "seizures": 0.38, "confusion": 0.52,
        "neck_pain": 0.78,
    },

    "Chronic Kidney Disease": {
        "fatigue": 0.92, "nausea": 0.82,
        "vomiting": 0.68, "loss_of_appetite": 0.80,
        "swelling_of_stomach": 0.72, "breathlessness": 0.70,
        "back_pain": 0.75, "muscle_pain": 0.62,
        "polyuria": 0.68, "weight_loss": 0.62,
        "dehydration": 0.52, "malaise": 0.58,
        "headache": 0.48, "swollen_legs": 0.65,
    },

    "PCOS": {
        "irregular_periods": 0.92, "weight_gain": 0.85,
        "pus_filled_pimples": 0.70, "facial_hair_growth": 0.75,
        "abnormal_menstruation": 0.88, "hair_loss": 0.68,
        "fatigue": 0.72, "mood_swings": 0.65,
        "pelvic_pain": 0.62, "anxiety": 0.48,
        "depression": 0.42, "lack_of_concentration": 0.38,
        "weight_loss": 0.22,
    },

    "Celiac Disease": {
        "abdominal_pain": 0.88, "diarrhoea": 0.85,
        "fatty_stool": 0.78, "bloating": 0.82,
        "weight_loss": 0.80, "fatigue": 0.85,
        "nausea": 0.62, "constipation": 0.42,
        "malaise": 0.58, "hair_loss": 0.52,
        "joint_pain": 0.38, "irritability": 0.42,
        "vomiting": 0.45,
    },

    "Fibromyalgia": {
        "widespread_pain": 0.92, "fatigue": 0.90,
        "tender_points": 0.88, "muscle_pain": 0.85,
        "headache": 0.75, "restlessness": 0.70,
        "lack_of_concentration": 0.80, "depression": 0.62,
        "mood_swings": 0.58, "joint_pain": 0.52,
        "malaise": 0.62, "anxiety": 0.52,
        "irritability": 0.48, "dizziness": 0.42,
    },

    "Lupus": {
        "butterfly_rash": 0.82, "joint_pain": 0.88,
        "fatigue": 0.92, "high_fever": 0.62,
        "hair_loss": 0.72, "photosensitivity_rash": 0.80,
        "ulcers_on_tongue": 0.60, "chest_pain": 0.52,
        "swelled_lymph_nodes": 0.58, "muscle_pain": 0.62,
        "breathlessness": 0.48, "redness_of_eyes": 0.38,
        "malaise": 0.68, "loss_of_appetite": 0.45,
    },

    "Multiple Sclerosis": {
        "weakness_in_limbs": 0.90,
        "blurred_and_distorted_vision": 0.82,
        "fatigue": 0.92, "loss_of_balance": 0.80,
        "muscle_pain": 0.70, "numbness_in_feet": 0.78,
        "visual_disturbances": 0.72,
        "lack_of_concentration": 0.62,
        "depression": 0.52, "constipation": 0.42,
        "irritability": 0.48, "dizziness": 0.58,
        "muscle_weakness": 0.75,
    },

    "Parkinson's Disease": {
        "tremors": 0.92, "loss_of_balance": 0.85,
        "stiff_muscles": 0.82, "fatigue": 0.75,
        "depression": 0.68, "constipation": 0.62,
        "memory_loss": 0.70, "lack_of_concentration": 0.65,
        "confusion": 0.48, "malaise": 0.52,
        "muscle_weakness": 0.58, "movement_stiffness": 0.78,
        "unsteadiness": 0.72,
    },

    "Epilepsy": {
        "seizures": 0.95, "altered_sensorium": 0.85,
        "confusion": 0.80, "headache": 0.70,
        "fatigue": 0.72, "anxiety": 0.52,
        "memory_loss": 0.62, "muscle_pain": 0.48,
        "nausea": 0.42, "depression": 0.38,
        "restlessness": 0.48, "irritability": 0.45,
    },

    "Gallstones": {
        "abdominal_pain": 0.92, "nausea": 0.85,
        "vomiting": 0.72, "indigestion": 0.80,
        "yellowish_skin": 0.58, "back_pain": 0.62,
        "belly_pain": 0.82, "loss_of_appetite": 0.68,
        "sweating": 0.52, "high_fever": 0.42,
        "dark_urine": 0.38, "yellowing_of_eyes": 0.35,
        "chills": 0.40,
    },
}

print(f"    New diseases defined: {len(NEW_DISEASES)}")

# ──────────────────────────────────────────────────────────────
# STEP 4: GENERATE NEW TRAINING ROWS
# ──────────────────────────────────────────────────────────────
print("\n[4/7] Generating training samples for new diseases...")

new_rows = []
for disease_name, symptom_probs in NEW_DISEASES.items():
    for _ in range(N_SAMPLES):
        row = {col: 0 for col in all_symptom_cols}
        for symptom, prob in symptom_probs.items():
            if symptom in row and prob > 0:
                row[symptom] = 1 if np.random.random() < prob else 0
        row['prognosis'] = disease_name
        new_rows.append(row)
    print(f"    ✓ {disease_name:<35} ({N_SAMPLES} samples)")

df_new = pd.DataFrame(new_rows)[all_symptom_cols + ['prognosis']]

# Make sure existing df has all columns
for col in all_symptom_cols:
    if col not in df_existing.columns:
        df_existing[col] = 0
df_existing = df_existing[all_symptom_cols + ['prognosis']]

# ──────────────────────────────────────────────────────────────
# STEP 5: COMBINE, SHUFFLE, SAVE Training.csv
# ──────────────────────────────────────────────────────────────
print("\n[5/7] Combining datasets and saving Training.csv...")

df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = os.path.join(DATA_DIR, "Training.csv")
df_combined.to_csv(output_path, index=False)

print(f"    Rows     : {len(df_combined)} (was {len(df_existing)})")
print(f"    Diseases : {df_combined['prognosis'].nunique()} (was {df_existing['prognosis'].nunique()})")
print(f"    Symptoms : {df_combined.shape[1] - 1}")
print(f"    Saved to : {output_path}")

# ──────────────────────────────────────────────────────────────
# STEP 6: UPDATE SUPPORTING CSV FILES
# ──────────────────────────────────────────────────────────────
print("\n[6/7] Updating supporting CSV files...")

# ── 6a. DESCRIPTIONS ──────────────────────────────────────────
NEW_DESCRIPTIONS = {
    "COVID-19":
        "COVID-19 is an infectious respiratory illness caused by the SARS-CoV-2 virus, "
        "characterized by fever, cough, loss of taste and smell, and in severe cases, breathing difficulty.",
    "Appendicitis":
        "Appendicitis is an inflammation of the appendix causing sharp abdominal pain, "
        "nausea, and fever that requires prompt medical attention.",
    "Kidney Stones":
        "Kidney stones are hard mineral deposits that form in the kidneys and cause severe "
        "flank or back pain, painful urination, and blood in urine.",
    "Iron Deficiency Anemia":
        "Iron deficiency anemia occurs when the body lacks enough iron to produce sufficient "
        "hemoglobin, causing fatigue, weakness, and breathlessness.",
    "Anxiety Disorder":
        "Anxiety disorder is a mental health condition characterized by excessive worry, "
        "restlessness, palpitations, and physical symptoms like sweating and chest tightness.",
    "Clinical Depression":
        "Clinical depression is a mood disorder causing persistent feelings of sadness, "
        "hopelessness, fatigue, and loss of interest in daily activities.",
    "Chronic Sinusitis":
        "Chronic sinusitis is a long-term inflammation of the sinuses causing nasal congestion, "
        "facial pressure, headache, and difficulty breathing through the nose.",
    "Tonsillitis":
        "Tonsillitis is an inflammation of the tonsils causing severe sore throat, high fever, "
        "swollen lymph nodes, and difficulty swallowing.",
    "Ear Infection":
        "Ear infection (otitis media) is an infection of the middle ear causing ear pain, "
        "fever, and temporary hearing loss.",
    "Conjunctivitis":
        "Conjunctivitis (pink eye) is an inflammation of the eye's conjunctiva causing redness, "
        "discharge, itching, and watery eyes.",
    "Shingles":
        "Shingles is a viral infection caused by reactivation of the chickenpox virus, "
        "producing a painful one-sided rash with blisters and nerve pain.",
    "Measles":
        "Measles is a highly contagious viral illness characterized by red spots, high fever, "
        "cough, runny nose, and red watery eyes.",
    "Gout":
        "Gout is a form of arthritis caused by high uric acid levels that form crystals in "
        "joints, causing sudden severe joint pain and swelling.",
    "Irritable Bowel Syndrome":
        "Irritable Bowel Syndrome (IBS) is a chronic digestive disorder causing abdominal pain, "
        "bloating, and alternating constipation and diarrhea.",
    "Pancreatitis":
        "Pancreatitis is an inflammation of the pancreas causing severe abdominal pain, "
        "nausea, vomiting, and fever, often requiring hospitalization.",
    "Meningitis":
        "Meningitis is an infection of the membranes surrounding the brain and spinal cord, "
        "causing sudden severe headache, stiff neck, high fever, and sensitivity to light.",
    "Chronic Kidney Disease":
        "Chronic kidney disease is a progressive loss of kidney function causing fatigue, "
        "swelling, nausea, and metabolic disturbances over time.",
    "PCOS":
        "Polycystic Ovary Syndrome (PCOS) is a hormonal disorder in women causing irregular "
        "periods, excessive hair growth, acne, and fertility issues.",
    "Celiac Disease":
        "Celiac disease is an autoimmune reaction to gluten that damages the small intestine, "
        "causing diarrhea, fatty stools, bloating, and malabsorption.",
    "Fibromyalgia":
        "Fibromyalgia is a chronic pain disorder characterized by widespread musculoskeletal "
        "pain, fatigue, tender points, and cognitive difficulties.",
    "Lupus":
        "Lupus (SLE) is a chronic autoimmune disease where the immune system attacks its own "
        "tissues, causing joint pain, butterfly rash, fatigue, and organ damage.",
    "Multiple Sclerosis":
        "Multiple sclerosis is an autoimmune disease affecting the brain and spinal cord, "
        "causing muscle weakness, vision problems, balance issues, and fatigue.",
    "Parkinson's Disease":
        "Parkinson's disease is a progressive neurological disorder affecting movement, "
        "causing tremors, muscle stiffness, loss of balance, and slowed movement.",
    "Epilepsy":
        "Epilepsy is a neurological disorder characterized by recurrent seizures due to "
        "abnormal electrical activity in the brain.",
    "Gallstones":
        "Gallstones are hardened deposits of bile in the gallbladder causing abdominal pain, "
        "nausea, and jaundice when they block the bile duct.",
}

desc_path = os.path.join(DATA_DIR, "description.csv")
df_desc = pd.read_csv(desc_path)
new_desc_rows = [{"Disease": d, "Description": v} for d, v in NEW_DESCRIPTIONS.items()
                 if d not in df_desc['Disease'].values]
if new_desc_rows:
    df_desc = pd.concat([df_desc, pd.DataFrame(new_desc_rows)], ignore_index=True)
df_desc.to_csv(desc_path, index=False)
print(f"    description.csv  updated — total diseases: {len(df_desc)}")

# ── 6b. PRECAUTIONS ───────────────────────────────────────────
NEW_PRECAUTIONS = {
    "COVID-19":                ["isolate yourself", "wear a mask", "stay hydrated and rest", "consult a doctor if breathing difficulty"],
    "Appendicitis":            ["seek emergency care immediately", "avoid eating or drinking", "do not apply heat to abdomen", "follow surgeon instructions post-op"],
    "Kidney Stones":           ["drink plenty of water", "reduce sodium and oxalate foods", "avoid prolonged dehydration", "consult a urologist"],
    "Iron Deficiency Anemia":  ["eat iron-rich foods", "take iron supplements as prescribed", "avoid tea and coffee with meals", "get regular blood tests"],
    "Anxiety Disorder":        ["practice deep breathing exercises", "limit caffeine intake", "maintain regular sleep schedule", "seek therapy or counselling"],
    "Clinical Depression":     ["consult a mental health professional", "maintain a routine", "exercise regularly", "reach out to supportive people"],
    "Chronic Sinusitis":       ["use saline nasal rinse", "avoid allergens and pollutants", "stay hydrated", "consult an ENT specialist"],
    "Tonsillitis":             ["rest and drink warm fluids", "gargle with warm salt water", "avoid cold drinks and ice cream", "consult a doctor for antibiotics"],
    "Ear Infection":           ["avoid inserting objects in ear", "keep ear dry", "complete prescribed antibiotic course", "follow up with ENT if persistent"],
    "Conjunctivitis":          ["avoid touching or rubbing eyes", "wash hands frequently", "do not share towels or pillows", "use prescribed eye drops"],
    "Shingles":                ["consult a doctor early for antivirals", "keep rash clean and dry", "avoid contact with pregnant or immunocompromised people", "manage pain with prescribed medication"],
    "Measles":                 ["isolate to prevent spread", "rest and stay hydrated", "take vitamin A supplements", "monitor for complications like pneumonia"],
    "Gout":                    ["avoid high-purine foods like red meat and alcohol", "stay well hydrated", "take prescribed medications", "maintain a healthy weight"],
    "Irritable Bowel Syndrome":["identify and avoid trigger foods", "manage stress effectively", "eat smaller and more frequent meals", "keep a food and symptom diary"],
    "Pancreatitis":            ["avoid alcohol completely", "follow a low-fat diet", "rest and stay hydrated", "seek urgent medical care for acute episodes"],
    "Meningitis":              ["seek emergency medical care immediately", "complete the full antibiotic course", "get vaccinated if eligible", "monitor recovery closely"],
    "Chronic Kidney Disease":  ["follow a kidney-friendly low-potassium diet", "control blood pressure and blood sugar", "avoid NSAIDs and nephrotoxic drugs", "attend regular nephrology checkups"],
    "PCOS":                    ["maintain a healthy weight", "exercise regularly", "follow a low glycaemic index diet", "consult a gynaecologist for hormonal management"],
    "Celiac Disease":          ["strictly follow a gluten-free diet", "read food labels carefully", "avoid cross-contamination", "get regular nutritional monitoring"],
    "Fibromyalgia":            ["maintain a gentle exercise routine", "prioritise quality sleep", "manage stress through mindfulness", "work with a rheumatologist"],
    "Lupus":                   ["avoid prolonged sun exposure", "use sunscreen daily", "take prescribed immunosuppressants", "monitor for organ involvement regularly"],
    "Multiple Sclerosis":      ["take disease-modifying therapy as prescribed", "exercise regularly within your limits", "manage fatigue with rest and pacing", "maintain a support network"],
    "Parkinson's Disease":     ["take medications on a strict schedule", "engage in physical therapy and exercise", "maintain home safety to prevent falls", "consult a neurologist regularly"],
    "Epilepsy":                ["take anti-epileptic medication consistently", "avoid seizure triggers like sleep deprivation", "do not drive until seizure-free as advised", "wear a medical alert bracelet"],
    "Gallstones":              ["follow a low-fat diet", "maintain a healthy weight", "avoid rapid weight loss", "consult a gastroenterologist for treatment options"],
}

prec_path = os.path.join(DATA_DIR, "precautions_df.csv")
df_prec = pd.read_csv(prec_path)
existing_prec_diseases = df_prec['Disease'].values if 'Disease' in df_prec.columns else []
new_prec_rows = []
for d, tips in NEW_PRECAUTIONS.items():
    if d not in existing_prec_diseases:
        while len(tips) < 4:
            tips.append("")
        new_prec_rows.append({
            "Disease": d,
            "Precaution_1": tips[0], "Precaution_2": tips[1],
            "Precaution_3": tips[2], "Precaution_4": tips[3],
        })
if new_prec_rows:
    df_prec = pd.concat([df_prec, pd.DataFrame(new_prec_rows)], ignore_index=True)
df_prec.to_csv(prec_path, index=False)
print(f"    precautions_df.csv updated — total diseases: {df_prec['Disease'].dropna().nunique()}")

# ── 6c. MEDICATIONS ───────────────────────────────────────────
NEW_MEDICATIONS = {
    "COVID-19":                "['Paracetamol', 'Antiviral agents (Nirmatrelvir/Ritonavir)', 'Dexamethasone (severe cases)', 'Oxygen therapy', 'Bronchodilators']",
    "Appendicitis":            "['IV Antibiotics (Cefazolin)', 'Metronidazole', 'Analgesics', 'Appendectomy (surgery)', 'IV Fluids']",
    "Kidney Stones":           "['Alpha-blockers (Tamsulosin)', 'Pain relief (NSAIDs)', 'Potassium Citrate', 'Extracorporeal Shockwave Lithotripsy', 'Ureterorenoscopy']",
    "Iron Deficiency Anemia":  "['Ferrous Sulphate tablets', 'Vitamin C supplements', 'Iron-rich diet', 'Folic Acid', 'IV Iron infusion (severe cases)']",
    "Anxiety Disorder":        "['SSRIs (Sertraline, Escitalopram)', 'Benzodiazepines (short-term)', 'Buspirone', 'Beta-blockers (Propranolol)', 'Cognitive Behavioural Therapy']",
    "Clinical Depression":     "['SSRIs (Fluoxetine, Sertraline)', 'SNRIs (Venlafaxine)', 'Tricyclic antidepressants', 'Psychotherapy', 'Lifestyle interventions']",
    "Chronic Sinusitis":       "['Saline nasal irrigation', 'Nasal corticosteroid sprays', 'Decongestants', 'Antibiotics (if bacterial)', 'Endoscopic sinus surgery (if refractory)']",
    "Tonsillitis":             "['Penicillin or Amoxicillin', 'Paracetamol for fever', 'Ibuprofen for pain', 'Salt water gargle', 'Tonsillectomy (if recurrent)']",
    "Ear Infection":           "['Amoxicillin (antibiotic)', 'Ear drops (antibiotic/analgesic)', 'Paracetamol for fever', 'Decongestants', 'Grommets if recurrent']",
    "Conjunctivitis":          "['Antibiotic eye drops (Chloramphenicol)', 'Antihistamine eye drops (allergic)', 'Artificial tears', 'Antiviral drops (viral)', 'Cold compresses']",
    "Shingles":                "['Acyclovir or Valacyclovir (antiviral)', 'Paracetamol or Ibuprofen', 'Gabapentin for nerve pain', 'Topical anaesthetics', 'Shingles vaccine (prevention)']",
    "Measles":                 "['Vitamin A supplements', 'Paracetamol for fever', 'Rest and hydration', 'MMR vaccine (prevention)', 'Antibiotics (if secondary bacterial infection)']",
    "Gout":                    "['NSAIDs (Indomethacin, Naproxen)', 'Colchicine', 'Corticosteroids', 'Allopurinol (long-term uric acid control)', 'Febuxostat']",
    "Irritable Bowel Syndrome":"['Antispasmodics (Mebeverine)', 'Loperamide for diarrhoea', 'Laxatives for constipation', 'Low-FODMAP diet', 'Antidepressants (low dose)']",
    "Pancreatitis":            "['IV Fluids and nil-by-mouth', 'Analgesics (IV Morphine)', 'Pancreatic enzyme supplements (chronic)', 'Antibiotics (if infected)', 'Endoscopic or surgical intervention']",
    "Meningitis":              "['IV Antibiotics (Ceftriaxone)', 'Dexamethasone (corticosteroid)', 'Antiviral drugs (viral meningitis)', 'IV Fluids', 'Meningitis vaccine (prevention)']",
    "Chronic Kidney Disease":  "['ACE inhibitors or ARBs', 'Erythropoietin (for anaemia)', 'Phosphate binders', 'Diuretics', 'Dialysis or kidney transplant (advanced)']",
    "PCOS":                    "['Oral contraceptives (hormonal regulation)', 'Metformin (insulin resistance)', 'Anti-androgens (Spironolactone)', 'Clomiphene (fertility)', 'Lifestyle modifications']",
    "Celiac Disease":          "['Strict gluten-free diet (only treatment)', 'Nutritional supplements (Iron, B12, Calcium)', 'Vitamin D supplements', 'Corticosteroids (refractory)', 'Regular monitoring']",
    "Fibromyalgia":            "['Duloxetine (SNRI)', 'Pregabalin', 'Amitriptyline (low dose)', 'Physiotherapy', 'Cognitive Behavioural Therapy']",
    "Lupus":                   "['Hydroxychloroquine (Plaquenil)', 'NSAIDs', 'Corticosteroids', 'Immunosuppressants (Azathioprine)', 'Belimumab (biologic)']",
    "Multiple Sclerosis":      "['Interferon beta', 'Glatiramer acetate', 'Natalizumab', 'Corticosteroids (for relapses)', 'Physical and occupational therapy']",
    "Parkinson's Disease":     "['Levodopa/Carbidopa', 'Dopamine agonists (Pramipexole)', 'MAO-B inhibitors (Selegiline)', 'Physiotherapy', 'Deep Brain Stimulation (advanced)']",
    "Epilepsy":                "['Valproate', 'Carbamazepine', 'Levetiracetam', 'Lamotrigine', 'Phenytoin']",
    "Gallstones":              "['Ursodeoxycholic acid (dissolution)', 'NSAIDs for pain', 'Laparoscopic cholecystectomy (surgery)', 'Extracorporeal shockwave lithotripsy', 'Antispasmodics']",
}

meds_path = os.path.join(DATA_DIR, "medications.csv")
df_meds = pd.read_csv(meds_path)
existing_med_diseases = df_meds['Disease'].values
new_med_rows = [{"Disease": d, "Medication": v} for d, v in NEW_MEDICATIONS.items()
                if d not in existing_med_diseases]
if new_med_rows:
    df_meds = pd.concat([df_meds, pd.DataFrame(new_med_rows)], ignore_index=True)
df_meds.to_csv(meds_path, index=False)
print(f"    medications.csv    updated — total diseases: {len(df_meds)}")

# ── 6d. DIETS ────────────────────────────────────────────────
NEW_DIETS = {
    "COVID-19":                "['High-Protein Diet', 'Vitamin C-rich foods', 'Zinc-rich foods', 'Warm broths and soups', 'Ginger and turmeric tea']",
    "Appendicitis":            "['Clear liquids post-surgery', 'Soft foods initially', 'High-fibre diet (recovery)', 'Lean proteins', 'Avoid spicy foods initially']",
    "Kidney Stones":           "['Increase water intake (2-3 litres/day)', 'Reduce sodium', 'Limit oxalate-rich foods (spinach, nuts)', 'Low-animal protein diet', 'Citrate-rich foods (lemonade)']",
    "Iron Deficiency Anemia":  "['Iron-rich foods (spinach, lentils, red meat)', 'Vitamin C to aid absorption', 'Avoid tea/coffee with meals', 'Fortified cereals', 'Dark leafy vegetables']",
    "Anxiety Disorder":        "['Magnesium-rich foods (nuts, seeds)', 'Omega-3 fatty acids', 'Reduce caffeine and alcohol', 'Chamomile and herbal teas', 'Probiotic-rich foods']",
    "Clinical Depression":     "['Mediterranean diet', 'Omega-3 rich foods (fish, walnuts)', 'Folate-rich foods (leafy greens)', 'Reduce sugar and processed foods', 'Tryptophan-rich foods (turkey, eggs)']",
    "Chronic Sinusitis":       "['Anti-inflammatory foods (berries, ginger)', 'Warm fluids and herbal teas', 'Reduce dairy (can thicken mucus)', 'Vitamin C-rich citrus', 'Spicy foods (can clear sinuses)']",
    "Tonsillitis":             "['Cold foods for comfort (ice cream, yogurt)', 'Warm broths and soups', 'Soft easily swallowed foods', 'Honey and ginger tea', 'Stay hydrated with water']",
    "Ear Infection":           "['Anti-inflammatory foods', 'Vitamin C and zinc-rich foods', 'Garlic (natural antibiotic)', 'Warm fluids', 'Avoid dairy if mucus-promoting']",
    "Conjunctivitis":          "['Vitamin A-rich foods (carrots, sweet potato)', 'Omega-3 fatty acids', 'Antioxidant-rich foods', 'Stay well hydrated', 'Zinc-rich foods']",
    "Shingles":                "['High-lysine foods (fish, chicken, eggs)', 'Avoid arginine-rich foods (nuts, chocolate)', 'Vitamin B12-rich foods', 'Anti-inflammatory diet', 'Plenty of fluids']",
    "Measles":                 "['Vitamin A-rich foods (sweet potato, carrots)', 'Soft easy-to-eat foods', 'Broths and soups', 'Citrus for vitamin C', 'Stay well hydrated']",
    "Gout":                    "['Low-purine diet (avoid red meat, shellfish)', 'Cherry juice or cherries', 'Reduce alcohol especially beer', 'Increase water intake', 'Low-fat dairy products']",
    "Irritable Bowel Syndrome":"['Low-FODMAP diet', 'Soluble fibre (oats, bananas)', 'Avoid trigger foods (onions, garlic, cabbage)', 'Small frequent meals', 'Stay well hydrated']",
    "Pancreatitis":            "['Clear liquids initially', 'Low-fat diet', 'Small frequent meals', 'Avoid alcohol completely', 'High-protein low-fat foods during recovery']",
    "Meningitis":              "['Nutrient-dense soft foods', 'High-calorie diet for recovery', 'Vitamin C and antioxidants', 'Probiotics post-antibiotics', 'Stay well hydrated']",
    "Chronic Kidney Disease":  "['Low potassium foods', 'Low phosphorus diet', 'Limit protein intake', 'Reduce sodium', 'Restrict fluid intake if advised']",
    "PCOS":                    "['Low-glycaemic index foods', 'Anti-inflammatory diet', 'High-fibre vegetables', 'Lean proteins', 'Avoid refined sugars and processed foods']",
    "Celiac Disease":          "['Strict gluten-free diet (no wheat, rye, barley)', 'Rice, quinoa, and gluten-free grains', 'Certified gluten-free oats', 'Fresh fruits and vegetables', 'Lean meats and fish']",
    "Fibromyalgia":            "['Anti-inflammatory foods', 'Avoid MSG and artificial additives', 'Magnesium-rich foods', 'Increase antioxidants', 'Reduce caffeine and alcohol']",
    "Lupus":                   "['Anti-inflammatory diet', 'Omega-3-rich foods', 'Calcium and vitamin D (for bone health)', 'Avoid alfalfa sprouts', 'Low-sodium diet']",
    "Multiple Sclerosis":      "['Anti-inflammatory diet', 'Vitamin D-rich foods', 'Omega-3 fatty acids', 'High-fibre diet', 'Avoid processed and fast foods']",
    "Parkinson's Disease":     "['Mediterranean diet', 'High-fibre foods for constipation', 'Antioxidant-rich berries', 'Avoid high-protein meals near levodopa', 'Stay hydrated']",
    "Epilepsy":                "['Ketogenic diet (medically supervised)', 'Regular meal schedule', 'Avoid skipping meals', 'Stay hydrated', 'Limit alcohol']",
    "Gallstones":              "['Low-fat diet', 'High-fibre diet', 'Avoid fried and greasy foods', 'Maintain healthy weight gradually', 'Olive oil in moderation']",
}

diets_path = os.path.join(DATA_DIR, "diets.csv")
df_diets = pd.read_csv(diets_path)
existing_diet_diseases = df_diets['Disease'].values
new_diet_rows = [{"Disease": d, "Diet": v} for d, v in NEW_DIETS.items()
                 if d not in existing_diet_diseases]
if new_diet_rows:
    df_diets = pd.concat([df_diets, pd.DataFrame(new_diet_rows)], ignore_index=True)
df_diets.to_csv(diets_path, index=False)
print(f"    diets.csv          updated — total diseases: {len(df_diets)}")

# ── 6e. WORKOUTS ──────────────────────────────────────────────
NEW_WORKOUTS = {
    "COVID-19":                ["Complete rest during acute illness", "Light walking during recovery", "Breathing exercises (diaphragmatic)", "Gentle stretching post-recovery", "Avoid strenuous activity until cleared by doctor"],
    "Appendicitis":            ["Complete bed rest post-surgery", "Short walks after 24-48 hours", "Avoid lifting heavy objects for 4-6 weeks", "Gradual return to normal activity", "Light stretching during late recovery"],
    "Kidney Stones":           ["Stay well hydrated during exercise", "Light walking and gentle cardio", "Avoid high-impact exercise during acute episodes", "Swimming and cycling are good choices", "Consult doctor before resuming intense workouts"],
    "Iron Deficiency Anemia":  ["Light aerobic exercise like walking", "Yoga and stretching", "Avoid overexertion until iron levels improve", "Gradually increase exercise intensity", "Swimming for low-impact cardio"],
    "Anxiety Disorder":        ["Regular aerobic exercise (30 min, 5 days/week)", "Yoga and meditation", "Deep breathing exercises daily", "Walking in nature", "Swimming or cycling"],
    "Clinical Depression":     ["Daily moderate exercise (walking, jogging)", "Yoga and mindfulness sessions", "Group fitness activities for social engagement", "Outdoor activities in sunlight", "Resistance training 3 times weekly"],
    "Chronic Sinusitis":       ["Avoid outdoor exercise on high pollen days", "Swimming in chlorinated pools may irritate sinuses", "Light indoor exercise", "Breathing exercises", "Yoga focusing on breathing techniques"],
    "Tonsillitis":             ["Complete rest during acute illness", "Light walking when fever-free", "Avoid strenuous activity until fully recovered", "Gargling as a throat exercise", "Gentle yoga post-recovery"],
    "Ear Infection":           ["Avoid swimming until infection clears", "Light walking is acceptable", "No activities that increase ear pressure (diving, flying)", "Balance exercises post-recovery", "Gentle neck exercises"],
    "Conjunctivitis":          ["Avoid swimming until cleared", "Light exercise is fine if feeling well", "Protect eyes from dust and wind outdoors", "No contact sports during infection", "Normal activities with good hygiene"],
    "Shingles":                ["Rest during acute phase", "Gentle walking when able", "Avoid contact sports", "Gentle stretching away from rash area", "Progressive return to normal activity post-recovery"],
    "Measles":                 ["Complete bed rest during acute illness", "No exercise until fever-free for 48 hours", "Light walking during recovery", "Gradual return to activity", "Avoid contact sports for 2 weeks post-recovery"],
    "Gout":                    ["Low-impact exercise (swimming, cycling)", "Avoid high-impact exercise during flare-ups", "Maintain healthy weight through exercise", "Walking regularly", "Range-of-motion exercises for joints"],
    "Irritable Bowel Syndrome":["Regular moderate exercise (reduces symptoms)", "Yoga for stress and digestive health", "Walking after meals", "Avoid intense exercise that triggers symptoms", "Swimming and cycling are good choices"],
    "Pancreatitis":            ["Complete rest during acute episodes", "Very gentle walking during recovery", "Avoid strenuous activity for weeks", "Gradual return to light exercise", "No alcohol-fuelled social activities"],
    "Meningitis":              ["Complete bed rest during acute illness", "Very gradual return to activity over weeks", "Light walking as first step", "Avoid contact sports for 3-6 months", "Guided physiotherapy if neurological effects"],
    "Chronic Kidney Disease":  ["Regular low-impact exercise (walking, swimming)", "Avoid overexertion", "Resistance training to prevent muscle wasting", "Yoga for flexibility", "Consult nephrologist before starting exercise program"],
    "PCOS":                    ["Regular aerobic exercise 150 min/week", "Resistance training to improve insulin sensitivity", "Yoga to manage stress and hormones", "HIIT workouts 2-3 times weekly", "Swimming and cycling"],
    "Celiac Disease":          ["Regular moderate exercise is beneficial", "Yoga to aid digestion", "Walking and swimming", "Build intensity gradually as nutrition improves", "No specific restrictions once in remission"],
    "Fibromyalgia":            ["Low-impact aerobic exercise (aqua therapy, walking)", "Gentle stretching daily", "Yoga and Tai Chi", "Gradually increase duration before intensity", "Avoid overexertion that causes symptom flare"],
    "Lupus":                   ["Low-to-moderate intensity exercise", "Avoid exercise in intense sun and heat", "Swimming and water aerobics", "Yoga for joint flexibility", "Rest during flares; exercise during remission"],
    "Multiple Sclerosis":      ["Regular low-impact exercise (swimming, cycling)", "Yoga and stretching for flexibility", "Balance and coordination exercises", "Cool down before and after to manage heat sensitivity", "Physiotherapy-guided exercise program"],
    "Parkinson's Disease":     ["Regular aerobic exercise", "Balance and gait training", "Tai Chi for stability and coordination", "Resistance training for muscle strength", "Dance therapy (shown to improve motor function)"],
    "Epilepsy":                ["Regular moderate exercise is safe and beneficial", "Avoid solo swimming or high-altitude activities", "Always exercise with a companion", "Yoga and relaxation techniques", "Cycling with safety gear"],
    "Gallstones":              ["Regular moderate exercise helps prevent recurrence", "Walking and swimming", "Avoid rapid weight loss through crash diets", "Yoga for digestive health", "Maintain healthy BMI through consistent activity"],
}

workout_path = os.path.join(DATA_DIR, "workout_df.csv")
df_workout = pd.read_csv(workout_path)
existing_workout_diseases = df_workout['disease'].values if 'disease' in df_workout.columns else []
new_workout_rows = []
for d, tips in NEW_WORKOUTS.items():
    if d not in existing_workout_diseases:
        for tip in tips:
            new_workout_rows.append({"disease": d, "workout": tip})
if new_workout_rows:
    df_workout_new = pd.DataFrame(new_workout_rows)
    df_workout = pd.concat([df_workout, df_workout_new], ignore_index=True)
df_workout.to_csv(workout_path, index=False)
print(f"    workout_df.csv     updated — total rows: {len(df_workout)}")

# ── 6f. SYMPTOM SEVERITY ──────────────────────────────────────
NEW_SEVERITY = {
    "loss_of_taste": 3, "oxygen_saturation_drop": 7,
    "blood_in_urine": 5, "flank_pain": 5,
    "hair_loss": 2, "cold_intolerance": 2,
    "facial_pain_pressure": 3, "post_nasal_drip": 2,
    "bad_breath": 1, "swollen_tonsils": 4,
    "ear_pain": 4, "discharge_from_ear": 4,
    "hearing_loss": 5, "eye_discharge": 3,
    "eye_pain": 4, "crusting_of_eyelids": 2,
    "sensitivity_to_light": 4, "burning_sensation_in_skin": 5,
    "nerve_pain": 6, "one_sided_rash": 5,
    "knee_swelling": 4, "tophi": 4,
    "bloating": 3, "mucus_in_stool": 3,
    "sudden_severe_headache": 7, "widespread_pain": 5,
    "tender_points": 4, "fatty_stool": 3,
    "irregular_periods": 3, "pelvic_pain": 4,
    "facial_hair_growth": 2, "butterfly_rash": 5,
    "photosensitivity_rash": 4, "numbness_in_feet": 5,
    "tremors": 6, "stiff_muscles": 5,
    "memory_loss": 6, "seizures": 7,
    "confusion": 5,
}

sev_path = os.path.join(DATA_DIR, "Symptom-severity.csv")
df_sev = pd.read_csv(sev_path)
df_sev.columns = [c.strip() for c in df_sev.columns]
existing_sev_syms = df_sev['Symptom'].values
new_sev_rows = [{"Symptom": s, "weight": w} for s, w in NEW_SEVERITY.items()
                if s not in existing_sev_syms]
if new_sev_rows:
    df_sev = pd.concat([df_sev, pd.DataFrame(new_sev_rows)], ignore_index=True)
df_sev.to_csv(sev_path, index=False)
print(f"    Symptom-severity   updated — total symptoms: {len(df_sev)}")

# ──────────────────────────────────────────────────────────────
# STEP 7: SUMMARY
# ──────────────────────────────────────────────────────────────
print("\n[7/7] EXPANSION COMPLETE!")
print("=" * 60)
print(f"  Training data  : {len(df_combined)} rows")
print(f"  Diseases       : {df_combined['prognosis'].nunique()}")
print(f"  Symptom columns: {df_combined.shape[1] - 1}")
print(f"  All CSVs updated: description, precautions,")
print(f"                    medications, diets, workout, severity")
print("=" * 60)
print("\n  NEXT STEP: Run  python train_model.py")
print("=" * 60)

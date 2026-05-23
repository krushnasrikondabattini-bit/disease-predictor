"""
=============================================================
  DISEASE PREDICTION - MODEL TRAINING SCRIPT
=============================================================
  Run this script ONCE to train and save the best ML model.
  
  Usage:
      python train_model.py

  Output:
      models/best_model.pkl      ← trained model
      models/label_encoder.pkl   ← disease label encoder
      models/feature_columns.pkl ← list of 132 symptom columns
      models/model_report.txt    ← accuracy comparison of all models
=============================================================
"""

import zipfile
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

print("=" * 60)
print("  DISEASE PREDICTION MODEL TRAINER")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")

# Training.csv is stored as a zip-inside-zip from the original dataset
TRAINING_PATH = "data/Training.csv"

try:
    # Try reading it as a nested zip first (original format)
    with zipfile.ZipFile(TRAINING_PATH, 'r') as z:
        with z.open('Training.csv') as f:
            df = pd.read_csv(f)
    print("    Loaded from nested zip format.")
except Exception:
    # Fallback: plain CSV
    df = pd.read_csv(TRAINING_PATH)
    print("    Loaded as plain CSV.")

print(f"    Dataset shape: {df.shape}")
print(f"    Diseases (classes): {df['prognosis'].nunique()}")
print(f"    Symptom features:   {df.shape[1] - 1}")

# ─────────────────────────────────────────────────────────────
# STEP 2: PREPARE FEATURES & LABELS
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Preparing features and labels...")

# All columns except last are symptom binary features (0 or 1)
feature_columns = list(df.columns[:-1])
X = df[feature_columns].values
y = df['prognosis'].values

# Encode disease names → integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"    Features: {len(feature_columns)} symptoms")
print(f"    Classes : {len(le.classes_)} diseases")

# Train / test split — 80% train, 20% test, stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"    Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# STEP 3: TRAIN & COMPARE MULTIPLE MODELS
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Training and comparing ML models...")
print("    (This may take 1-2 minutes)\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest":         RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":     GradientBoostingClassifier(n_estimators=150, random_state=42),
    "SVM (RBF)":             SVC(kernel='rbf', probability=True, random_state=42),
    "Naive Bayes":           GaussianNB(),
    "Decision Tree":         DecisionTreeClassifier(random_state=42),
}

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = {
        "cv_mean": cv_scores.mean(),
        "cv_std":  cv_scores.std(),
        "test_acc": test_acc,
        "model": model
    }
    print(f"    {name:<22} CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%  |  Test: {test_acc*100:.2f}%")

# ─────────────────────────────────────────────────────────────
# STEP 4: ENSEMBLE (VOTING) - combines top 3 models
# ─────────────────────────────────────────────────────────────
print("\n    Building Ensemble (Voting Classifier) from top models...")
ensemble = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ('gb',  GradientBoostingClassifier(n_estimators=150, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
    ],
    voting='soft'
)
cv_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
ensemble.fit(X_train, y_train)
test_acc = accuracy_score(y_test, ensemble.predict(X_test))
results["Ensemble (RF+GB+SVM)"] = {
    "cv_mean": cv_scores.mean(),
    "cv_std":  cv_scores.std(),
    "test_acc": test_acc,
    "model": ensemble
}
print(f"    {'Ensemble (RF+GB+SVM)':<22} CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%  |  Test: {test_acc*100:.2f}%")

# Pick best model by CV accuracy
best_name = max(results, key=lambda k: results[k]["cv_mean"])
best_model = results[best_name]["model"]
print(f"\n    ✅ Best model: {best_name} ({results[best_name]['cv_mean']*100:.2f}% CV accuracy)")

# ─────────────────────────────────────────────────────────────
# STEP 5: SAVE MODELS & REPORT
# ─────────────────────────────────────────────────────────────
print("\n[4/5] Saving model files...")

os.makedirs("models", exist_ok=True)

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("    Saved: models/best_model.pkl")
print("    Saved: models/label_encoder.pkl")
print("    Saved: models/feature_columns.pkl")

# Save report
report_lines = ["DISEASE PREDICTION - MODEL COMPARISON REPORT\n", "="*55 + "\n\n"]
for name, res in results.items():
    star = " ← BEST" if name == best_name else ""
    report_lines.append(f"{name}{star}\n")
    report_lines.append(f"  CV Accuracy : {res['cv_mean']*100:.2f}% ± {res['cv_std']*100:.2f}%\n")
    report_lines.append(f"  Test Accuracy: {res['test_acc']*100:.2f}%\n\n")

# Detailed classification report of best model
y_pred = best_model.predict(X_test)
report_lines.append("\nDETAILED CLASSIFICATION REPORT (Best Model)\n")
report_lines.append("="*55 + "\n")
report_lines.append(classification_report(y_test, y_pred, target_names=le.classes_))

with open("models/model_report.txt", "w", encoding="utf-8") as f:
    f.writelines(report_lines)
print("    Saved: models/model_report.txt")

print("\n[5/5] Done! ✅")
print("="*60)
print("  Now run:  python app/predict.py")
print("="*60)

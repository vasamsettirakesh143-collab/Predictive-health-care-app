"""
train_model.py
Generates a synthetic (but realistic-shaped) health dataset and trains a
Random Forest classifier to predict Type-2 Diabetes risk.

Run this once locally (or Render will NOT run it automatically -- see README)
to produce model/risk_model.joblib, which app.py loads at request time.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

N = 4000  # synthetic patients

def generate_synthetic_data(n=N):
    age = np.random.randint(18, 80, n)
    bmi = np.round(np.random.normal(27, 5, n).clip(15, 50), 1)
    glucose = np.round(np.random.normal(100, 25, n).clip(60, 300), 1)
    blood_pressure = np.round(np.random.normal(120, 15, n).clip(80, 200), 1)
    cholesterol = np.round(np.random.normal(190, 35, n).clip(100, 350), 1)
    activity_level = np.random.randint(0, 3, n)     # 0=low,1=moderate,2=high
    smoking = np.random.randint(0, 2, n)             # 0=no,1=yes
    family_history = np.random.randint(0, 2, n)      # 0=no,1=yes

    # A hand-built risk function so labels correlate sensibly with features
    risk_score = (
        0.03 * (age - 40)
        + 0.08 * (bmi - 25)
        + 0.04 * (glucose - 100)
        + 0.02 * (blood_pressure - 120)
        + 0.015 * (cholesterol - 190)
        - 0.6 * activity_level
        + 0.9 * smoking
        + 1.1 * family_history
        + np.random.normal(0, 1.5, n)   # noise
    )
    probability = 1 / (1 + np.exp(-0.15 * risk_score))
    label = (probability > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "bmi": bmi, "glucose": glucose,
        "blood_pressure": blood_pressure, "cholesterol": cholesterol,
        "activity_level": activity_level, "smoking": smoking,
        "family_history": family_history, "risk_label": label,
    })
    return df


def main():
    df = generate_synthetic_data()
    feature_cols = ["age", "bmi", "glucose", "blood_pressure", "cholesterol",
                     "activity_level", "smoking", "family_history"]
    X = df[feature_cols]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print("Accuracy:", accuracy_score(y_test, preds))
    print("ROC-AUC :", roc_auc_score(y_test, proba))

    os.makedirs("model", exist_ok=True)
    joblib.dump({"model": model, "features": feature_cols}, "model/risk_model.joblib")
    print("Saved model to model/risk_model.joblib")


if __name__ == "__main__":
    main()

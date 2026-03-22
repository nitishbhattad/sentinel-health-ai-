import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

engine = create_engine("postgresql://localhost:5432/healthcare_db")

# ── 1. Pull features + ward labels ───────────────────────────────
query = """
    SELECT 
        f.admission_count,
        f.emergency_ratio,
        f.avg_los_days,
        f.max_los_days,
        COALESCE(f.days_since_last_admission, -1) AS days_since_last_admission,
        f.has_previous_admission,
        r.risk_score,
        r.predicted_ward
    FROM patient_features f
    JOIN patient_risk_scores r ON f.subject_id = r.subject_id
    WHERE r.predicted_ward IS NOT NULL;
"""

df = pd.read_sql(query, engine)
print(f"Dataset shape: {df.shape}")
print(f"\nWard distribution:\n{df['predicted_ward'].value_counts()}")

# ── 2. Encode ward labels ─────────────────────────────────────────
le = LabelEncoder()
df["ward_encoded"] = le.fit_transform(df["predicted_ward"])
print(f"\nWard encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

feature_cols = [
    "admission_count", "emergency_ratio",
    "avg_los_days", "max_los_days",
    "days_since_last_admission",
    "has_previous_admission", "risk_score"
]

X = df[feature_cols]
y = df["ward_encoded"]

# ── 3. Train/test split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train XGBoost classifier ───────────────────────────────────
print("\n── Training Ward Prediction Model ──")
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0
)
xgb_model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────
y_pred = xgb_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

# ── 6. Feature importance ─────────────────────────────────────────
print("\n── Feature Importances ──")
importances = pd.Series(xgb_model.feature_importances_, index=feature_cols)
print(importances.sort_values(ascending=False))

# ── 7. Save model + encoder ───────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(xgb_model, "models/ward_model.pkl")
joblib.dump(le, "models/ward_encoder.pkl")
print("\n✅ Ward model saved to models/ward_model.pkl")
print("✅ Label encoder saved to models/ward_encoder.pkl")

# ── 8. Generate predictions for ALL patients ──────────────────────
all_query = """
    SELECT 
        f.subject_id,
        f.admission_count,
        f.emergency_ratio,
        f.avg_los_days,
        f.max_los_days,
        COALESCE(f.days_since_last_admission, -1) AS days_since_last_admission,
        f.has_previous_admission,
        r.risk_score
    FROM patient_features f
    JOIN patient_risk_scores r ON f.subject_id = r.subject_id;
"""
all_df = pd.read_sql(all_query, engine)
all_X = all_df[feature_cols]

all_df["predicted_ward"] = le.inverse_transform(xgb_model.predict(all_X))
all_df["ward_probabilities"] = xgb_model.predict_proba(all_X).max(axis=1).round(3)

# ── 9. Write back to DB ───────────────────────────────────────────
for _, row in all_df.iterrows():
    with engine.connect() as conn:
        conn.execute(text("""
            UPDATE patient_risk_scores
            SET predicted_ward = :ward
            WHERE subject_id = :sid
        """), {"ward": row["predicted_ward"], "sid": int(row["subject_id"])})
        conn.commit()

print("\n✅ Ward predictions written to patient_risk_scores")

print("\nWard distribution (ML predicted):")
print(all_df["predicted_ward"].value_counts())

print("\nSample predictions:")
print(
    all_df[["subject_id", "risk_score", "predicted_ward", "ward_probabilities"]]
    .sort_values("risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)
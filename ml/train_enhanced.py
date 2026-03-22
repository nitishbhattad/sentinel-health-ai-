"""
Enhanced XGBoost Training with Charlson + Age + ICU features
Target: ROC-AUC 0.80+
"""

import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://localhost:5432/healthcare_db"
)
engine = create_engine(DATABASE_URL)

# ── Load Features ──────────────────────────────────────────────────────────
print("="*55)
print("Enhanced XGBoost Training — MIMIC-III")
print("="*55)

print("\nLoading features from database...")
df = pd.read_sql("""
    SELECT
        subject_id,
        admission_count,
        avg_los_days,
        max_los_days,
        emergency_ratio,
        days_since_last_admission,
        has_previous_admission,
        high_risk_label,
        COALESCE(age, 0)             AS age,
        COALESCE(gender_male, 0)     AS gender_male,
        COALESCE(charlson_score, 0)  AS charlson_score,
        COALESCE(num_diagnoses, 0)   AS num_diagnoses,
        COALESCE(num_icu_stays, 0)   AS num_icu_stays,
        COALESCE(total_icu_hours, 0) AS total_icu_hours,
        COALESCE(had_icu, 0)         AS had_icu
    FROM patient_features
    WHERE high_risk_label IS NOT NULL
""", engine)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['high_risk_label'].value_counts()}")

# ── Features ───────────────────────────────────────────────────────────────
FEATURES = [
    "admission_count",
    "emergency_ratio",
    "days_since_last_admission",
    "has_previous_admission",
    "age",
    "gender_male",
    "charlson_score",
    "num_diagnoses",
    "num_icu_stays",
    "total_icu_hours",
    "had_icu",
]

# Only use features that have non-zero variance
available = []
for f in FEATURES:
    if f in df.columns:
        if df[f].std() > 0:
            available.append(f)
        else:
            print(f"  Skipping {f} (zero variance)")

print(f"\nUsing {len(available)} features: {available}")

X = df[available].fillna(0)
y = df["high_risk_label"]

# ── Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Class balance ──────────────────────────────────────────────────────────
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / pos if pos > 0 else 1.0
print(f"\nClass balance: {neg} negative, {pos} positive")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── Train XGBoost ──────────────────────────────────────────────────────────
print("\n── Training Enhanced XGBoost ──")
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
auc     = roc_auc_score(y_test, y_proba)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {auc:.4f}")

# ── Cross Validation ───────────────────────────────────────────────────────
print("\n── Cross-Validation (5-fold) ──")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
print(f"CV ROC-AUC: {cv_scores.round(3)}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── Feature Importances ────────────────────────────────────────────────────
print("\n── Feature Importances ──")
importances = pd.Series(model.feature_importances_, index=available)
print(importances.sort_values(ascending=False).round(4))

# ── Save Model ─────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/xgb_risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature list for predict.py
with open("models/feature_list.pkl", "wb") as f:
    pickle.dump(available, f)

print(f"\n✅ Enhanced model saved to models/xgb_risk_model.pkl")
print(f"✅ Feature list saved to models/feature_list.pkl")
print(f"\nFinal ROC-AUC: {auc:.4f}")

if auc >= 0.80:
    print("🎉 Target achieved! ROC-AUC >= 0.80")
elif auc >= 0.75:
    print("✅ Good score! ROC-AUC >= 0.75")
elif auc >= 0.70:
    print("📈 Decent score. ROC-AUC >= 0.70")
else:
    print("⚠️  Still improving. Consider adding more features.")

print("\nNext step: python ml/predict.py")
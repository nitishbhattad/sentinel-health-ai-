import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ── 1. Connect ────────────────────────────────────────────────────
engine = create_engine("postgresql://localhost:5432/healthcare_db")

# ── 2. Pull features ──────────────────────────────────────────────
query = """
    SELECT 
        admission_count,
        emergency_ratio,
        COALESCE(days_since_last_admission, -1) AS days_since_last_admission,
        has_previous_admission,
        high_risk_label
    FROM patient_features
    WHERE high_risk_label IS NOT NULL;
"""
df = pd.read_sql(query, engine)

X = df.drop(columns=["high_risk_label"])
y = df["high_risk_label"]

# ── 3. Split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 4. Logistic Regression (balanced) ────────────────────────────
print("\n── Logistic Regression (balanced) ──")
lr = LogisticRegression(class_weight="balanced", random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr  = lr.predict_proba(X_test_scaled)[:, 1]
print(classification_report(y_test, y_pred_lr))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

# ── 5. Random Forest (balanced) ───────────────────────────────────
print("\n── Random Forest (balanced) ──")
rf = RandomForestClassifier(
    n_estimators=100, 
    class_weight="balanced", 
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf  = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# ── 6. XGBoost ────────────────────────────────────────────────────
print("\n── XGBoost ──")

# scale_pos_weight handles imbalance in XGBoost
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale = neg / pos
print(f"scale_pos_weight: {scale:.2f}")

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale,   # ← handles imbalance
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb  = xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_xgb))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_xgb):.4f}")

# ── 7. Feature Importances (XGBoost) ─────────────────────────────
print("\n── XGBoost Feature Importances ──")
importances = pd.Series(xgb.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False))

# ── 8. Cross-validation (XGBoost) ────────────────────────────────
print("\n── Cross-Validation (XGBoost, 5-fold) ──")
cv_scores = cross_val_score(xgb, X, y, cv=5, scoring="roc_auc")
print(f"CV ROC-AUC scores: {cv_scores.round(3)}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
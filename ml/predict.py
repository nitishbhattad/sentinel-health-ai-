import pandas as pd
import joblib
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
import shap
import matplotlib.pyplot as plt

engine = create_engine("postgresql://localhost:5432/healthcare_db")

# ── 1. Pull features ──────────────────────────────────────────────
query = """
    SELECT 
        subject_id,
        admission_count,
        emergency_ratio,
        COALESCE(days_since_last_admission, -1) AS days_since_last_admission,
        has_previous_admission,
        high_risk_label
    FROM patient_features
    WHERE high_risk_label IS NOT NULL;
"""
df = pd.read_sql(query, engine)

feature_cols = [
    "admission_count", "emergency_ratio",
    "days_since_last_admission", "has_previous_admission"
]

X = df[feature_cols]
y = df["high_risk_label"]

# ── 2. Train final model on ALL data ─────────────────────────────
neg, pos = (y == 0).sum(), (y == 1).sum()

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=neg/pos,
    random_state=42,
    eval_metric="logloss",
    verbosity=0
)
xgb.fit(X, y)

# ── 3. Save model to disk ─────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(xgb, "models/xgb_risk_model.pkl")
print("✅ Model saved to models/xgb_risk_model.pkl")

# ── 4. Generate risk scores for ALL patients ──────────────────────
df["risk_score"]       = xgb.predict_proba(X)[:, 1]
df["risk_prediction"]  = xgb.predict(X)
df["risk_tier"] = pd.cut(
    df["risk_score"],
    bins=[0, 0.3, 0.6, 1.0],
    labels=["LOW", "MEDIUM", "HIGH"]
)

print(f"\nRisk tier distribution:")
print(df["risk_tier"].value_counts())

# ── 5. Write predictions back to PostgreSQL ───────────────────────
predictions_df = df[[
    "subject_id", "risk_score", 
    "risk_prediction", "risk_tier"
]].copy()

predictions_df.to_sql(
    "patient_risk_scores",
    engine,
    if_exists="replace",
    index=False
)
print("\n✅ Predictions written to patient_risk_scores table")

# ── 6. Preview top high-risk patients ────────────────────────────
print("\nTop 10 highest risk patients:")
print(
    df[["subject_id", "risk_score", "risk_tier"]]
    .sort_values("risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)

# ── 7. SHAP Explanation ───────────────────────────────────────────
print("\n── SHAP Analysis ──")

# Create SHAP explainer
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X)

# ── 8. Summary Plot (global — which features matter most overall) ──
shap.summary_plot(
    shap_values, X,
    plot_type="bar",
    show=False
)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ SHAP summary plot saved to models/shap_summary.png")

# ── 9. Individual explanation for highest risk patient ────────────
top_patient_idx = df["risk_score"].idxmax()
top_patient_id  = df.loc[top_patient_idx, "subject_id"]
top_shap        = shap_values[top_patient_idx]

print(f"\nSHAP breakdown for highest risk patient (ID: {top_patient_id}):")
for feat, val, shap_val in zip(
    feature_cols,
    X.iloc[top_patient_idx],
    top_shap
):
    direction = "↑ increases risk" if shap_val > 0 else "↓ decreases risk"
    print(f"  {feat:<35} value={val:.2f}   SHAP={shap_val:+.4f}  {direction}")

# ── 10. Save SHAP values for ALL patients to DB ───────────────────
shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in feature_cols])
shap_df["subject_id"] = df["subject_id"].values

shap_df.to_sql(
    "patient_shap_values",
    engine,
    if_exists="replace",
    index=False
)
print("\n✅ SHAP values written to patient_shap_values table")

# ── 11. Waterfall plot for top patient ────────────────────────────
shap.waterfall_plot(
    shap.Explanation(
        values=top_shap,
        base_values=explainer.expected_value,
        data=X.iloc[top_patient_idx].values,
        feature_names=feature_cols
    ),
    show=False
)
plt.tight_layout()
plt.savefig(f"models/shap_patient_{top_patient_id}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✅ Waterfall plot saved to models/shap_patient_{top_patient_id}.png")
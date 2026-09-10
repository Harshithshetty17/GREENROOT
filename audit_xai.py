import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular

# 1. Load Data and Artifacts
df = pd.read_csv(os.path.join("data", "Crop_recommendation.csv"))
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']

scaler = joblib.load(os.path.join("models", "scaler.pkl"))
classes = joblib.load(os.path.join("models", "class_names.pkl"))
model = joblib.load(os.path.join("models", "stacking_model.pkl"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Surrogate Model for SHAP & LIME
surrogate = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_scaled, y_train)
shap_explainer = shap.TreeExplainer(surrogate)
shap_values = shap_explainer.shap_values(X_test_scaled)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=features,
    class_names=classes,
    mode='classification'
)

# 3. Quantitative Explainability Consensus Audit
audit_indices = [0, 15, 30]
print("\n" + "=" * 70)
print("QUANTITATIVE XAI CONSISTENCY AUDIT (SHAP vs. LIME)")
print("=" * 70)

for idx in audit_indices:
    instance = X_test_scaled[idx]
    pred_label = model.predict([instance])[0]
    class_idx = classes.index(pred_label)

    # Robust SHAP attribution extraction across SHAP versions
    if isinstance(shap_values, list):
        shap_attr = np.abs(shap_values[class_idx][idx])
    elif len(getattr(shap_values, "shape", [])) == 3:
        shap_attr = np.abs(shap_values[idx, :, class_idx])
    else:
        shap_attr = np.abs(shap_values[idx])

    top_shap_idx = np.argsort(shap_attr)[-3:]
    top_shap = set([features[i] for i in top_shap_idx])

    # Top-3 LIME attributions
    exp = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=surrogate.predict_proba,
        num_features=3,
        labels=[class_idx]
    )
    top_lime = set([features[x[0]] for x in exp.as_map()[class_idx][:3]])

    # Mathematical Consensus: Jaccard Similarity Index (k=3)
    intersection = len(top_shap.intersection(top_lime))
    union = len(top_shap.union(top_lime))
    jaccard = intersection / union if union != 0 else 0.0

    print(f"\nSample #{idx} | Predicted Crop: {pred_label.upper()}")
    print(f"SHAP Top-3 Drivers: {top_shap}")
    print(f"LIME Top-3 Drivers: {top_lime}")
    print(f"Jaccard Agreement Index (k=3): {jaccard:.2f} ({intersection}/{union} overlap)")

print("=" * 70)
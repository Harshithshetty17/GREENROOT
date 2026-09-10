import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

# 1. Load Primary Benchmark Dataset
data_path = os.path.join("data", "Crop_recommendation.csv")
df = pd.read_csv(data_path)
print(f"Benchmark dataset loaded successfully. Shape: {df.shape}")

# 2. Separate Features and Targets
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[feature_cols]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Define Stacking Architecture
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
    ('adaboost', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

meta_learner = LogisticRegression(max_iter=1000)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)

# 4. Stratified 5-Fold Cross-Validation
print("\nRunning Stratified 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stacking_clf, X_scaled, y, cv=cv, scoring='accuracy', n_jobs=-1)
print(f"Mean CV Accuracy: {scores.mean() * 100:.2f}% (± {scores.std() * 100:.2f}%)")

# 5. Fit Model and Save Artifacts
print("\nFitting final stacking model...")
stacking_clf.fit(X_scaled, y)

joblib.dump(stacking_clf, os.path.join("models", "stacking_model.pkl"))
joblib.dump(scaler, os.path.join("models", "scaler.pkl"))
joblib.dump(list(stacking_clf.classes_), os.path.join("models", "class_names.pkl"))

print("[✓] Model artifacts saved to 'models/' directory.")
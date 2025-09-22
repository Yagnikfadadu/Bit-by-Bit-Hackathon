#!/usr/bin/env python3
import os, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix

ART_DIR = "artifacts"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load features
feat_path = os.path.join(ART_DIR, "features.csv")
if not os.path.exists(feat_path):
    raise SystemExit("features.csv not found. Run extract_features.py first.")

df = pd.read_csv(feat_path)
labels = df['label'].values
X = df.drop(columns=['label']).values
feat_names = df.drop(columns=['label']).columns.tolist()

# Standardize
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# RandomForest with LOOCV
clf = RandomForestClassifier(n_estimators=200, random_state=42)

if len(labels) > 1:
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(Xs):
        clf.fit(Xs[train_idx], labels[train_idx])
        p = clf.predict(Xs[test_idx])
        y_true.append(labels[test_idx][0])
        y_pred.append(p[0])
    print("RandomForest LOOCV report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=np.unique(labels)))

# Fit final model on all data
final_clf = RandomForestClassifier(n_estimators=200, random_state=42)
final_clf.fit(Xs, labels)
joblib.dump(final_clf, os.path.join(MODEL_DIR, "rf_model.joblib"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
joblib.dump(feat_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
print("[OK] Saved RF model and scaler")

# Print top features
importances = final_clf.feature_importances_
top_idx = np.argsort(importances)[::-1][:min(20, len(importances))]
print("Top features (name, importance):")
for i in top_idx:
    print(feat_names[i], importances[i])

#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

REFERENCE_DIR = "."
VICTIM_FILES = [
    "victim_cycles_features.csv",
    "victim_cachemisses_features.csv",
    "victim_branchmisses_features.csv",
]

TARGET_CLASSES = {"alexnet", "mobilenet_v2"}  # only focus on these

def load_reference_features():
    dfs = []
    labels = []
    for fname in os.listdir(REFERENCE_DIR):
        if not fname.endswith(".csv"):
            continue
        if fname.startswith("victim") or fname.startswith("trace"):
            continue

        if "_data" in fname:
            label = fname.split("_data")[0]
        elif "_part" in fname:
            label = fname.split("_part")[0]
        else:
            continue

        if label not in TARGET_CLASSES:
            continue  # skip unrelated models

        df = pd.read_csv(os.path.join(REFERENCE_DIR, fname))
        df = df.drop(columns=["event"], errors="ignore")

        if df.empty:
            continue

        df["__label__"] = label
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No reference features found for alexnet/mobilenet.")

    combined = pd.concat(dfs, ignore_index=True).fillna(0)
    y = combined["__label__"].values
    X = combined.drop(columns=["__label__"]).values
    return X, y, combined.drop(columns=["__label__"]).columns

def load_victim_features(columns):
    victim_data = {}
    for vf in VICTIM_FILES:
        if os.path.exists(vf):
            df = pd.read_csv(vf)
            df = df.drop(columns=["event"], errors="ignore")
            df_aligned = df.reindex(columns=columns, fill_value=0)
            victim_data[vf] = df_aligned.values[0]
    return victim_data

def main():
    print("[*] Loading reference features...")
    X_ref, y_ref, feature_cols = load_reference_features()
    print(f"[+] Loaded {len(y_ref)} samples from {set(y_ref)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_ref)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    clf.fit(X_ref, y_enc)

    print("[*] Loading victim features...")
    victim_features = load_victim_features(feature_cols)
    if not victim_features:
        print("[!] No victim feature files found.")
        return

    for vf, feats in victim_features.items():
        feats = feats.reshape(1, -1)
        probs = clf.predict_proba(feats)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.classes_[pred_idx]
        print(f"\n=== Prediction for {vf} ===")
        print(f"Predicted: {pred_label} (p={probs[pred_idx]:.3f})")
        print("Class probabilities:")
        for cls, p in zip(le.classes_, probs):
            print(f"  {cls:12s}: {p:.3f}")

if __name__ == "__main__":
    main()

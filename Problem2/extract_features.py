#!/usr/bin/env python3
import glob, os, joblib
import pandas as pd
import numpy as np
from utils import extract_features_from_df, resample_df_to_vector, EVENTS

CLEANED_DIR = "cleaned"
OUT_DIR = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

features_rows = []
labels = []
ts_vectors = []
filenames = []

for path in sorted(glob.glob(os.path.join(CLEANED_DIR, "*_clean.csv"))):
    name = os.path.basename(path).replace("_clean.csv", "")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print("[WARN] failed reading", path, e)
        continue
    feats = extract_features_from_df(df)
    features_rows.append(feats)
    labels.append(name)
    filenames.append(name)
    # build a time-series vector (resampled)
    v = resample_df_to_vector(df, events=EVENTS, length=1000)
    ts_vectors.append(v)

# Save features CSV
if features_rows:
    X = pd.DataFrame(features_rows).fillna(0.0)
    X.insert(0, "label", labels)
    X.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    print("[OK] Saved features.csv with shape", X.shape)

    # Save time-series vectors and labels (numpy + joblib)
    np.save(os.path.join(OUT_DIR,"ts_vectors.npy"), np.vstack(ts_vectors))
    joblib.dump(labels, os.path.join(OUT_DIR, "ts_labels.joblib"))
    print("[OK] Saved time-series vectors and labels")
else:
    print("[WARN] No cleaned files found to extract features from.")

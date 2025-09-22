#!/usr/bin/env python3
"""
Utilities: parse perf csv, resample time-series, extract features, DTW
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft
from math import inf

# canonical events we expect
EVENTS = ["cycles", "instructions", "cache-references", "cache-misses", "branches", "branch-misses"]

def detect_event_index(parts):
    """Try to find which column contains the event name by matching known events."""
    for idx in range(min(6, len(parts))):
        p = parts[idx].strip().lower()
        if p in EVENTS:
            return idx
    if len(parts) > 3:
        return 3
    if len(parts) > 2:
        return 2
    return None

def parse_perf_csv(path):
    """Robust parser for perf stat -I CSV output."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            idx_event = detect_event_index(parts)
            if idx_event is None:
                continue
            time_str = parts[0]
            event = parts[idx_event]
            if not event:
                continue
            try:
                time_sec = float(time_str)
            except:
                continue
            try:
                value = parts[1].replace(",", "")
                if value == "":
                    value = parts[2] if len(parts) > 2 else ""
                    value = value.replace(",", "")
                value_num = float(value)
            except:
                continue
            rows.append((time_sec, value_num, event))

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["time_sec", "value", "event"])
    df = df.pivot_table(index="time_sec", columns="event", values="value")
    df = df.reset_index().sort_values("time_sec")
    for e in EVENTS:
        if e not in df.columns:
            df[e] = np.nan
    return df[["time_sec"] + EVENTS]

def resample_df_to_vector(df, events=None, length=1000):
    """Resample numeric event channels to a fixed length vector."""
    if events is None:
        events = EVENTS
    df = df.sort_values("time_sec")
    if df.empty:
        return np.zeros(length * len(events))
    t = np.array(df["time_sec"].values)
    t_norm = (t - t[0]) / max(1e-12, (t[-1] - t[0]))
    new_t = np.linspace(0.0, 1.0, num=length)
    cols = []
    for ev in events:
        if ev not in df.columns:
            col = np.zeros(len(t))
        else:
            col = np.array(df[ev].fillna(method="ffill").fillna(method="bfill").fillna(0.0))
        try:
            interp = np.interp(new_t, t_norm, col)
        except Exception:
            if len(col) >= length:
                interp = col[:length]
            else:
                interp = np.pad(col, (0, length - len(col)), 'edge')
        std = np.std(interp)
        if std > 0:
            interp = (interp - np.mean(interp)) / std
        cols.append(interp)
    return np.concatenate(cols)

def extract_features_from_df(df):
    """Extract stats/FFT/correlation features from cleaned df."""
    feats = {}
    events = [c for c in df.columns if c != "time_sec"]
    for ev in events:
        x = df[ev].fillna(0.0).values
        feats[f"{ev}_mean"] = float(np.mean(x)) if len(x) else 0.0
        feats[f"{ev}_std"] = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        feats[f"{ev}_min"] = float(np.min(x)) if len(x) else 0.0
        feats[f"{ev}_max"] = float(np.max(x)) if len(x) else 0.0
        feats[f"{ev}_median"] = float(np.median(x)) if len(x) else 0.0
        feats[f"{ev}_skew"] = float(stats.skew(x)) if len(x) > 2 else 0.0
        feats[f"{ev}_kurtosis"] = float(stats.kurtosis(x)) if len(x) > 3 else 0.0
    return feats

def dtw_distance(a, b):
    """Simple O(N^2) DTW distance between 1D arrays."""
    n, m = len(a), len(b)
    dtw = np.full((n+1, m+1), inf)
    dtw[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            dtw[i,j] = cost + min(dtw[i-1,j], dtw[i,j-1], dtw[i-1,j-1])
    return float(dtw[n,m])

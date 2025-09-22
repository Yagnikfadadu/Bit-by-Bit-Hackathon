#!/usr/bin/env python3
"""
Clean perf stat -I CSV logs into usable per-event traces.
"""

import os
import pandas as pd

INPUT_DIR = "traces"
OUTPUT_DIR = "cleaned"

def parse_perf_csv(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            time_str, value_str, event = parts[0], parts[1], parts[3]

            # Parse time
            try:
                time_sec = float(time_str)
            except:
                continue

            # Parse value (remove commas)
            try:
                value = int(value_str.replace(",", ""))
            except:
                try:
                    value = float(value_str.replace(",", ""))
                except:
                    value = None

            rows.append((time_sec, value, event))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["time_sec", "value", "event"])

    # Pivot: event names become columns
    df = df.pivot_table(index="time_sec", columns="event", values="value")
    df = df.reset_index().sort_values("time_sec")

    return df


def clean_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.endswith(".csv"):
            continue
        in_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        out_path = os.path.join(output_dir, base.replace("_run", "_clean") + ".csv")

        print(f"[INFO] Cleaning {in_path} -> {out_path}")
        df = parse_perf_csv(in_path)

        if df.empty:
            print(f"[WARN] {in_path} produced empty DataFrame, skipping")
            continue

        df.to_csv(out_path, index=False)
        print(f"[OK] Saved cleaned CSV with shape {df.shape}")


if __name__ == "__main__":
    clean_all()

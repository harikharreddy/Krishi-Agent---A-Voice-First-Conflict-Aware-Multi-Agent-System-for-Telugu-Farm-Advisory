"""
Build a filtered, multi-year price history for the Price Agent's trend
logic, from the raw Kaggle Agmarknet archive (data/raw_agmarknet_kaggle/*.csv).

Filters down to Telangana + Andhra Pradesh, Tomato + Potato, across all
available years, and writes one combined CSV to
data/price_history_ap_telangana.csv.

Run once (or whenever a new yearly file is added) -- not part of the
live agent's runtime path.
"""

import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "..", "..", "data", "raw_agmarknet_kaggle")
OUT_PATH = os.path.join(HERE, "..", "..", "data", "price_history_ap_telangana.csv")

TARGET_STATES = {"Telangana", "Andhra Pradesh"}
TARGET_COMMODITIES = {"Tomato", "Potato"}


def build():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {RAW_DIR}")

    filtered_frames = []
    for path in files:
        year = os.path.basename(path).replace(".csv", "")
        print(f"Processing {year}...")
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df["State"] = df["State"].astype(str).str.strip()
        df["Commodity"] = df["Commodity"].astype(str).str.strip()
        subset = df[df["State"].isin(TARGET_STATES) & df["Commodity"].isin(TARGET_COMMODITIES)]
        print(f"  {len(subset)} matching rows")
        filtered_frames.append(subset)

    combined = pd.concat(filtered_frames, ignore_index=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {len(combined)} rows to {OUT_PATH}")


if __name__ == "__main__":
    build()

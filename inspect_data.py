import os
import pandas as pd

data_dir = "data"
files = [
    "Crop_recommendation.csv",
    "GreenRoot_Consolidated_Master_Dataset.csv",
    "Cleaned_NFSM_Dataset.csv",
    "greenroot_consolidated_dataset.csv"
]

print("=" * 80)
print("DATASET INVENTORY AND SCHEMA AUDIT")
print("=" * 80)

for filename in files:
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            print(f"\n[✓] File: {filename}")
            print(f"    Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
            print(f"    Columns: {list(df.columns[:10])}{' ...' if df.shape[1] > 10 else ''}")
        except Exception as e:
            print(f"\n[!] Error loading {filename}: {e}")
    else:
        print(f"\n[✗] Missing: {path}")

print("\n" + "=" * 80)
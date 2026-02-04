'''
    根据门店不同的贡献 给门店画像
'''
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="outputs/datasets/store_level_strategy.csv"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/datasets/store_profiles.csv"
)
args = parser.parse_args()

df = pd.read_csv(args.input_path)

def assign_profile(row):
    if row.get("A_core_share", 0) >= 0.6:
        return "Core_driven_store"
    elif row.get("B_holiday_share", 0) >= 0.25:
        return "Holiday_driven_store"
    else:
        return "Tail_heavy_store"

df["store_profile"] = df.apply(assign_profile, axis=1)

print("[INFO] store profile distribution:")
print(df["store_profile"].value_counts())

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
df.to_csv(args.output_path, index=False)

print("[OK] saved store profiles to:", args.output_path)
print(df.head())

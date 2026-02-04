'''
    给每个商品一个画像
'''
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="outputs/datasets/dept_assortment_metrics.csv"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/datasets/dept_assortment_tiers.csv"
)
args = parser.parse_args()

df = pd.read_csv(args.input_path)

# 阈值（可解释、可调）
sales_threshold = df["avg_weekly_sales"].quantile(0.70)
cv_threshold = df["cv"].median()
holiday_lift_threshold = 1.15

def assign_tier(row):
    if row["avg_weekly_sales"] >= sales_threshold and row["cv"] <= cv_threshold:
        return "A_core"
    elif row["holiday_lift"] >= holiday_lift_threshold:
        return "B_holiday"
    else:
        return "C_tail"

df["tier"] = df.apply(assign_tier, axis=1)

print("[INFO] tier distribution:")
print(df["tier"].value_counts())

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
df.to_csv(args.output_path, index=False)
print(f"[OK] saved tiered assortment to: {args.output_path}")


'''
    根据其卖商品的画像 给一个store的不同商品的贡献值
'''
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    type=str,
    default="outputs/datasets/walmart_clean.csv"
)
parser.add_argument(
    "--tier_path",
    type=str,
    default="outputs/datasets/dept_assortment_tiers.csv"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/datasets/store_level_strategy.csv"
)
args = parser.parse_args()

# -- 加载 --
df = pd.read_csv(args.data_path)
tiers = pd.read_csv(args.tier_path)

# -- 将dept的画像赋予到mature data --
df = df.merge(tiers[["Dept", "tier"]], on="Dept", how="left")

print("[INFO] merged tier into main data:", df.shape)

# -- 按照 store*tier 分小表 --
grp = df.groupby(["Store", "tier"])

store_strategy = grp["Weekly_Sales"].sum().reset_index()

# -- melt 一下 --
store_strategy = store_strategy.pivot(
    index="Store",
    columns="tier",
    values="Weekly_Sales"
).fillna(0).reset_index()

# -- 计算总销售额 --
tier_cols = [c for c in store_strategy.columns if c != "Store"]
store_strategy["total_sales"] = store_strategy[tier_cols].sum(axis=1)

# -- 计算贡献度 --
for c in tier_cols:
    store_strategy[f"{c}_share"] = store_strategy[c] / store_strategy["total_sales"]

# -- 保存 --
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
store_strategy.to_csv(args.output_path, index=False)

print("[OK] saved store-level strategy to:", args.output_path)
print(store_strategy.head())

import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
args = parser.parse_args()
data_dir = args.data_dir
# 1) 读数据（你现在是 csv 工作表）
train = pd.read_csv(f"{data_dir}/train.csv")
features = pd.read_csv(f"{data_dir}/features.csv")
stores = pd.read_csv(f"{data_dir}/stores.csv")
test = pd.read_csv(f"{data_dir}/test.csv")  # 先读着，后面预测用

# 2) 统一日期格式（避免 join 对不上）
for df in (train, features, test):
    df["Date"] = pd.to_datetime(df["Date"])

# 3) 以 train 为主表：先贴 features（Store+Date），再贴 stores（Store）
df = (
    train
    .merge(features, on=["Store", "Date"], how="left", suffixes=("", "_feat"))
    .merge(stores, on=["Store"], how="left", suffixes=("", "_store"))
)

# 4) 快速检查：行数不应该变（left join）
print("train rows:", len(train), "merged rows:", len(df))
print("columns:", df.columns.tolist())
print(df.head(3))

out_dir = "outputs/datasets"
os.makedirs(out_dir, exist_ok=True)

out_path = f"{out_dir}/walmart_merged.csv"
df.to_csv(out_path, index=False)
print(f"[OK] saved merged dataset to: {out_path}")

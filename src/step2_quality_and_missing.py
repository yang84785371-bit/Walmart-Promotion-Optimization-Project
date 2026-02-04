import os
import pandas as pd
import argparse

# -- 命令行参数 --
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="outputs/datasets/walmart_merged.csv",
    help="path to merged dataset"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/datasets/walmart_clean.csv",
    help="path to save cleaned dataset"
)
args = parser.parse_args()

# -- 加载 --
df = pd.read_csv(args.input_path)

print("[INFO] loaded dataset shape:", df.shape)

# -- 对markdown的缺失值进行处理 --
markdown_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]

for col in markdown_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0.0)

print("[OK] MarkDown1-5 missing values filled with 0")

# -- 统一holiday的格式 --
# train 里的 IsHoliday 和 features 里的 IsHoliday_feat
if "IsHoliday_feat" in df.columns:
    # -- 简单一致性检查（不一致也不报错，只提示）--
    mismatch = (df["IsHoliday"] != df["IsHoliday_feat"]).sum()
    print(f"[INFO] IsHoliday mismatch count: {mismatch}")

    # -- 保留 IsHoliday_feat，删除 IsHoliday --
    df = df.drop(columns=["IsHoliday"])
    df = df.rename(columns={"IsHoliday_feat": "IsHoliday"})

    print("[OK] unified IsHoliday column (kept IsHoliday_feat)")

# -- 保存 clean 数据 --
out_dir = os.path.dirname(args.output_path)
os.makedirs(out_dir, exist_ok=True)

df.to_csv(args.output_path, index=False)
print(f"[OK] saved clean dataset to: {args.output_path}")
print("[INFO] final dataset shape:", df.shape)

import pandas as pd
import argparse
import os
# -- 命令行参数 --
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path",
    type=str,
    default="outputs/datasets/walmart_clean.csv"
)
parser.add_argument(
    "--output_path",
    type=str,
    default="outputs/datasets/dept_assortment_metrics.csv"
)
args = parser.parse_args()

# -- 读取 --
df = pd.read_csv(args.input_path)

print("[INFO] loaded clean dataset:", df.shape)

df = df[df["Weekly_Sales"] >= 0] # 不要销量不正常的
df["Date"] = pd.to_datetime(df["Date"], errors="coerce") # todatatime
df = df[df["Date"].notna()].copy() # 不要转换datatime失误的 并且给别名
df["IsHoliday"] = df["IsHoliday"].map(lambda v: str(v).strip().lower() in ["true", "1", "t", "yes"])
df["WeekKey"] = df["Date"].dt.to_period("W").astype(str)
weeks = df.groupby("Dept")["WeekKey"].nunique()

# -- eda --
group = df.groupby("Dept")

assortment = pd.DataFrame({
    "avg_weekly_sales": group["Weekly_Sales"].mean(),
    "std_weekly_sales": group["Weekly_Sales"].std(),
    "holiday_sales": group.apply(
        lambda x: x.loc[x["IsHoliday"] == True, "Weekly_Sales"].mean()
    ),
    "non_holiday_sales": group.apply(
        lambda x: x.loc[x["IsHoliday"] == False, "Weekly_Sales"].mean()
    ),
    "store_coverage": group["Store"].nunique(),
    "n_weeks": group.size(),
    "holiday_obs": group.apply(lambda x: x["IsHoliday"].sum()),
    "non_holiday_obs": group.apply(lambda x: (~x["IsHoliday"]).sum()),
}).reset_index()

# -- 计算指标 --
assortment["cv"] = assortment["std_weekly_sales"] / assortment["avg_weekly_sales"]
assortment["holiday_lift"] = assortment["holiday_sales"] / assortment["non_holiday_sales"]
assortment.loc[assortment["non_holiday_sales"].isna() | (assortment["non_holiday_sales"] <= 0), "holiday_lift"] = pd.NA
assortment.loc[assortment["avg_weekly_sales"] <= 0, "cv"] = pd.NA
assortment["holiday_lift"] = assortment["holiday_lift"].replace([float("inf"), -float("inf")], pd.NA)
assortment["n_weeks"] = assortment["Dept"].map(weeks) #  map是查询的意思


print("[INFO] assortment metrics preview:")
print(assortment.head())

# ---------- save ----------
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
assortment.to_csv(args.output_path, index=False)

print(f"[OK] saved assortment metrics to: {args.output_path}")

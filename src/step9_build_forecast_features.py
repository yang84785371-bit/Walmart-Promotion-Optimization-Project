'''
整个项目分两段：

第一段（step1–8）是规则型 what-if 分析，用于在不做因果的前提下，稳健筛选对促销可能敏感的商品。

第二段（step9–15B）是在已筛选对象上，引入预测模型与响应曲线，用于估计促销强度、制定预算分配方案，并通过多种切片做稳健性验证。

这个step 其实就是为了第二阶段构造feature
'''
import os
import argparse
import pandas as pd
import numpy as np

# -- 增加时间特征 --
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Date 必须是 datetime
    df["Date"] = pd.to_datetime(df["Date"]) # df 是clean的数据 我们对周级数据 进行datatime的转换
    iso = df["Date"].dt.isocalendar() # 转换成iso周 就是为了避免一周被拆成两年 造成feature的断裂 确保一周只属于一年
    df["weekofyear"] = iso.week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52) # 周不能作为连续变量 要做成类似于位置编码的东西
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    df["month"] = df["Date"].dt.month.astype(int) # 这里也是同理 但由于有业务语义  所以两个都要保留
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df

# -- 建立滚动延迟特征 --
def build_lag_rolling_features(df: pd.DataFrame, group_keys=("Store", "Dept")) -> pd.DataFrame:
    # -- 窗口特征 必须先按时间排序 --
    df = df.sort_values(list(group_keys) + ["Date"]).copy()

    # --- lag features ---
    for k in [1, 2, 4]:
        df[f"sales_lag_{k}"] = df.groupby(list(group_keys))["Weekly_Sales"].shift(k) # 上周 / 上两周 / 上四周卖了多少 不是均值 单一值

    # -- 这里要shift 是因为不能泄露信息 --
    # -- rolling mean/std 要先 shift(1)，确保只用到历史 rolling 是“往前看最多 N 个数” --
    g = df.groupby(list(group_keys))["Weekly_Sales"] # 先定义好聚合 是按照 store*dept
    df["sales_mean_4"] = g.shift(1).rolling(window=4, min_periods=1).mean() 
    df["sales_mean_8"] = g.shift(1).rolling(window=8, min_periods=1).mean()
    df["sales_std_8"] = g.shift(1).rolling(window=8, min_periods=2).std()

    # -- std 可能出现 NaN（样本太少），后面统一处理 --
    return df


def attach_business_tags(
    df: pd.DataFrame,
    dept_tier_path: str,
    store_profile_path: str
) -> pd.DataFrame:
    # -- 商品画像：来自 step4 的输出（必须包含 Dept, tier）--
    if os.path.exists(dept_tier_path):
        tiers = pd.read_csv(dept_tier_path)
        if "Dept" in tiers.columns and "tier" in tiers.columns:
            df = df.merge(tiers[["Dept", "tier"]], on="Dept", how="left")
        else:
            print("[WARN] dept_tier_path found but missing columns Dept/tier, skip attach tier")
    else:
        print("[WARN] dept_tier_path not found, skip attach tier")

    # -- 商店画像：来自 step6 的输出（必须包含 Store, store_profile）--
    if os.path.exists(store_profile_path):
        sp = pd.read_csv(store_profile_path)
        if "Store" in sp.columns and "store_profile" in sp.columns:
            df = df.merge(sp[["Store", "store_profile"]], on="Store", how="left")
        else:
            print("[WARN] store_profile_path found but missing columns Store/store_profile, skip attach store_profile")
    else:
        print("[WARN] store_profile_path not found, skip attach store_profile")

    return df


def main():
    # -- 命令行参数 --
    # -- 这里面其实是和第一阶段的规则what if 断开的 基本上只用到一些基本信息 --
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="outputs/datasets/walmart_clean.csv")
    parser.add_argument("--dept_tier_path", type=str, default="outputs/datasets/dept_assortment_tiers.csv")
    parser.add_argument("--store_profile_path", type=str, default="outputs/datasets/store_profiles.csv")
    parser.add_argument("--output_path", type=str, default="outputs/datasets/forecast_features_w1.csv")
    args = parser.parse_args()

    # -- 读取数据 --
    df = pd.read_csv(args.input_path)
    print("[INFO] loaded clean data:", df.shape)

    # -- 基础清洗：去掉负销量（若有）--
    df = df[df["Weekly_Sales"] >= 0].copy()

    # -- 时间特征 -- 
    df = add_time_features(df)

    # -- 促销强度（可用作一个连续特征）--
    md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
    for c in md_cols:
        if c not in df.columns:
            df[c] = 0.0
    df["promo_intensity"] = df[md_cols].sum(axis=1)
    df["promo_intensity_log"] = np.log1p(df["promo_intensity"]) # 这里转对数会稳点

    # -- 附上商店分层以及物品分类 --
    df = attach_business_tags(df, args.dept_tier_path, args.store_profile_path)

    # -- lag / rolling 获取窗口特征 --
    df = build_lag_rolling_features(df, group_keys=("Store", "Dept"))

    # -- label：下一周销量 --
    df = df.sort_values(["Store", "Dept", "Date"]).copy()
    df["y_next_week"] = df.groupby(["Store", "Dept"])["Weekly_Sales"].shift(-1) # 类似的 改个shift参数罢了

    # 选择特征列（先保持克制，够用即可）
    feature_cols = [
        "Store", "Dept", "Date",
        "IsHoliday", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "promo_intensity_log",
        "weekofyear", "week_sin", "week_cos",
        "month", "month_sin", "month_cos",
        "sales_lag_1", "sales_lag_2", "sales_lag_4",
        "sales_mean_4", "sales_mean_8", "sales_std_8",
        "tier", "store_profile",
        "y_next_week",
    ]

    # -- 防御一下 必要的列一定要有 其他的随便 --
    keep_cols = [c for c in feature_cols if c in df.columns]
    out = df[keep_cols].copy()

    # -- y 和最关键的 lag_1 一定要有值 没有就丢掉 --
    out = out.dropna(subset=["y_next_week", "sales_lag_1"])

    # -- 对于数值的列 不行就填 0 对na的轻微处理 --
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].fillna(0.0)

    # 保存
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.to_csv(args.output_path, index=False)

    print("[OK] saved forecast feature table to:", args.output_path)
    print("[INFO] feature table shape:", out.shape)
    print(out.head(3))


if __name__ == "__main__":
    main()

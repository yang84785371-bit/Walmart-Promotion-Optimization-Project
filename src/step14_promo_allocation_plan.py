'''
step14: promo allocation plan (from step13 curves)

只读取 step13 输出的【标准字段】：
    group_col, group
    promo_intensity
    pred_mean_sales
     lift_vs_zero, lift_pct_vs_zero
    n_rows

输出：
    每个 group 的 baseline、best promo、turning promo（边际转负点）
'''

import os
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "group_col", "group",
    "promo_intensity",
    "pred_mean_sales",
    "lift_vs_zero", "lift_pct_vs_zero",
    "n_rows",
]


def _load_curve(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[step14] {os.path.basename(path)} missing columns: {missing}")

    df = df[REQUIRED_COLS].copy()
    df["promo_intensity"] = df["promo_intensity"].astype(float)
    df["pred_mean_sales"] = df["pred_mean_sales"].astype(float)
    df["lift_vs_zero"] = df["lift_vs_zero"].astype(float)
    df["lift_pct_vs_zero"] = df["lift_pct_vs_zero"].astype(float)
    df["n_rows"] = df["n_rows"].astype(int)

    df = df.sort_values(["group", "promo_intensity"]).reset_index(drop=True)
    return df

# -- 根据group 对曲线进行总结出plan --
def _summarize_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for g, gdf in df.groupby("group", sort=True):
        gdf = gdf.sort_values("promo_intensity").copy()
        if len(gdf) < 2:
            continue

        # -- 基线：promo=0 --
        # -- step13 是把 baseline 固定为 promo=0 来算 lift，所以这里优先找 promo=0 的点 这里得到一个loc 就是促销为0 的分位数的位置 --
        idx0 = None
        zero_mask = np.isclose(gdf["promo_intensity"].values, 0.0)
        if zero_mask.any():
            idx0 = int(np.where(zero_mask)[0][0])
        else:
            idx0 = 0
        # -- 得到base --
        base_row = gdf.iloc[idx0]
        base_sales = float(base_row["pred_mean_sales"])

        # -- best：lift 直接取最大值就可以了 --
        best_idx = int(gdf["lift_vs_zero"].values.argmax())
        best_row = gdf.iloc[best_idx]

        best_promo = float(best_row["promo_intensity"])
        best_lift = float(best_row["lift_vs_zero"])
        best_lift_pct = float(best_row["lift_pct_vs_zero"])

        # -- turning：边际收益首次为负的点（Δlift/Δpromo < 0）--
        promos = gdf["promo_intensity"].values.astype(float)
        lifts = gdf["lift_vs_zero"].values.astype(float)

        d_lift = np.diff(lifts)
        d_promo = np.diff(promos)
        marginal = np.divide(d_lift, d_promo, out=np.zeros_like(d_lift), where=d_promo != 0)

        turning_promo = np.nan
        turning_marginal = np.nan
        for i, m in enumerate(marginal):
            if m < 0:
                turning_promo = float(promos[i + 1])
                turning_marginal = float(m)
                break # 首个边际收益率为负的 就可以break了

        rows.append({
            "group": str(g),
            "n_rows": int(base_row["n_rows"]),
            "base_pred_at_zero": base_sales,
            "best_promo_intensity": best_promo,
            "best_lift_vs_zero": best_lift,
            "best_lift_pct_vs_zero": best_lift_pct,
            "turning_promo_intensity": turning_promo,
            "turning_marginal": turning_marginal,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("best_lift_vs_zero", ascending=False).reset_index(drop=True)
    return out


def _run_one(curve_path: str, out_path: str):
    # -- load step13的输出 促销分位数曲线 --
    df = _load_curve(curve_path)
    group_col_name = str(df["group_col"].iloc[0]) if len(df) > 0 else "unknown" # groupby标准

    plan = _summarize_group(df)
    if plan.empty:
        print(f"[WARN] empty plan for {curve_path}")
        return

    plan.insert(0, "group_col", group_col_name) # 插入group类型
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plan.to_csv(out_path, index=False)
    print("[OK] saved:", out_path)
    print(plan.head(10))


def main():
    # -- 确认输出目录存在 --
    os.makedirs("outputs/metrics", exist_ok=True)

    # -- 定义input path 和 output path  --
    files = [
        ("outputs/metrics/step13_promo_curve_overall.csv", "outputs/metrics/step14_promo_plan_overall.csv"),
        ("outputs/metrics/step13_promo_curve_by_tier.csv", "outputs/metrics/step14_promo_plan_by_tier.csv"),
        ("outputs/metrics/step13_promo_curve_by_store_profile.csv", "outputs/metrics/step14_promo_plan_by_store_profile.csv"),
        ("outputs/metrics/step13_promo_curve_by_tier_store_profile.csv", "outputs/metrics/step14_promo_plan_by_tier_store_profile.csv"),
    ]

    found = 0 # 防御一下 
    # -- 对于每个促销曲线 我们都要输出一个plan 
    for in_path, out_path in files:
        if os.path.exists(in_path):
            _run_one(in_path, out_path)
            found += 1

    if found == 0:
        print("[WARN] step13 curve files not found. Please run step13 first.")


if __name__ == "__main__":
    main()

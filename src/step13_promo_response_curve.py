'''
step13: Promotion response curve (decision visualization, NOT forecasting)

语义说明：
    使用 step10 训练好的 LightGBM 作为“促销响应函数”
    在【历史销量轨迹固定不变】前提下，对 promo 强度做 what-if 扫描
    本质是 partial dependence，用于决策解释，而不是销量预测
'''

import os
import argparse
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from pandas.api.types import CategoricalDtype

# -- 定义未知参数 --
UNKNOWN = "UNKNOWN"


def simulate_response_curve(
    df_full: pd.DataFrame,
    booster: lgb.Booster,
    feature_names: list[str],
    promo_grid_log: np.ndarray,
    promo_col: str,
    group_col: str | None,
    output_prefix: str,
    min_group_rows: int = 500,
):
    '''
    对 promo 轴做 what-if 扫描：
        固定所有非 promo 特征
        仅替换 promo_col
        计算 mean(pred) 作为该 promo 水平下的期望销量

    注意：predict 输入严格使用 feature_names 子集，避免 categorical mismatch。
    '''

    rows = []

    if group_col:
        groups = sorted(df_full[group_col].dropna().unique())
    else:
        groups = [None]

    # -- 只拿训练时的特征列，保证“列集合+顺序”完全一致 --
    # -- 注意：df_full 里可以有额外列（用于分组），但绝不带进模型输入 --
    X_all = df_full[feature_names].copy()

    # -- 防御：datetime 不允许 --
    bad_dt = X_all.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
    if bad_dt:
        raise ValueError(f"[step13] Found datetime cols in predict input: {bad_dt}")

    for g in groups:
        if g is None:
            idx = df_full.index
            g_name = "overall"
            g_col_name = "overall"
        else:
            mask = (df_full[group_col] == g)
            idx = df_full[mask].index
            g_name = str(g)
            g_col_name = group_col

        baseX = X_all.loc[idx].copy()
        n_rows = len(baseX)
        if n_rows == 0:
            continue
        if (group_col is not None) and (n_rows < int(min_group_rows)):
            continue

        # -- baseline：promo = 0 --
        X0 = baseX.copy()
        X0[promo_col] = 0.0
        y0 = float(booster.predict(X0).mean())

        for pv_log in promo_grid_log:
            X = baseX.copy()
            X[promo_col] = pv_log
            yp = float(booster.predict(X).mean())

            # -- 横轴统一 raw 语义 --
            if promo_col.endswith("_log"):
                promo_raw = float(np.expm1(pv_log))
            else:
                promo_raw = float(pv_log)

            lift = yp - y0
            lift_pct = 0.0 if abs(y0) < 1e-9 else lift / y0

            rows.append({
                "group_col": g_col_name,
                "group": g_name,
                "promo_value": promo_raw,          # 兼容旧字段
                "promo_intensity": promo_raw,      # 标准字段
                "promo_value_log": float(pv_log),
                "pred_sales": yp,                  # 兼容旧字段
                "pred_mean_sales": yp,             # 标准字段
                "lift_vs_zero": lift,
                "lift_pct_vs_zero": lift_pct,
                "n_rows": int(n_rows),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print(f"[WARN] empty curve output: {output_prefix} (maybe min_group_rows too large?)")
        return

    out = out.sort_values(["group", "promo_intensity"]).reset_index(drop=True)

    # -- sale abs plot --
    plt.figure(figsize=(8, 5))
    for g, sub in out.groupby("group"):
        plt.plot(sub["promo_intensity"], sub["pred_mean_sales"], marker="o", label=g)
    plt.xlabel("promo_intensity (raw)")
    plt.ylabel("Predicted mean weekly sales")
    plt.title(f"Promo response curve ({os.path.basename(output_prefix)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_abs.png")
    plt.close()

    # -- vs lift plot --
    plt.figure(figsize=(8, 5))
    for g, sub in out.groupby("group"):
        plt.plot(sub["promo_intensity"], sub["lift_vs_zero"], marker="o", label=g)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("promo_intensity (raw)")
    plt.ylabel("Lift vs promo=0")
    plt.title(f"Promo lift curve ({os.path.basename(output_prefix)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_lift.png")
    plt.close()

    out.to_csv(f"{output_prefix}.csv", index=False)
    print("[OK] saved:", output_prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="outputs/datasets/forecast_features_w1.csv")
    parser.add_argument("--model_path", type=str, default="outputs/models/lgbm_baseline.txt")
    parser.add_argument("--cat_vocab_path", type=str, default="outputs/models/cat_vocab.json")
    parser.add_argument("--split_date", type=str, default="2012-01-01")
    parser.add_argument("--output_dir", type=str, default="outputs/metrics")
    parser.add_argument("--min_group_rows", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -- load model + feature names (source of truth) --
    # -- 一定要保证和训练时一样 --
    booster = lgb.Booster(model_file=args.model_path)
    feature_names = booster.feature_name()

    # -- load cat vocab --
    with open(args.cat_vocab_path, "r", encoding="utf-8") as f:
        cat_vocab = json.load(f)

    # -- load feature data --
    df = pd.read_csv(args.feature_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # -- 在验证集做 --
    split_dt = pd.to_datetime(args.split_date)
    base_df = df[df["Date"] >= split_dt].copy()
    if len(base_df) == 0:
        raise ValueError("[step13] empty base_df after split_date filter. Check --split_date.")

    # -- 促销字段 --
    if "promo_intensity_log" in base_df.columns:
        promo_col = "promo_intensity_log"
        raw_promo = np.expm1(base_df[promo_col].astype(float).values)
    else:
        promo_col = "promo_intensity"
        raw_promo = base_df[promo_col].astype(float).values

    # -- 旋钮刻度 --
    qs = [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]
    promo_grid_raw = np.unique(np.quantile(raw_promo, qs))
    promo_grid_log = np.log1p(promo_grid_raw) if promo_col.endswith("_log") else promo_grid_raw

    print("[INFO] promo grid (raw):", promo_grid_raw)
    print("[INFO] promo grid (model input):", promo_grid_log)

    # -- 对齐 --
    for c, vocab in cat_vocab.items():
        if c not in base_df.columns:
            continue
        if c not in feature_names:
            # 训练模型没用到这个列，就不要强行搞成 category，避免引入 mismatch 风险
            continue

        s = base_df[c].astype("string").fillna(UNKNOWN)
        s = s.where(s.isin(vocab), UNKNOWN)
        base_df[c] = s.astype(CategoricalDtype(categories=vocab, ordered=False))

    # -- 创造仅仅用于分组的交叉字段 --
    if ("tier" in base_df.columns) and ("store_profile" in base_df.columns):
        base_df["tier_store_profile"] = (
            base_df["tier"].astype("string").fillna(UNKNOWN)
            + "@"
            + base_df["store_profile"].astype("string").fillna(UNKNOWN)
        )

    # -- 必须保证训练时的所有字段都在新的/验证的数据中有--
    missing_feats = [c for c in feature_names if c not in base_df.columns]
    if missing_feats:
        raise ValueError(f"[step13] base_df missing model features: {missing_feats[:10]} ... (total={len(missing_feats)})")

    # -- overall --
    simulate_response_curve(
        base_df, booster, feature_names, promo_grid_log, promo_col,
        group_col=None,
        output_prefix=os.path.join(args.output_dir, "step13_promo_curve_overall"),
        min_group_rows=args.min_group_rows,
    )

    # -- by tier --
    if "tier" in base_df.columns:
        simulate_response_curve(
            base_df, booster, feature_names, promo_grid_log, promo_col,
            group_col="tier",
            output_prefix=os.path.join(args.output_dir, "step13_promo_curve_by_tier"),
            min_group_rows=args.min_group_rows,
        )

    # -- by store_profile --
    if "store_profile" in base_df.columns:
        simulate_response_curve(
            base_df, booster, feature_names, promo_grid_log, promo_col,
            group_col="store_profile",
            output_prefix=os.path.join(args.output_dir, "step13_promo_curve_by_store_profile"),
            min_group_rows=args.min_group_rows,
        )

    # -- by tier_store_profile --
    if "tier_store_profile" in base_df.columns:
        simulate_response_curve(
            base_df, booster, feature_names, promo_grid_log, promo_col,
            group_col="tier_store_profile",
            output_prefix=os.path.join(args.output_dir, "step13_promo_curve_by_tier_store_profile"),
            min_group_rows=args.min_group_rows,
        )


if __name__ == "__main__":
    main()




'''
step15B: Robustness check by SALES STATE slice (low/mid/high based on sales_lag_1 quantiles)

目标：
    按 “销量状态” 把样本切成 low / mid / high
    在每个切片内做 promo 扫描（what-if），输出每个 group 的推荐档位
    额外输出：跨 low/mid/high 的稳定性对比表（避免人工逐行对比）
'''

import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.api.types import CategoricalDtype

UNKNOWN = "UNKNOWN"


def _align_categories_with_vocab(df: pd.DataFrame, vocab_path: str) -> pd.DataFrame:
    df = df.copy()
    if not os.path.exists(vocab_path):
        print(f"[WARN] vocab file not found: {vocab_path} (skip category alignment)")
        return df

    with open(vocab_path, "r", encoding="utf-8") as f:
        cat_vocab = json.load(f)

    for c, vocab in cat_vocab.items():
        if c not in df.columns:
            continue
        s = df[c].astype("string").fillna(UNKNOWN)
        s = s.where(s.isin(vocab), UNKNOWN)
        df[c] = s.astype(CategoricalDtype(categories=vocab, ordered=False))

    return df


def _get_promo_col_and_grid(
    base_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[str, np.ndarray, np.ndarray]:
    '''
    统一促销旋钮的入口函数（适配当前项目）

    约定：
        模型真实使用的促销特征为 `promo_intensity_log`
        promo_intensity_log = log1p(promo_intensity)
        step15 中仅做 what-if 扫描，不依赖 raw promo 列是否存在
    '''

    # 这里用 overall 的分位数作为“统一刻度”（金额模型下更合理）
    qs = np.array([0.0, 0.2, 0.5, 0.75, 0.9, 0.95, 0.99], dtype=float)

    if "promo_intensity_log" in feature_names:
        promo_col = "promo_intensity_log"
        if promo_col not in base_df.columns:
            raise ValueError(f"Missing column in base_df: {promo_col}")

        s = base_df[promo_col].astype(float).values
        s = s[np.isfinite(s)]
        if len(s) == 0:
            raise ValueError("promo_intensity_log has no finite values to build quantile grid.")

        promo_grid_model = np.nanquantile(s, qs).astype(float)
        promo_grid_model = np.unique(np.concatenate([np.array([0.0], dtype=float), promo_grid_model]))
        promo_grid_model = np.sort(promo_grid_model)

        promo_grid_raw = np.expm1(promo_grid_model).astype(float)
        return promo_col, promo_grid_raw, promo_grid_model

    if "promo_intensity" in feature_names:
        promo_col = "promo_intensity"
        if promo_col not in base_df.columns:
            raise ValueError(f"Missing column in base_df: {promo_col}")

        s = base_df[promo_col].astype(float).values
        s = s[np.isfinite(s)]
        if len(s) == 0:
            raise ValueError("promo_intensity has no finite values to build quantile grid.")

        promo_grid_raw = np.nanquantile(s, qs).astype(float)
        promo_grid_raw = np.unique(np.concatenate([np.array([0.0], dtype=float), promo_grid_raw]))
        promo_grid_raw = np.sort(promo_grid_raw)

        promo_grid_model = promo_grid_raw.copy()
        return promo_col, promo_grid_raw, promo_grid_model
    
    raise ValueError(
        f"Cannot find promo feature in model features. "
        f"Expect 'promo_intensity_log' or 'promo_intensity', "
        f"got: {feature_names}"
    )


def _ensure_tier_store_profile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if ("tier" in df.columns) and ("store_profile" in df.columns):
        df["tier_store_profile"] = (
            df["tier"].astype("string").fillna(UNKNOWN)
            + "@"
            + df["store_profile"].astype("string").fillna(UNKNOWN)
        )
    return df


def _simulate_curve_mean(
    booster: lgb.Booster,
    df: pd.DataFrame,
    feature_names: list[str],
    promo_col: str,
    promo_grid_model: np.ndarray,
    promo_grid_raw: np.ndarray,
    group_col: str | None,
    min_group_rows: int,
) -> pd.DataFrame:
    rows: list[dict] = []

    if group_col is None:
        groups = [("overall", df)]
        group_col_name = "overall"
    else:
        groups = [(str(g), gdf) for g, gdf in df.groupby(group_col, dropna=False, observed=True)]
        group_col_name = group_col

    for gname, gdf in groups:
        if len(gdf) == 0:
            continue
        if group_col is not None and len(gdf) < int(min_group_rows):
            continue

        X_base = gdf[feature_names].copy()

        X0 = X_base.copy()
        X0[promo_col] = 0.0
        y0 = float(booster.predict(X0).mean())

        for pv_model, pv_raw in zip(promo_grid_model, promo_grid_raw):
            X = X_base.copy()
            X[promo_col] = float(pv_model)
            yp = float(booster.predict(X).mean())

            lift = yp - y0
            lift_pct = 0.0 if abs(y0) < 1e-9 else lift / y0

            rows.append({
                "group_col": group_col_name,
                "group": gname,
                "promo_intensity": float(pv_raw),
                "pred_mean_sales": yp,
                "base_pred_at_zero": y0,
                "lift_vs_zero": lift,
                "lift_pct_vs_zero": lift_pct,
                "n_rows": int(len(gdf)),
            })

    return pd.DataFrame(rows)


def _summarize_plan(curve_df: pd.DataFrame, slice_name: str, state_col: str, q30: float, q70: float) -> pd.DataFrame:
    rows: list[dict] = []

    if curve_df is None or len(curve_df) == 0:
        return pd.DataFrame()

    for (group_col, group), gdf in curve_df.groupby(["group_col", "group"], observed=True):
        gdf = gdf.sort_values("promo_intensity").reset_index(drop=True)

        y0 = float(gdf["base_pred_at_zero"].iloc[0])

        best_idx = int(gdf["lift_vs_zero"].values.argmax())
        best = gdf.iloc[best_idx]

        promos = gdf["promo_intensity"].values.astype(float)
        lifts = gdf["lift_vs_zero"].values.astype(float)

        delta_lift = np.diff(lifts)
        delta_promo = np.diff(promos)
        marginal = np.divide(
            delta_lift,
            delta_promo,
            out=np.zeros_like(delta_lift),
            where=np.abs(delta_promo) > 1e-12,
        )

        turning_idx = None
        for i, m in enumerate(marginal, start=1):
            if m < 0:
                turning_idx = i
                break

        turning_promo = float(promos[turning_idx]) if turning_idx is not None else float(promos[-1])
        turning_marginal = float(marginal[turning_idx - 1]) if turning_idx is not None else float(marginal[-1] if len(marginal) else 0.0)

        rows.append({
            "slice": slice_name,
            "state_col": state_col,
            "q30": float(q30),
            "q70": float(q70),
            "group_col": str(group_col),
            "group": str(group),
            "n_rows": int(gdf["n_rows"].iloc[0]),
            "base_pred_at_zero": y0,
            "best_promo_intensity": float(best["promo_intensity"]),
            "best_lift_vs_zero": float(best["lift_vs_zero"]),
            "best_lift_pct_vs_zero": float(best["lift_pct_vs_zero"]),
            "turning_promo_intensity": turning_promo,
            "turning_marginal": turning_marginal,
        })

    return pd.DataFrame(rows)


def _compare_three_slices(plan_df: pd.DataFrame, promo_grid_raw: np.ndarray) -> pd.DataFrame:
    ''''
    跨 low/mid/high 汇总稳定性：
        best_promo 的 std / range
        best_lift_pct 的 std / range
        stable_flag：best_promo 的 range <= 1 个档位步长 且 best_lift_pct range <= 0.03
    '''
    if plan_df is None or len(plan_df) == 0:
        return pd.DataFrame()

    step = float(np.min(np.diff(np.sort(np.unique(promo_grid_raw))))) if len(np.unique(promo_grid_raw)) > 1 else 0.0

    agg = []
    for (gc, g), gdf in plan_df.groupby(["group_col", "group"], observed=True):
        slices = set(gdf["slice"].unique().tolist())
        if len(slices) < 2:
            continue

        bp = gdf["best_promo_intensity"].astype(float).values
        bl = gdf["best_lift_pct_vs_zero"].astype(float).values
        tp = gdf["turning_promo_intensity"].astype(float).values

        agg.append({
            "group_col": str(gc),
            "group": str(g),
            "n_slices": int(len(slices)),
            "best_promo_min": float(np.min(bp)),
            "best_promo_max": float(np.max(bp)),
            "best_promo_range": float(np.max(bp) - np.min(bp)),
            "best_promo_std": float(np.std(bp)),
            "best_lift_pct_min": float(np.min(bl)),
            "best_lift_pct_max": float(np.max(bl)),
            "best_lift_pct_range": float(np.max(bl) - np.min(bl)),
            "best_lift_pct_std": float(np.std(bl)),
            "turning_promo_min": float(np.min(tp)),
            "turning_promo_max": float(np.max(tp)),
            "turning_promo_range": float(np.max(tp) - np.min(tp)),
        })

    out = pd.DataFrame(agg)
    if len(out) == 0:
        return out

    out["stable_best_promo"] = out["best_promo_range"] <= step + 1e-12
    out["stable_turning_promo"] = out["turning_promo_range"] <= step + 1e-12
    out["stable_lift_pct"] = out["best_lift_pct_range"] <= 0.03  # 3% 绝对差
    out["stable_all"] = out["stable_best_promo"] & out["stable_turning_promo"] & out["stable_lift_pct"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="outputs/datasets/forecast_features_w1.csv")
    parser.add_argument("--model_path", type=str, default="outputs/models/lgbm_baseline.txt")
    parser.add_argument("--vocab_path", type=str, default="outputs/models/cat_vocab.json")
    parser.add_argument("--split_date", type=str, default="2012-01-01")
    parser.add_argument("--output_path", type=str, default="outputs/metrics/step15B_sales_state_robustness_plan.csv")
    parser.add_argument("--compare_path", type=str, default="outputs/metrics/step15B_sales_state_robustness_compare.csv")
    parser.add_argument("--summary_path", type=str, default="outputs/metrics/step15B_sales_state_robustness_summary.txt")
    parser.add_argument("--state_col", type=str, default="sales_lag_1")
    parser.add_argument("--q30", type=float, default=0.30)
    parser.add_argument("--q70", type=float, default=0.70)
    parser.add_argument("--min_group_rows", type=int, default=500)
    parser.add_argument("--topn_print", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    booster = lgb.Booster(model_file=args.model_path)
    feature_names = booster.feature_name()

    df = pd.read_csv(args.feature_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    base_df = df[df["Date"] >= pd.to_datetime(args.split_date)].copy()
    print("[INFO] base_df:", base_df.shape)

    base_df = _align_categories_with_vocab(base_df, args.vocab_path)
    base_df = _ensure_tier_store_profile(base_df)

    promo_col, promo_grid_raw, promo_grid_model = _get_promo_col_and_grid(base_df, feature_names)
    print("[INFO] promo grid (raw):", promo_grid_raw)
    print("[INFO] promo feature (model input):", promo_col)

    state_col = args.state_col
    if state_col not in base_df.columns:
        raise ValueError(f"Missing state_col: {state_col}")

    q30 = float(np.nanquantile(base_df[state_col].astype(float).values, args.q30))
    q70 = float(np.nanquantile(base_df[state_col].astype(float).values, args.q70))
    print(f"[INFO] {state_col} q30/q70:", q30, q70)

    slices = {
        "low_sales": base_df[base_df[state_col] <= q30].copy(),
        "mid_sales": base_df[(base_df[state_col] > q30) & (base_df[state_col] < q70)].copy(),
        "high_sales": base_df[base_df[state_col] >= q70].copy(),
    }

    group_cols: list[str | None] = [None]
    for c in ["tier", "store_profile", "tier_store_profile"]:
        if c in base_df.columns:
            group_cols.append(c)

    all_plans: list[pd.DataFrame] = []

    for sname, sdf in slices.items():
        print(f"[INFO] slice={sname} shape={sdf.shape}")
        if len(sdf) == 0:
            continue

        for gc in group_cols:
            curve = _simulate_curve_mean(
                booster, sdf, feature_names,
                promo_col, promo_grid_model, promo_grid_raw,
                group_col=gc,
                min_group_rows=args.min_group_rows,
            )
            plan = _summarize_plan(curve, sname, state_col, q30, q70)
            if len(plan):
                all_plans.append(plan)

    if not all_plans:
        print("[WARN] no plans generated")
        return

    out = pd.concat(all_plans, ignore_index=True)
    out.to_csv(args.output_path, index=False)
    print("[OK] saved:", args.output_path)

    cmp_df = _compare_three_slices(out, promo_grid_raw)
    if len(cmp_df):
        cmp_df.to_csv(args.compare_path, index=False)
        print("[OK] saved:", args.compare_path)
    else:
        print("[WARN] compare table is empty (need >=2 slices overlap per group)")

    lines = []
    if len(cmp_df):
        lines.append("step15B robustness summary (low/mid/high sales state)")
        for gc in sorted(cmp_df["group_col"].unique()):
            sub = cmp_df[cmp_df["group_col"] == gc]
            n = len(sub)
            stable = int(sub["stable_all"].sum())
            lines.append(f"- group_col={gc}: stable_all {stable}/{n} ({stable/n:.1%})")
        with open(args.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("[OK] saved:", args.summary_path)

    topn = int(args.topn_print)
    for gc in out["group_col"].unique():
        sub = out[out["group_col"] == gc].sort_values(["slice", "best_lift_vs_zero"], ascending=[True, False])
        print(f"\n=== TOP {topn} by {gc} (per slice) ===")
        for s in sub["slice"].unique():
            print(f"[{s}]")
            print(sub[sub["slice"] == s].head(topn))


if __name__ == "__main__":
    main()



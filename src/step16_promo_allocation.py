'''
step16: Promo allocation under budget constraint (greedy)

Inputs:
    outputs/datasets/forecast_features_w1.csv
    outputs/datasets/dept_promo_responsiveness.csv  (from step8)
    outputs/metrics/step13_promo_curve_by_tier_store_profile.csv (preferred)
  fallback:
    outputs/metrics/step13_promo_curve_by_tier.csv
    outputs/metrics/step13_promo_curve_by_store_profile.csv

Output:
    outputs/metrics/step16_promo_allocation.csv
'''

import os#
import argparse
import numpy as np
import pandas as pd
import heapq


UNKNOWN = "UNKNOWN"

# -- 从响应强度里面拿到对应的最高的lift分位 --
def _pick_best_from_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    '''
    给每个 group（例如 A_core@High_profile）挑一个推荐点：
        promo=0 作为 baseline
        在 lift_vs_zero 最大处作为 best
        turning_promo_intensity：第一个出现边际变负的位置（简化版）
    '''
    rows = []
    curve_df = curve_df.sort_values(["group", "promo_intensity"]).copy()

    for g, gdf in curve_df.groupby("group"):
        gdf = gdf.sort_values("promo_intensity").copy()
        if len(gdf) < 2:
            continue

        # -- baseline: promo=0（如果不存在，就用最小 promo 近似）--
        base = gdf.iloc[0]
        base_sales = float(base["pred_mean_sales"])

        # -- best point: max lift --
        best_idx = int(np.argmax(gdf["lift_vs_zero"].values.astype(float)))
        best = gdf.iloc[best_idx]

        promos = gdf["promo_intensity"].values.astype(float)
        lifts = gdf["lift_vs_zero"].values.astype(float)

        # -- turning: first time marginal becomes negative --
        delta_lift = np.diff(lifts)
        delta_promo = np.diff(promos)
        marginal = np.divide(delta_lift, delta_promo, out=np.zeros_like(delta_lift), where=delta_promo != 0)

        turning = np.nan
        for i, m in enumerate(marginal):
            if m < 0:
                turning = float(promos[i + 1])
                break

        rows.append({
            "group": g,
            "base_pred_at_zero": base_sales,
            "best_promo_intensity": float(best["promo_intensity"]),
            "best_lift_vs_zero": float(best["lift_vs_zero"]),
            "best_lift_pct_vs_zero": float(best["lift_pct_vs_zero"]),
            "turning_promo_intensity": turning,
            "n_rows": int(best.get("n_rows", 0)),
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out = out.sort_values("best_lift_vs_zero", ascending=False).reset_index(drop=True)
    return out

# -- 加载促销响应强度曲线 --
def _load_curve_any(paths: list[str]) -> tuple[pd.DataFrame, str]:
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            # 兼容字段名：我们要求至少有 group + promo_intensity + pred_mean_sales + lift_vs_zero
            must = {"group", "promo_intensity", "pred_mean_sales", "lift_vs_zero", "lift_pct_vs_zero"}
            if not must.issubset(set(df.columns)):
                raise ValueError(f"[step16] curve file {p} missing required columns: {must - set(df.columns)}")
            return df, p
    raise FileNotFoundError(f"[step16] No curve files found in: {paths}")


def _build_group_curve_lookup(curve_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    '''
        把 step13 曲线整理成：group -> 按 promo_intensity 排序的曲线表
        要求字段：promo_intensity, lift_vs_zero, lift_pct_vs_zero, pred_mean_sales
    '''
    out: dict[str, pd.DataFrame] = {}
    curve_df = curve_df.copy()
    curve_df["group"] = curve_df["group"].astype("string")
    curve_df["promo_intensity"] = curve_df["promo_intensity"].astype(float)
    curve_df["lift_vs_zero"] = curve_df["lift_vs_zero"].astype(float)

    for g, gdf in curve_df.groupby("group", observed=True):
        gdf = gdf.sort_values("promo_intensity").reset_index(drop=True)
        if len(gdf) < 2:
            continue
        out[str(g)] = gdf

    return out


def _turning_from_curve(gdf: pd.DataFrame) -> float:
    '''
        turning：第一个出现边际变负的位置（简化版）
        若从未变负，则返回最后一个 promo 档位
    '''
    promos = gdf["promo_intensity"].values.astype(float)
    lifts = gdf["lift_vs_zero"].values.astype(float)
    if len(promos) < 2:
        return float(promos[-1]) if len(promos) else np.nan

    delta_lift = np.diff(lifts)
    delta_promo = np.diff(promos)
    marginal = np.divide(delta_lift, delta_promo, out=np.zeros_like(delta_lift), where=delta_promo != 0)

    for i, m in enumerate(marginal):
        if m < 0:
            return float(promos[i + 1])
    return float(promos[-1])

# -- 最大的安全区 --
def _safe_max_index(promos: np.ndarray, turning: float) -> int:
    """
    安全区：promo <= turning
    返回最大可用的档位 index
    """
    if turning is None or (isinstance(turning, float) and np.isnan(turning)):
        return int(len(promos) - 1)
    ok = np.where(promos <= float(turning) + 1e-12)[0]
    if len(ok) == 0:
        return 0
    return int(ok[-1])


def main():
    # -- 命令行参数 --
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, default="outputs/datasets/forecast_features_w1.csv")
    parser.add_argument("--dept_pool_path", type=str, default="outputs/datasets/dept_promo_responsiveness.csv") # step 8
    parser.add_argument("--curve_dir", type=str, default="outputs/metrics")
    parser.add_argument("--split_date", type=str, default="2012-01-01", help="use validation period as base universe")
    parser.add_argument("--max_depts", type=int, default=30, help="top K depts from step8 pool")
    parser.add_argument("--max_stores", type=int, default=45, help="top N stores by recent sales to limit size")
    parser.add_argument("--budget", type=float, default=2.0e6, help="total promo_intensity budget (raw units)")
    parser.add_argument("--min_lift", type=float, default=0.0, help="only allocate if lift > min_lift")
    parser.add_argument("--output_path", type=str, default="outputs/metrics/step16_promo_allocation.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # -- 读取特征数据 --
    df = pd.read_csv(args.feature_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    split_dt = pd.to_datetime(args.split_date)

    base_df = df[df["Date"] >= split_dt].copy() # 只用验证集
    print("[INFO] base_df:", base_df.shape)

    # -- 建立候选商品池 --
    pool = pd.read_csv(args.dept_pool_path)
    # -- step 8 中只拿 eligible 的 + score 高的 --
    pool = pool[pool.get("eligible", True) == True].copy()
    if "score" in pool.columns:
        pool = pool.sort_values("score", ascending=False)
    top_depts = pool["Dept"].dropna().astype(int).unique().tolist()[: int(args.max_depts)]
    print("[INFO] top_depts:", len(top_depts))

    # -- 只要销量高的前K个商店进行促销 --
    # 用验证期 baseline_sales 近似：用 Weekly_Sales 的均值/和都行，这里用 mean
    if "Weekly_Sales" in base_df.columns:
        store_rank = (
            base_df.groupby("Store")["Weekly_Sales"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        top_stores = store_rank["Store"].head(int(args.max_stores)).tolist()
    else:
        top_stores = sorted(base_df["Store"].dropna().unique().tolist())[: int(args.max_stores)]
    print("[INFO] top_stores:", len(top_stores))
    # -- 总的候选池 --
    universe = base_df[
        base_df["Dept"].isin(top_depts) & base_df["Store"].isin(top_stores)
    ].copy()
    if len(universe) == 0:
        raise ValueError("[step16] universe empty after filtering by top_depts/top_stores")

    # -- 取每个 Store×Dept 的“代表行”（用最近一条记录作为状态）--
    universe = universe.sort_values(["Store", "Dept", "Date"]).copy()
    rep = universe.groupby(["Store", "Dept"]).tail(1).copy()
    rep["tier"] = rep.get("tier", UNKNOWN).astype("string").fillna(UNKNOWN)
    rep["store_profile"] = rep.get("store_profile", UNKNOWN).astype("string").fillna(UNKNOWN)
    rep["tier_store_profile"] = rep["tier"] + "@" + rep["store_profile"]
    print("[INFO] candidate combos (Store×Dept):", rep.shape[0])

    # --加载曲线--
    curve_paths = [
        os.path.join(args.curve_dir, "step13_promo_curve_by_tier_store_profile.csv"),
        os.path.join(args.curve_dir, "step13_promo_curve_by_tier.csv"),
        os.path.join(args.curve_dir, "step13_promo_curve_by_store_profile.csv"),
        os.path.join(args.curve_dir, "step13_promo_curve_overall.csv"),
    ]
    curve_df, used_curve_path = _load_curve_any(curve_paths)
    print("[INFO] use curve:", used_curve_path)

    # -- 识别这份 curve 的 group 对应的是哪个维度（最稳：看 group_col 列）
    group_col_mode = None
    if "group_col" in curve_df.columns:
        # -- 例如 group_col 可能是 "tier_store_profile" / "tier" / "store_profile" / "overall" --
        group_col_mode = str(curve_df["group_col"].iloc[0])
    else:
        # -- 没有就猜：如果 group 里包含 '@'，认为是 tier_store_profile --
        sample_g = str(curve_df["group"].iloc[0])
        group_col_mode = "tier_store_profile" if "@" in sample_g else "tier"
    print("[INFO] curve group mode:", group_col_mode)
    # -- 选择一条较好的曲线 --
    curve_summary = _pick_best_from_curve(curve_df)
    if len(curve_summary) == 0:
        raise ValueError("[step16] curve_summary empty, check step13 output")

    # -- 固定类别名 --
    if group_col_mode == "tier_store_profile":
        rep["curve_group"] = rep["tier_store_profile"]
    elif group_col_mode == "tier":
        rep["curve_group"] = rep["tier"]
    elif group_col_mode == "store_profile":
        rep["curve_group"] = rep["store_profile"]
    else:
        rep["curve_group"] = "overall"

    plan = rep.merge(
        curve_summary.rename(columns={"group": "curve_group"}),
        on="curve_group",
        how="left",
    )

    # -- 不行就用 overall  --
    if plan["best_promo_intensity"].isna().any():
        overall_row = curve_summary[curve_summary["group"] == "overall"]
        if len(overall_row) > 0:
            overall = overall_row.iloc[0].to_dict()
            fill_cols = ["base_pred_at_zero", "best_promo_intensity", "best_lift_vs_zero",
                         "best_lift_pct_vs_zero", "turning_promo_intensity"]
            for c in fill_cols:
                plan[c] = plan[c].fillna(overall.get(c, np.nan))

    # -- 若 turning 存在 -> 推荐强度不超过 turning --
    # --  后面按边际收益逐档加码 --
    plan["recommended_promo_intensity"] = 0.0

    # -- 成本：用 promo_intensity 本身作为 budget --
    plan["promo_cost"] = 0.0

    # -- ROI：lift / cost（cost=0 时设 0） --
    plan["roi_lift_per_cost"] = 0.0

    # -- 构建 group -> 曲线查找表（用于“逐档加码”）--
    group_curve_lookup = _build_group_curve_lookup(curve_df)

    # -- 若某些 group 在曲线里找不到，则 fallback 到 overall（如果存在）--
    has_overall_curve = "overall" in group_curve_lookup
    if not has_overall_curve:
        # 如果 curve_summary 有 overall，但曲线文件没 overall（极少见），就只能跳过这些 group
        pass

    # -- 全局边际 ROI 贪心加码，直到预算用完 --
    budget = float(args.budget)
    used = 0.0

    # --记录每个 (Store,Dept) 当前档位 index / 当前promo / 当前lift --
    # -- key: (Store, Dept) --
    state = {}

    # -- 用一个堆存所有候选的“下一步升级”：(-marginal_roi, -delta_lift, step_id, key) --
    heap = []
    step_seq = 0

    # -- 先为每个组合准备曲线 & 安全上限 & 从0档开始的“第一步升级” --
    for i, r in plan.iterrows():
        st = int(r["Store"])
        dp = int(r["Dept"])
        key = (st, dp)

        # -- 如果lift太小就不推荐进行促销 --
        if float(r["best_lift_vs_zero"]) <= float(args.min_lift):
            state[key] = {
                "idx": 0, "promo": 0.0, "lift": 0.0,
                "turning": float(r["turning_promo_intensity"]) if pd.notna(r["turning_promo_intensity"]) else np.nan,
                "promos": None, "lifts": None,
                "max_idx": 0,
                "picked": [],
            }
            continue

        gname = str(r["curve_group"])
        gdf = group_curve_lookup.get(gname)

        if gdf is None and has_overall_curve:
            gname = "overall"
            gdf = group_curve_lookup.get("overall")

        if gdf is None:
            # 没有曲线，跳过
            state[key] = {
                "idx": 0, "promo": 0.0, "lift": 0.0,
                "turning": float(r["turning_promo_intensity"]) if pd.notna(r["turning_promo_intensity"]) else np.nan,
                "promos": None, "lifts": None,
                "max_idx": 0,
                "picked": [],
            }
            continue

        promos = gdf["promo_intensity"].values.astype(float)
        lifts = gdf["lift_vs_zero"].values.astype(float)

        # -- turning：优先用 curve_summary 里算好的；若缺失则从曲线再算一次 --
        turning = float(r["turning_promo_intensity"]) if pd.notna(r["turning_promo_intensity"]) else _turning_from_curve(gdf)
        max_idx = _safe_max_index(promos, turning)

        # -- 强制从 index=0 视为 baseline（promo=0 若没有也用最小档）--
        cur_idx = 0
        cur_promo = float(promos[cur_idx])
        cur_lift = float(lifts[cur_idx])

        state[key] = {
            "idx": cur_idx,
            "promo": cur_promo,
            "lift": cur_lift,
            "turning": turning,
            "promos": promos,
            "lifts": lifts,
            "max_idx": max_idx,
            "picked": [],  # 记录升级步骤（用于 budget_used_so_far）
        }

        # -- 如果 baseline 档位不是 0，允许从 baseline -> 下一档仍按增量算 --
        if cur_idx < max_idx:
            nxt = cur_idx + 1
            delta_cost = float(promos[nxt] - promos[cur_idx])
            delta_lift = float(lifts[nxt] - lifts[cur_idx])
            if delta_cost > 1e-12:
                marginal_roi = delta_lift / delta_cost
                # 只推“有意义”的升级：允许负的也推？这里不推负的，避免越投越亏
                if marginal_roi > 0:
                    heapq.heappush(heap, (-marginal_roi, -delta_lift, step_seq, key))
                    step_seq += 1

    # -- 开始全局贪心：每次挑“下一步边际 ROI”最高的升级 --
    while heap:
        neg_roi, neg_dl, _, key = heapq.heappop(heap)
        marginal_roi = -float(neg_roi)

        info = state.get(key)
        if info is None:
            continue

        cur_idx = int(info["idx"])
        promos = info["promos"]
        lifts = info["lifts"]
        max_idx = int(info["max_idx"])

        if promos is None or lifts is None:
            continue
        if cur_idx >= max_idx:
            continue

        nxt = cur_idx + 1
        delta_cost = float(promos[nxt] - promos[cur_idx])
        delta_lift = float(lifts[nxt] - lifts[cur_idx])

        # -- 预算是否够这一步升级 --
        if used + delta_cost > budget:
            continue

        # -- 执行升级 --
        used += delta_cost
        info["idx"] = nxt
        info["promo"] = float(promos[nxt])
        info["lift"] = float(lifts[nxt])
        info["picked"].append({"delta_cost": delta_cost, "delta_lift": delta_lift})

        # -- 推入下一步升级 --
        if nxt < max_idx:
            nxt2 = nxt + 1
            delta_cost2 = float(promos[nxt2] - promos[nxt])
            delta_lift2 = float(lifts[nxt2] - lifts[nxt])
            if delta_cost2 > 1e-12:
                marginal_roi2 = delta_lift2 / delta_cost2
                if marginal_roi2 > 0:
                    heapq.heappush(heap, (-marginal_roi2, -delta_lift2, step_seq, key))
                    step_seq += 1

    # -- 把分配结果写回 plan --
    final_promos = []
    final_costs = []
    final_lifts = []
    final_rois = []
    allocated_flags = []
    step_counts = []

    for _, r in plan.iterrows():
        key = (int(r["Store"]), int(r["Dept"]))
        info = state.get(key)
        if info is None:
            p = 0.0
            l = 0.0
            c = 0.0
            nst = 0
        else:
            p = float(info["promo"])
            l = float(info["lift"])
            c = float(info["promo"])  # 这里保持你原先的“成本=promo强度本身”的口径
            nst = int(len(info.get("picked", [])))

        final_promos.append(p)
        final_lifts.append(l)
        final_costs.append(c)
        final_rois.append(0.0 if c <= 0 else (l / c))
        allocated_flags.append(bool(c > 0))
        step_counts.append(nst)

    plan["recommended_promo_intensity"] = final_promos
    plan["promo_cost"] = np.array(final_costs, dtype=float)
    plan["roi_lift_per_cost"] = np.array(final_rois, dtype=float)

    # -- 这里做一个累加 就是到了这个dept花了多少钱了  --
    # 说明：方案A是“升级序列”式贪心，plan 行顺序本身不等于升级顺序；
    #      这里用“按 ROI 再排序”的顺序做累计展示，便于阅读。
    plan["allocated"] = allocated_flags
    plan = plan.sort_values(["roi_lift_per_cost", "best_lift_vs_zero"], ascending=False).reset_index(drop=True)
    plan["budget_used_so_far"] = plan["promo_cost"].where(plan["allocated"], 0.0).cumsum()

    # -- 如果我推荐的强度 已经接近边际为 0 的 turning point 那我告诉你：这条有风险 --
    plan["risk_flag"] = ""
    plan.loc[plan["turning_promo_intensity"].notna() &
             (plan["recommended_promo_intensity"] >= 0.95 * plan["turning_promo_intensity"]),
             "risk_flag"] = "NEAR_TURNING"

    # -- 输出列 --
    out_cols = [
        "Store", "Dept", "tier", "store_profile", "tier_store_profile",
        "curve_group",
        "base_pred_at_zero", "best_lift_vs_zero", "best_lift_pct_vs_zero",
        "best_promo_intensity", "turning_promo_intensity",
        "recommended_promo_intensity", "promo_cost", "roi_lift_per_cost",
        "allocated", "budget_used_so_far", "risk_flag",
    ]
    out_cols = [c for c in out_cols if c in plan.columns]
    out = plan[out_cols].copy()

    out.to_csv(args.output_path, index=False)
    print("[OK] saved:", args.output_path)
    print("[INFO] total budget:", budget, "used:", float(out.loc[out["allocated"], "promo_cost"].sum()))
    print(out.head(10))


if __name__ == "__main__":
    main()


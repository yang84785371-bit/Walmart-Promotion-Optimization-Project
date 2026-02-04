'''
    step17: Generate execution sheet for ops (store×dept action list)

    Input:
        outputs/metrics/step16_promo_allocation.csv

    Output:
        outputs/metrics/step17_execution_sheet.csv
'''

import os
import argparse
import pandas as pd
import numpy as np


def main():
    # -- 命令行参数 --
    parser = argparse.ArgumentParser()
    parser.add_argument("--allocation_path", type=str, default="outputs/metrics/step16_promo_allocation.csv")
    parser.add_argument("--output_path", type=str, default="outputs/metrics/step17_execution_sheet.csv")
    parser.add_argument("--only_allocated", action="store_true", help="only export allocated=True rows")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    df = pd.read_csv(args.allocation_path)
    if args.only_allocated and "allocated" in df.columns:
        df = df[df["allocated"] == True].copy()

    if len(df) == 0:
        raise ValueError("[step17] empty after filtering, check step16 outputs / --only_allocated")

    # -- 对接运营的文案
    def make_note(r):
        promo = float(r.get("recommended_promo_intensity", 0.0))
        lift = float(r.get("best_lift_vs_zero", 0.0))
        risk = str(r.get("risk_flag", ""))

        if promo <= 0 or lift <= 0:
            return "建议不投放促销：模型在该状态下预期 lift≤0，避免无效促销。"

        if risk == "NEAR_TURNING":
            return "谨慎投放：已接近边际拐点（turning），建议小步试探+监控销量波动。"

        return "可投放：处于边际空间区间，建议按推荐强度投放并监控销量/库存。"

    df["action_note"] = df.apply(make_note, axis=1)

    # -- 风险等级（便于排序） --
    df["risk_level"] = 0
    df.loc[df.get("risk_flag", "") == "NEAR_TURNING", "risk_level"] = 2
    df.loc[(df["recommended_promo_intensity"] <= 0) | (df["best_lift_vs_zero"] <= 0), "risk_level"] = 3

    # -- 输出字段 --
    out_cols = [
        "Store", "Dept", "tier", "store_profile",
        "recommended_promo_intensity",
        "best_lift_vs_zero", "best_lift_pct_vs_zero",
        "turning_promo_intensity",
        "allocated", "promo_cost",
        "risk_flag", "risk_level",
        "action_note",
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    out = df[out_cols].copy()

    # -- 先看 allocated，再看 lift，再看风险 --
    sort_cols = []
    if "allocated" in out.columns:
        sort_cols.append("allocated")
    if "best_lift_vs_zero" in out.columns:
        sort_cols.append("best_lift_vs_zero")
    if "risk_level" in out.columns:
        sort_cols.append("risk_level")

    if sort_cols:
        ascending = [False] * len(sort_cols)
        # -- risk_level 越小越好 --
        if "risk_level" in sort_cols:
            ascending[sort_cols.index("risk_level")] = True
        out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    out.to_csv(args.output_path, index=False)
    print("[OK] saved:", args.output_path)
    print(out.head(15))


if __name__ == "__main__":
    main()

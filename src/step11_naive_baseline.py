import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # ---- 命令行参数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="outputs/datasets/forecast_features_w1.csv",
        help="feature table with lag features and y_next_week"
    )
    parser.add_argument(
        "--split_date",
        type=str,
        default="2012-01-01",
        help="time split for train / validation"
    )
    args = parser.parse_args()

    # ---- 加载数据 ----
    df = pd.read_csv(args.data_path)
    print("[INFO] loaded feature table:", df.shape)

    # -- 防一手 --
    # -- 这几个一定要有 --
    if "Date" not in df.columns:
        raise ValueError("Missing column: Date")
    if "y_next_week" not in df.columns:
        raise ValueError("Missing column: y_next_week")
    if "sales_lag_1" not in df.columns:
        raise ValueError("Missing column: sales_lag_1")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # -- data 用不着 只是用来切分训练集和验证集--
    split_dt = pd.to_datetime(args.split_date)
    val_df = df[df["Date"] >= split_dt].copy()

    if len(val_df) == 0:
        raise ValueError("Validation set is empty. Check split_date.")

    # -- 朴素基线：y_hat = sales_lag_1 --
    y_true = val_df["y_next_week"]
    y_pred = val_df["sales_lag_1"]

    # -- 由于用到了 lag1 所以前面几周会有na 直接 过滤掉--
    mask = y_true.notna() & y_pred.notna()
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No valid rows after NaN filtering for naive baseline.")

    # -- 评估 --
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"[NAIVE BASELINE] MAE (lag1):  {mae:.2f}")
    print(f"[NAIVE BASELINE] RMSE (lag1): {rmse:.2f}")

    # -- 保存结果 --
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/step11_naive_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"data_path={args.data_path}\n")
        f.write(f"split_date={args.split_date}\n")
        f.write(f"n_val={len(y_true)}\n")
        f.write(f"MAE={mae:.4f}\n")
        f.write(f"RMSE={rmse:.4f}\n")

    print("[OK] naive baseline metrics saved to outputs/metrics/step11_naive_metrics.txt")


if __name__ == "__main__":
    main()


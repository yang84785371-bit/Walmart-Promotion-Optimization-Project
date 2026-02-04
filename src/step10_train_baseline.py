import os
import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


UNKNOWN = "UNKNOWN"


def _prepare_categorical_with_unknown(train_df: pd.DataFrame, val_df: pd.DataFrame, cat_cols):
    """
    关键：真正做到 unseen -> UNKNOWN
    - 词表（categories）以训练集为准 + UNKNOWN
    - 验证集/未来数据中不在词表的值，统一映射为 UNKNOWN
    - train/val 使用同一个 categories，保证语义空间一致
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    vocabs = {}  # 保存每个列的训练词表（可选：后续想保存也行）

    for c in cat_cols:
        if c not in train_df.columns:
            continue

        # -- 统一成 string，缺失填 UNKNOWN
        train_s = train_df[c].astype("string").fillna(UNKNOWN)
        val_s = val_df[c].astype("string").fillna(UNKNOWN) if c in val_df.columns else pd.Series([UNKNOWN] * len(val_df), index=val_df.index) # 防御性一下而已 别想太多

        # -- 训练词表：训练集中出现过的 + UNKNOWN（强制加入）-- 
        known = pd.Index(sorted(train_s.unique().astype(str)))
        if UNKNOWN not in known:
            known = known.append(pd.Index([UNKNOWN])) # 强制把 UNKNOWN 加进合法世界

        # -- 验证/未来：不在 known 的，全部 -> UNKNOWN --
        val_s = val_s.where(val_s.isin(known), UNKNOWN)

        # -- 这个列的世界观只有这些值，别的都不允许-- 
        cat_dtype = pd.api.types.CategoricalDtype(categories=list(known), ordered=False)
        # -- 强制 train / val 都用这套格式 --
        train_df[c] = train_s.astype(cat_dtype)
        val_df[c] = val_s.astype(cat_dtype)

        vocabs[c] = list(known)

    return train_df, val_df, vocabs


def main():
    # -- 命令行参数 --
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="outputs/datasets/forecast_features_w1.csv")
    parser.add_argument("--split_date", type=str, default="2012-01-01")
    parser.add_argument("--y_clip_q", type=float, default=0.005, help="clip y at [q, 1-q], set 0 to disable")
    parser.add_argument("--nonneg_pred", action="store_true", help="clip predictions to >=0")
    args = parser.parse_args()

    # -- 读取 feature --
    df = pd.read_csv(args.data_path)
    print("[INFO] loaded feature table:", df.shape)

    # -- Date 必须存在 --
    if "Date" not in df.columns:
        raise ValueError("Missing column: Date")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()

    # -- target 必须存在 --
    target = "y_next_week"
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    # ---- time split 切分训练集和验证集  ----
    split_dt = pd.to_datetime(args.split_date)
    train_df = df[df["Date"] < split_dt].copy()
    val_df = df[df["Date"] >= split_dt].copy()

    print("[INFO] train shape:", train_df.shape)
    print("[INFO] valid shape:", val_df.shape)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Empty train/val after split. Check --split_date and Date range.")

    # -- categorical columns (重点：unseen -> UNKNOWN) --
    cat_candidates = ["Store", "Dept", "tier", "store_profile"]
    cat_cols = [c for c in cat_candidates if c in train_df.columns]  # 只以 train 为准决定
    train_df, val_df, vocabs = _prepare_categorical_with_unknown(train_df, val_df, cat_cols) # 获得三个集合 一个是训练的 一个是验证 一个是know的字典

    # -- features & labels --
    drop_cols = ["Date", target] # 丢掉一些不能进模型的东西 前者是有更好的时间特征 后者是标签
    # -- 这里是构造自变量和因变量
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns]).reset_index(drop=True)
    y_train = train_df[target].astype(float).reset_index(drop=True)

    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns]).reset_index(drop=True)
    y_val = val_df[target].astype(float).reset_index(drop=True)

    # -- DEBUG：确认没有 object LightGBM 最怕 object --
    bad = X_train.select_dtypes(include=["object"]).columns.tolist() # 检查类型 不允许出现object
    if bad:
        raise ValueError(f"Found object cols in X_train: {bad}")

    # -- 训练时传 categorical_feature 列名 -- 
    real_cat_cols = [c for c in cat_cols if c in X_train.columns and str(X_train[c].dtype) == "category"] # 这里是防御性写法 确保类别变量一定是category
    print("[DEBUG] categorical cols in X_train:", real_cat_cols)
    for c in real_cat_cols:
        print(f"[DEBUG] {c} vocab size:", len(vocabs.get(c, [])))

    # -- 对训练标签去极值（winsorize） --
    if args.y_clip_q and args.y_clip_q > 0:
        q = float(args.y_clip_q)
        lo = y_train.quantile(q)
        hi = y_train.quantile(1 - q)
        y_train = y_train.clip(lower=lo, upper=hi)

    # -- 建一个 LightGBM 回归模型 --
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1
    )
    # -- 拟合模型 --
    model.fit(
        X_train, y_train,
        categorical_feature=real_cat_cols
    )

    # -- 用val集进行eval --
    y_pred = model.predict(X_val)
    if args.nonneg_pred:
        y_pred = np.maximum(y_pred, 0.0) # 如果开了就把所有负预测值变成 0

    # -- 计算指标 --    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    print(f"[RESULT] MAE:  {mae:.2f}")
    print(f"[RESULT] RMSE: {rmse:.2f}")

    # -- 保存 --
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    model.booster_.save_model("outputs/models/lgbm_baseline.txt")

    with open("outputs/metrics/step10_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"data_path={args.data_path}\n")
        f.write(f"split_date={args.split_date}\n")
        f.write(f"y_clip_q={args.y_clip_q}\n")
        f.write(f"nonneg_pred={bool(args.nonneg_pred)}\n")
        f.write(f"n_train={len(X_train)}\n")
        f.write(f"n_val={len(X_val)}\n")
        f.write(f"n_features={X_train.shape[1]}\n")
        f.write(f"categorical={real_cat_cols}\n")
        f.write(f"MAE={mae:.4f}\n")
        f.write(f"RMSE={rmse:.4f}\n")

    print("[OK] model & metrics saved.")
    print("[INFO] categorical_feature:", real_cat_cols)

    # ===== save categorical vocab =====
    cat_vocab = {}

    for c in real_cat_cols:
        cat_vocab[c] = list(X_train[c].cat.categories)

    import json
    os.makedirs("outputs/models", exist_ok=True)
    with open("outputs/models/cat_vocab.json", "w", encoding="utf-8") as f:
        json.dump(cat_vocab, f, ensure_ascii=False, indent=2)

    print("[OK] saved categorical vocab to outputs/models/cat_vocab.json")



if __name__ == "__main__":
    main()



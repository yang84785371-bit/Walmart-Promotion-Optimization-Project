import os
import pandas as pd
import lightgbm as lgb
'''
    这里我们需要取得到每个特征对销量的贡献
'''

def main():
    model_path = "outputs/models/lgbm_baseline.txt"
    output_path = "outputs/metrics/feature_importance.csv"

    # -- 加载模型 --
    booster = lgb.Booster(model_file=model_path)

    # -- 得到feature name --
    feature_names = booster.feature_name()

    # ---- importance ----
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")

    out = pd.DataFrame({
        "feature": feature_names,
        "importance_gain": imp_gain,
        "importance_split": imp_split
    }).sort_values("importance_gain", ascending=False)

    # ---- diagnostics ----
    print("[INFO] top 15 features by gain:")
    print(out.head(15))

    # 特别关注 promo_intensity
    promo_rows = out[out["feature"].str.contains("promo", case=False, regex=True)]
    if len(promo_rows) > 0:
        print("\n[INFO] promo-related features:")
        print(promo_rows)
    else:
        print("\n[WARN] no promo-related feature found in importance list")

    # ---- save ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)

    print("[OK] saved feature importance to:", output_path)


if __name__ == "__main__":
    main()

'''
    从结果来说还是可以的 首先促销进入了前15 
    总体来说 销量和 往期销量有很大关系 之后就是 商店 商品以及 商品画像
    其次是时间属性
    最后菜蔬 汽油价格 温度 门店 促销 这些
    总体来说还是很健康

    销量惯性（lag / rolling） —— 物理规律级

    品类 & 门店结构（Dept / Store / tier） —— 结构性约束

    时间周期（week / seasonality） —— 年度节律

    促销（promo） —— 边际调节器


'''


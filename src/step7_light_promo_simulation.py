"""
step7 是一个规则型的 what-if 粗筛模块，用历史数据做“无促销 → 强促销”的条件对比，判断每个 Dept 是否“可能对促销有响应”，为后续更稳的响应分层与分配做过滤，不用于最终决策。
"""
import pandas as pd
import argparse
import os

# -- 命令行参数 --
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="outputs/datasets/walmart_clean.csv")
parser.add_argument("--tier_path", type=str, default="outputs/datasets/dept_assortment_tiers.csv")
parser.add_argument("--output_path", type=str, default="outputs/datasets/promo_simulation_ab.csv")
args = parser.parse_args()

# --------- 可调参数（尽量少） ----------
CLIP_Q = 0.01          # winsorize: 1%/99%（只对 Weekly_Sales 做）
HIGH_Q = 0.67          # 强促销分位：只在 promo>0 的周里取这个分位
MIN_BUCKET_N = 8       # baseline(0促销) 和 high(强促销) 桶最少周数
EPS = 1e-9
# --------------------------------------

# -- 加载 --
df = pd.read_csv(args.data_path)
tiers = pd.read_csv(args.tier_path)

# -- 只留AB的dept --
ab_depts = tiers[tiers["tier"].isin(["A_core", "B_holiday"])]["Dept"].unique()
df = df[df["Dept"].isin(ab_depts)].copy()

# -- 模拟促销强度的proxy --
md_cols = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
df["promo_intensity"] = df[md_cols].sum(axis=1)

def _clip_series(s: pd.Series, q: float) -> pd.Series:
    """winsorize: clamp to [q, 1-q] quantiles (do NOT drop rows)."""
    lo = s.quantile(q)
    hi = s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)

rows = []
for dept, g in df.groupby("Dept"):
    g = g.copy()

    # -- 1) 护栏：异常值被限制影响力 --
    g["Weekly_Sales"] = _clip_series(g["Weekly_Sales"], CLIP_Q)

    # -- 2) 定义 baseline / promo 周 --
    # -- 0代表没有促销 有数字代表有促销 它是离散的变量 将其变成连续的会出错 所以我们需要进行分层--
    no_promo = g["promo_intensity"] <= 0
    has_promo = g["promo_intensity"] > 0

    # -- 3) 数据不足：保守为 0  这里是一些保守的规则 --
    # -- 这里的护栏挺重要的 就是为了避免某些小样本看起来很猛 但其实是噪声 --
    if (no_promo.sum() < MIN_BUCKET_N) or (has_promo.sum() < MIN_BUCKET_N): #如果“无促销”或者“有促销”任意一方的样本太少，就不要做 uplift 判断。
        baseline_sales = float(g["Weekly_Sales"].mean()) # 数据不足 -> 不认为促销有提升 -> uplift 明确置 0
        baseline_promo = float(g["promo_intensity"].mean())
        high_sales = baseline_sales
        high_promo = baseline_promo
        uplift_sales = 0.0
        uplift_promo = 0.0
    else:
        # -- 4) 只在正促销周里选“强促销周”（避免大量 0 导致分位数为 0） --
        q_high = g.loc[has_promo, "promo_intensity"].quantile(HIGH_Q) # 强促销的分数
        high_mask = has_promo & (g["promo_intensity"] >= q_high) # 有促销 且强促销的周

        # -- high 桶太小 保守为不做uplift 判断 --
        if high_mask.sum() < MIN_BUCKET_N:
            baseline_sales = float(g["Weekly_Sales"].mean())
            baseline_promo = float(g["promo_intensity"].mean())
            high_sales = baseline_sales
            high_promo = baseline_promo
            uplift_sales = 0.0
            uplift_promo = 0.0
        else: # -- 这里对强促销周进行求sale和promo 同时对无促销计算baseline --
            baseline_sales = float(g.loc[no_promo, "Weekly_Sales"].mean())
            baseline_promo = float(g.loc[no_promo, "promo_intensity"].mean())  # 基本为 0，但保留字段
            high_sales = float(g.loc[high_mask, "Weekly_Sales"].mean())
            high_promo = float(g.loc[high_mask, "promo_intensity"].mean())

            uplift_sales = high_sales - baseline_sales # promo=0 不参与任何均值稀释
            uplift_promo = high_promo - baseline_promo

    # 5) 提供一个 elasticity 仅作为参考
    promo_elasticity = 0.0 if uplift_promo <= 0 else (uplift_sales / (uplift_promo + EPS))

    # 6) 输出两个 scenario
    rows.append({
        "Dept": dept,
        "scenario": "promo_up",
        "baseline_sales": baseline_sales,
        "avg_promo": baseline_promo,         # baseline promo
        "promo_elasticity": promo_elasticity,
        "pct_change": 'baseline -> strong',                  # 占位字段（此版本不做线性外推）
        "delta_promo": uplift_promo,          # baseline->high 的 promo 变化量
        "delta_sales": uplift_sales,          # baseline->high 的 sales 变化量
        "simulated_sales": max(baseline_sales + uplift_sales, 0.0),
    })
    rows.append({
        "Dept": dept,
        "scenario": "promo_down",
        "baseline_sales": baseline_sales,
        "avg_promo": baseline_promo,
        "promo_elasticity": promo_elasticity,
        "pct_change":  'weak -> baseline',                 # 占位字段
        "delta_promo": -uplift_promo,
        "delta_sales": -uplift_sales,
        "simulated_sales": max(baseline_sales - uplift_sales, 0.0),
    })

sim = pd.DataFrame(rows)

# -- 保存 --
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
sim.to_csv(args.output_path, index=False)
print("[OK] saved promo simulation to:", args.output_path)
print(sim.head())


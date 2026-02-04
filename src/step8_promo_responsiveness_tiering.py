"""
    step8 把 step7 得到的“无促销→强促销”的销量提升（delta_sales）变成一个“可靠度加权后的评分 score”，然后按 score 把 Dept 分成 High/Mid/Low（不够数据的标 Low_confidence），并给出一句策略建议。
    为什么要有 step8 因为只看delta sale 不现实 我们需要看到更多维度的影响 是不是小周 是不是小店 是不是临时被节假日冲上去的  因此我们要做一个score进行评价 并给出真正的策略。
    但目前 由于太规整了 所以看不出价值 但是以后更细的颗粒度的时候就看的出来了 例如时间窗口 聚合颗粒度 或者更强的惩罚或更不稳定的dept
"""
import pandas as pd
import argparse
import os

# -- 命令行参数 -- 
parser = argparse.ArgumentParser()
parser.add_argument("--sim_path", type=str, default="outputs/datasets/promo_simulation_ab.csv")
parser.add_argument("--tier_path", type=str, default="outputs/datasets/dept_assortment_tiers.csv")
parser.add_argument("--metrics_path", type=str, default="outputs/datasets/dept_assortment_metrics.csv")
parser.add_argument("--output_path", type=str, default="outputs/datasets/dept_promo_responsiveness.csv")

# -- 可用性门槛 --
parser.add_argument("--min_weeks", type=int, default=40)
parser.add_argument("--min_stores", type=int, default=10)
parser.add_argument("--min_avg_promo", type=float, default=0.0)  # 注意：baseline avg_promo 通常为 0

# -- 稳定性参数 --
parser.add_argument("--stability_weeks_cap", type=int, default=80)
parser.add_argument("--use_cv_penalty", action="store_true")

args = parser.parse_args()
EPS = 1e-9

# -- 加载三个表 分别是 粗糙whatif dept_tier 以及 dept_metric --
sim = pd.read_csv(args.sim_path)
tiers = pd.read_csv(args.tier_path)
metrics = pd.read_csv(args.metrics_path)

# -- 只用up 因为我们假设down up一致 --
up = sim[sim["scenario"] == "promo_up"].copy()

if up.empty:
    raise ValueError("No rows found for scenario=='promo_up'. Check step7 output.")

# -- 参考指标 调价可以提高销量的多少百分比 --
# -- 例如：从无促销到强促销，销量大概提升 41%（只是参考，不主导分层） --
up["lift_pct"] = up["delta_sales"] / up["baseline_sales"].replace(0, pd.NA)
up["lift_pct"] = up["lift_pct"].fillna(0.0)

# -- 增量 主要信号 --
up["uplift_strength"] = up["delta_sales"].fillna(0.0)

# -- 融合指标 --
need_cols = ["Dept", "n_weeks", "store_coverage"]
extra_cols = [c for c in ["cv", "sales_cv", "weekly_sales_cv", "cv_sales"] if c in metrics.columns]
use_cols = need_cols + extra_cols
up = up.merge(metrics[use_cols], on="Dept", how="left")

# -- 融合tier --
up = up.merge(tiers[["Dept", "tier"]], on="Dept", how="left")

# -- 可用性筛选 --
has_avg_promo = "avg_promo" in up.columns
avg_promo_cond = True
if has_avg_promo:
    # baseline avg_promo 通常为 0；如果你把 min_avg_promo 设成 >0，会把所有都筛掉
    avg_promo_cond = (up["avg_promo"].fillna(0.0) >= args.min_avg_promo)
# -- 新增一行 是否可用 --
up["eligible"] = (
    (up["n_weeks"].fillna(0) >= args.min_weeks) &
    (up["store_coverage"].fillna(0) >= args.min_stores) &
    (up["baseline_sales"].fillna(0) > 0) &
    avg_promo_cond
)

# -- 只保留AB商品 前面也做过 但这是防御性的 --
up = up[up["tier"].isin(["A_core", "B_holiday"])].copy()

# -- 稳定性权重 --
# -- 周越多 稳定性越大 但初衷是超过一定的周 我们就觉得它周的稳定性贡献可以了稳定性可以了 --
cap = max(int(args.stability_weeks_cap), 1)
up["stability_weight"] = (up["n_weeks"].fillna(0) / cap).clip(lower=0.0, upper=1.0)

# -- 使用变异系数进行惩罚 尝试抵消不分节日影响 --
if args.use_cv_penalty and len(extra_cols) > 0:
    cv_col = extra_cols[0]
    cv_val = up[cv_col].fillna(0.0).clip(lower=0.0)
    up["stability_weight"] = up["stability_weight"] * (1.0 / (1.0 + cv_val)) # 波动越大（cv 越高），越可能是节假日冲击/异常波动驱动

# -- 主要的得分 --
up["score"] = up["uplift_strength"] * up["stability_weight"]

# -- 只在可用的dept进行阈值 避免低置信度的进行计算 污染阈值 --
ref = up[up["eligible"]].copy()
if len(ref) < 20: # 太少就不分eligible了
    ref = up.copy()

q_low = ref["score"].quantile(0.33) # 两个阈值
q_high = ref["score"].quantile(0.67)

# -- 分层函数 --
def resp_level(row):
    if not bool(row["eligible"]):
        return "Low_confidence"
    x = float(row["score"])
    if x >= q_high:
        return "High_response"
    elif x <= q_low:
        return "Low_response"
    else:
        return "Mid_response"
# -- 进行分成 并且另开新列承载信息 --
up["promo_response_level"] = up.apply(resp_level, axis=1)

# -- 与上面的函数对应 结合业务 进行策略输出 -- 
def suggestion(row):
    lvl = row["promo_response_level"]
    tier = row["tier"]
    if lvl == "Low_confidence":
        return "数据不足/覆盖不足：仅做观察，不自动给促销强度建议"
    if tier == "A_core" and lvl == "Low_response":
        return "核心品类但促销不敏感：稳价保供，减少无效促销"
    if tier == "A_core" and lvl == "High_response":
        return "核心品类且促销敏感：可小幅促销换量，注意毛利"
    if tier == "B_holiday" and lvl == "High_response":
        return "节假日驱动且促销敏感：节前重点投放与备货"
    if tier == "B_holiday" and lvl == "Low_response":
        return "节假日驱动但促销不敏感：更依赖节点陈列/品类组合"
    return "常规策略：结合门店与库存约束微调"

up["strategy_note"] = up.apply(suggestion, axis=1)

# -- 显式指定等级顺序（避免字符串字母序导致 High 排在 Low 前面） --
level_rank = {
    "High_response": 0,
    "Mid_response": 1,
    "Low_response": 2,
    "Low_confidence": 3,
}
up["level_rank"] = up["promo_response_level"].map(level_rank).fillna(99).astype(int)

out_cols = [
    "Dept", "tier",
    "baseline_sales", "delta_sales", "lift_pct",
    "uplift_strength", "stability_weight", "score",
    "n_weeks", "store_coverage", "eligible",
    "promo_response_level", "strategy_note"
]
out_cols = [c for c in out_cols if c in up.columns]

out = up[out_cols + ["level_rank"]].sort_values(
    ["level_rank", "score"],
    ascending=[True, False]
).drop(columns=["level_rank"])

print("[INFO] promo response distribution:")
print(out["promo_response_level"].value_counts())
print(out.head(10))

os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
out.to_csv(args.output_path, index=False)
print("[OK] saved promo responsiveness to:", args.output_path)


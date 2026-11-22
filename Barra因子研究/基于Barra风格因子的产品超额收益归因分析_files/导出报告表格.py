# -*- coding: utf-8 -*-
"""导出表1（超额-因子相关）与表3（超额收益归因分解）"""
import os
import numpy as np
import pandas as pd
from scipy import stats

WORKDIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKDIR)

因子列 = [
    "beta", "book_to_price", "dividend", "earning_variability",
    "earning_yield", "growth", "investment_quality", "leverage",
    "liquidity", "long_term_reversal", "mid_cap", "momentum",
    "profitability", "size", "volatility",
]

因子中文名 = {
    "beta": "贝塔",
    "book_to_price": "账面市值比",
    "dividend": "股息",
    "earning_variability": "盈利波动",
    "earning_yield": "盈利收益率",
    "growth": "成长",
    "investment_quality": "投资质量",
    "leverage": "杠杆",
    "liquidity": "流动性",
    "long_term_reversal": "长期反转",
    "mid_cap": "中市值",
    "momentum": "动量",
    "profitability": "盈利能力",
    "size": "规模",
    "volatility": "波动率",
}

# ---------- 产品超额 ----------
产品净值 = pd.read_excel("产品净值.xlsx", sheet_name=0)
产品净值["日期"] = pd.to_datetime(产品净值["日期"])
产品净值 = 产品净值.sort_values("日期")

指数数据 = pd.read_excel("K线导出_000852_日线数据.xlsx", sheet_name=0)
指数数据["日期"] = pd.to_datetime(指数数据["交易时间"])
指数数据 = 指数数据[["日期", "收盘价"]].sort_values("日期")

aligned = 产品净值.merge(指数数据, on="日期", how="left")
aligned = aligned.dropna(subset=["收盘价"])
aligned["复权净值因子"] = aligned["复权净值"] / aligned["复权净值"].shift(1)
aligned["收盘价因子"] = aligned["收盘价"] / aligned["收盘价"].shift(1)
aligned["每日超额因子"] = aligned["复权净值因子"] - aligned["收盘价因子"]

# ---------- Barra 因子 ----------
barra = pd.read_excel("Barra因子.xlsx", sheet_name=0)
barra["日期"] = pd.to_datetime(barra["日期"].astype(str), format="%Y%m%d")
for c in 因子列:
    barra[c] = pd.to_numeric(barra[c], errors="coerce")

分析数据 = aligned[["日期", "每日超额因子"]].merge(barra, on="日期", how="inner")
分析数据 = 分析数据.sort_values("日期")

for f in 因子列:
    v = 分析数据[f].astype(float)
    分析数据[f] = v / v.shift(1) - 1

分析数据 = 分析数据.dropna(subset=["每日超额因子"])

# ---------- 表1：相关系数 ----------
rows = []
y = 分析数据["每日超额因子"].values
for f in 因子列:
    x = 分析数据[f].values
    r, p = stats.pearsonr(x, y)
    sig = "**" if p < 0.01 else "*" if p < 0.05 else "." if p < 0.10 else ""
    rows.append({
        "因子中文名": 因子中文名[f],
        "因子": f,
        "相关系数": round(r, 4),
        "p值": round(p, 4),
        "显著性": sig,
    })

表1 = pd.DataFrame(rows)
表1["绝对值"] = 表1["相关系数"].abs()
表1 = 表1.sort_values("绝对值", ascending=False).drop(columns=["绝对值"])

# ---------- 表3：归因分解 ----------
因子相关矩阵 = 分析数据[因子列].corr()
候选 = 表1.head(5)["因子"].tolist()
筛选因子 = []
for f in 候选:
    if not any(abs(因子相关矩阵.loc[f, g]) > 0.7 for g in 筛选因子):
        筛选因子.append(f)

X = 分析数据[筛选因子].copy()
X.insert(0, "const", 1.0)
y = 分析数据["每日超额因子"].values
beta, _, _, _ = np.linalg.lstsq(X.values, y, rcond=None)
拟合 = X.values @ beta
残差 = y - 拟合
alpha = beta[0]

表3 = pd.DataFrame({
    "指标": [
        "周均超额收益",
        "截距 α（周均无法被因子解释的部分）",
        "风格拟合均值",
        "残差均值",
    ],
    "数值": [
        round(y.mean(), 4),
        round(alpha, 4),
        round((拟合 - alpha).mean(), 4),
        round(残差.mean(), 4),
    ],
})

# ---------- 导出 ----------
表1_path_csv = "表1_超额收益与Barra因子相关系数.csv"
表3_path_csv = "表3_超额收益归因分解.csv"
表1.to_csv(表1_path_csv, index=False, encoding="utf-8-sig")
表3.to_csv(表3_path_csv, index=False, encoding="utf-8-sig")

excel_path = "报告核心表格.xlsx"
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    表1.to_excel(writer, sheet_name="表1_因子相关系数", index=False)
    表3.to_excel(writer, sheet_name="表3_归因分解", index=False)

print("已导出：")
print(f"  - {表1_path_csv}")
print(f"  - {表3_path_csv}")
print(f"  - {excel_path}")

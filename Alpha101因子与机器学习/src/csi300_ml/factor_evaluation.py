"""单因子Rank IC、ICIR、HAC显著性和五分组收益评价。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def 计算每日RankIC(
    数据: pd.DataFrame,
    因子名称: tuple[str, ...],
    收益列: str = "未来5日相对收益",
    最少股票数: int = 20,
) -> pd.DataFrame:
    """逐日计算各因子与未来收益的Spearman相关系数。"""
    记录: list[dict[str, object]] = []
    for 日期, 当日数据 in 数据.groupby("日期", sort=True):
        当日记录: dict[str, object] = {"日期": 日期}
        收益排名 = 当日数据[收益列].rank(method="average", pct=True)
        for 因子 in 因子名称:
            有效 = 当日数据[因子].notna() & 当日数据[收益列].notna()
            if int(有效.sum()) < 最少股票数:
                当日记录[因子] = np.nan
                continue
            因子排名 = 当日数据.loc[有效, 因子].rank(
                method="average",
                pct=True,
            )
            当日记录[因子] = 因子排名.corr(收益排名.loc[有效])
        记录.append(当日记录)
    return pd.DataFrame(记录)


def 计算HAC均值t值(序列: pd.Series, 最大滞后: int = 5) -> float:
    """使用Bartlett权重计算均值的Newey-West/HAC t值。"""
    数值 =序列.dropna().to_numpy(dtype=float)
    样本数 = len(数值)
    if 样本数 < 3:
        return np.nan
    中心值 = 数值 - 数值.mean()
    长期方差 = float(np.dot(中心值, 中心值) / 样本数)
    实际滞后 = min(最大滞后, 样本数 - 1)
    for 滞后 in range(1, 实际滞后 + 1):
        权重 = 1 - 滞后 / (实际滞后 + 1)
        自协方差 = float(
            np.dot(中心值[滞后:], 中心值[:-滞后]) / 样本数
        )
        长期方差 += 2 * 权重 * 自协方差
    if 长期方差 <= 0:
        return np.nan
    return float(数值.mean() / np.sqrt(长期方差 / 样本数))


def 汇总RankIC(
    每日IC: pd.DataFrame,
    日期数据集映射: pd.Series,
    因子名称: tuple[str, ...],
) -> pd.DataFrame:
    """按训练、验证、测试及全样本汇总IC指标。"""
    数据 = 每日IC.copy()
    数据["数据集"] = 数据["日期"].map(日期数据集映射)
    结果: list[dict[str, object]] = []
    分组列表 = [
        ("训练集", 数据.loc[数据["数据集"] == "训练集"]),
        ("验证集", 数据.loc[数据["数据集"] == "验证集"]),
        ("测试集", 数据.loc[数据["数据集"] == "测试集"]),
        ("全样本", 数据),
    ]
    for 数据集名称, 子集 in 分组列表:
        for 因子 in 因子名称:
            序列 = 子集[因子].dropna()
            标准差 = 序列.std(ddof=1)
            IC均值 = 序列.mean()
            结果.append(
                {
                    "数据集": 数据集名称,
                    "因子": 因子,
                    "IC均值": IC均值,
                    "IC标准差": 标准差,
                    "ICIR": IC均值 / 标准差 if 标准差 > 0 else np.nan,
                    "年化ICIR": (
                        IC均值 / 标准差 * np.sqrt(252)
                        if 标准差 > 0
                        else np.nan
                    ),
                    "正IC比例": (序列 > 0).mean(),
                    "HAC_t值": 计算HAC均值t值(序列, 最大滞后=5),
                    "有效交易日": len(序列),
                }
            )
    return pd.DataFrame(结果)


def 计算五分组收益(
    数据: pd.DataFrame,
    因子名称: tuple[str, ...],
    收益列: str = "未来5日收益",
    最少股票数: int = 20,
) -> pd.DataFrame:
    """逐日按因子分为五组，并计算各组平均未来收益。"""
    记录: list[dict[str, object]] = []
    for 日期, 当日数据 in 数据.groupby("日期", sort=True):
        数据集名称 = 当日数据["数据集"].dropna().iloc[0]
        for 因子 in 因子名称:
            有效数据 = 当日数据[[因子, 收益列]].dropna()
            if len(有效数据) < 最少股票数:
                continue
            排名 = 有效数据[因子].rank(method="first")
            分组 = pd.qcut(排名, 5, labels=[1, 2, 3, 4, 5])
            分组收益 = 有效数据.groupby(分组, observed=True)[收益列].mean()
            for 组别, 收益 in 分组收益.items():
                记录.append(
                    {
                        "日期": 日期,
                        "数据集": 数据集名称,
                        "因子": 因子,
                        "组别": int(组别),
                        "未来5日平均收益": float(收益),
                    }
                )
    return pd.DataFrame(记录)


def 汇总五分组收益(每日分组收益: pd.DataFrame) -> pd.DataFrame:
    """汇总各因子各组收益，并计算Q5减Q1。"""
    分组均值 = (
        每日分组收益.groupby(["数据集", "因子", "组别"])[
            "未来5日平均收益"
        ]
        .mean()
        .unstack("组别")
        .rename(columns=lambda 组别: f"Q{组别}")
        .reset_index()
    )
    分组均值["Q5-Q1"] = 分组均值["Q5"] - 分组均值["Q1"]
    return 分组均值

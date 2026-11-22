"""因子预处理、量价特征、未来收益标签和时间切分。"""

from __future__ import annotations

import numpy as np
import pandas as pd


普通量价特征名称 = (
    "收益率1日",
    "收益率5日",
    "收益率20日",
    "波动率5日",
    "波动率20日",
    "日内振幅",
    "成交量相对20日均量",
    "成交金额对数",
    "成交金额相对20日均值",
    "开盘至收盘收益",
)


def _单日缩尾标准化(数据: pd.Series) -> pd.Series:
    """按1%和99%分位缩尾后进行横截面Z标准化。"""
    有效数据 = 数据.dropna()
    if len(有效数据) < 3:
        return pd.Series(np.nan, index=数据.index)
    下界 = 有效数据.quantile(0.01)
    上界 = 有效数据.quantile(0.99)
    缩尾值 = 数据.clip(lower=下界, upper=上界)
    标准差 = 缩尾值.std(ddof=0)
    if pd.isna(标准差) or 标准差 == 0:
        return pd.Series(np.nan, index=数据.index)
    return (缩尾值 - 缩尾值.mean()) / 标准差


def 按日处理因子(
    因子数据: pd.DataFrame,
    因子名称: tuple[str, ...],
) -> pd.DataFrame:
    """对每个交易日的每个因子分别缩尾和标准化。"""
    结果 = 因子数据.copy()
    for 因子 in 因子名称:
        结果[因子] = 结果.groupby("日期", group_keys=False)[因子].transform(
            _单日缩尾标准化
        )
    return 结果


def 构造普通量价特征(面板: pd.DataFrame) -> pd.DataFrame:
    """在每只股票内部构造10个仅使用当日及历史数据的特征。"""
    数据 = 面板.sort_values(["证券代码", "日期"]).copy()
    分组 = 数据.groupby("证券代码", sort=False)

    前一日收盘价 = 分组["收盘价"].shift(1)
    数据["收益率1日"] = 数据["收盘价"] / 前一日收盘价 - 1
    数据["收益率5日"] = 数据["收盘价"] / 分组["收盘价"].shift(5) - 1
    数据["收益率20日"] = 数据["收盘价"] / 分组["收盘价"].shift(20) - 1
    数据["波动率5日"] = (
        数据.groupby("证券代码", sort=False)["收益率1日"]
        .rolling(5, min_periods=5)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    数据["波动率20日"] = (
        数据.groupby("证券代码", sort=False)["收益率1日"]
        .rolling(20, min_periods=20)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
    数据["日内振幅"] = (
        (数据["最高价"] - 数据["最低价"])
        / 数据["开盘价"].replace(0, np.nan)
    )

    二十日均量 = (
        分组["成交量"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    二十日平均成交金额 = (
        分组["成交金额"]
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    数据["成交量相对20日均量"] = (
        数据["成交量"] / 二十日均量.replace(0, np.nan)
    )
    数据["成交金额对数"] = np.log1p(数据["成交金额"].clip(lower=0))
    数据["成交金额相对20日均值"] = (
        数据["成交金额"] / 二十日平均成交金额.replace(0, np.nan)
    )
    数据["开盘至收盘收益"] = (
        数据["收盘价"] / 数据["开盘价"].replace(0, np.nan) - 1
    )
    return 数据


def 添加未来收益标签(
    面板: pd.DataFrame,
    预测周期: int = 5,
) -> pd.DataFrame:
    """构造t+1开盘建仓、持有指定完整交易日的收益和相对收益。"""
    if 预测周期 <= 0:
        raise ValueError("预测周期必须为正整数")

    数据 = 面板.sort_values(["证券代码", "日期"]).copy()
    分组 = 数据.groupby("证券代码", sort=False)
    数据["建仓日期"] = 分组["日期"].shift(-1)
    数据["退出日期"] = 分组["日期"].shift(-(预测周期 + 1))
    建仓开盘价 = 分组["开盘价"].shift(-1)
    退出开盘价 = 分组["开盘价"].shift(-(预测周期 + 1))
    建仓可交易 = 分组["是否可交易"].shift(-1).fillna(False).astype(bool)
    退出可交易 = (
        分组["是否可交易"]
        .shift(-(预测周期 + 1))
        .fillna(False)
        .astype(bool)
    )
    建仓时仍为成分 = (
        分组["是否沪深300成分"].shift(-1).fillna(False).astype(bool)
    )

    原始未来收益 = 退出开盘价 / 建仓开盘价 - 1
    标签有效 = 建仓可交易 & 退出可交易 & 建仓时仍为成分
    数据[f"未来{预测周期}日收益"] = 原始未来收益.where(标签有效)
    横截面中位数 = 数据.groupby("日期")[
        f"未来{预测周期}日收益"
    ].transform("median")
    数据[f"未来{预测周期}日相对收益"] = (
        数据[f"未来{预测周期}日收益"] - 横截面中位数
    )
    数据[f"未来{预测周期}日分类标签"] = np.where(
        数据[f"未来{预测周期}日相对收益"].isna(),
        np.nan,
        (数据[f"未来{预测周期}日相对收益"] > 0).astype(float),
    )
    return 数据


def 添加时间数据集标记(数据: pd.DataFrame) -> pd.DataFrame:
    """按冻结日期和完整标签区间划分训练、验证与测试集。"""
    结果 = 数据.copy()
    日期 = pd.to_datetime(结果["日期"])
    建仓日期 = pd.to_datetime(结果["建仓日期"])
    退出日期 = pd.to_datetime(结果["退出日期"])

    训练条件 = (
        日期.between("2019-01-02", "2022-12-30")
        & 建仓日期.ge("2019-01-03")
        & 退出日期.le("2022-12-30")
    )
    验证条件 = (
        日期.between("2023-01-03", "2023-12-29")
        & 建仓日期.ge("2023-01-04")
        & 退出日期.le("2023-12-29")
    )
    测试条件 = (
        日期.between("2024-01-02", "2025-12-31")
        & 建仓日期.ge("2024-01-03")
        & 退出日期.le("2025-12-31")
    )

    结果["数据集"] = pd.Series(pd.NA, index=结果.index, dtype="string")
    结果.loc[训练条件, "数据集"] = "训练集"
    结果.loc[验证条件, "数据集"] = "验证集"
    结果.loc[测试条件, "数据集"] = "测试集"
    return 结果

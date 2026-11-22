"""Alpha101 因子使用的时间序列和横截面基础运算符。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def 延迟(数据: pd.DataFrame, 期数: int) -> pd.DataFrame:
    """返回每只股票若干交易日前的值。"""
    return 数据.shift(期数)


def 差分(数据: pd.DataFrame, 期数: int) -> pd.DataFrame:
    """返回当前值与若干交易日前数值之差。"""
    return 数据 - 延迟(数据, 期数)


def 时间序列求和(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """过去指定窗口求和，窗口不足时返回缺失。"""
    return 数据.rolling(窗口, min_periods=窗口).sum()


def 时间序列均值(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """过去指定窗口求平均，窗口不足时返回缺失。"""
    return 数据.rolling(窗口, min_periods=窗口).mean()


def 时间序列最小值(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """过去指定窗口的最小值。"""
    return 数据.rolling(窗口, min_periods=窗口).min()


def 时间序列最大值(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """过去指定窗口的最大值。"""
    return 数据.rolling(窗口, min_periods=窗口).max()


def 时间序列标准差(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """过去指定窗口的样本标准差，使用自由度1。"""
    return 数据.rolling(窗口, min_periods=窗口).std(ddof=1)


def _末值百分位排名(窗口值: np.ndarray) -> float:
    """计算窗口最后一个有效值在窗口内的平均百分位排名。"""
    if np.isnan(窗口值).any():
        return np.nan
    末值 = 窗口值[-1]
    小于数量 = np.count_nonzero(窗口值 < 末值)
    等于数量 = np.count_nonzero(窗口值 == 末值)
    return float((小于数量 + (等于数量 + 1) / 2) / len(窗口值))


def 时间序列排名(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """最后一个值在过去窗口内的百分位排名，并列值取平均名次。"""
    return 数据.rolling(窗口, min_periods=窗口).apply(
        _末值百分位排名,
        raw=True,
    )


def 相关系数(
    数据甲: pd.DataFrame,
    数据乙: pd.DataFrame,
    窗口: int,
) -> pd.DataFrame:
    """逐股票计算两组数据过去窗口的时间序列相关系数。"""
    数据甲, 数据乙 = 数据甲.align(数据乙, join="outer")
    return 数据甲.rolling(窗口, min_periods=窗口).corr(数据乙)


def 协方差(
    数据甲: pd.DataFrame,
    数据乙: pd.DataFrame,
    窗口: int,
) -> pd.DataFrame:
    """逐股票计算两组数据过去窗口的样本协方差。"""
    数据甲, 数据乙 = 数据甲.align(数据乙, join="outer")
    return 数据甲.rolling(窗口, min_periods=窗口).cov(数据乙, ddof=1)


def _最大值位置(窗口值: np.ndarray) -> float:
    """返回最大值在窗口中从旧到新的1基位置。"""
    if np.isnan(窗口值).any():
        return np.nan
    return float(np.argmax(窗口值) + 1)


def _最小值位置(窗口值: np.ndarray) -> float:
    """返回最小值在窗口中从旧到新的1基位置。"""
    if np.isnan(窗口值).any():
        return np.nan
    return float(np.argmin(窗口值) + 1)


def 时间序列最大值位置(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """返回最大值在过去窗口内的位置，最旧为1、最新为窗口长度。"""
    return 数据.rolling(窗口, min_periods=窗口).apply(_最大值位置, raw=True)


def 时间序列最小值位置(数据: pd.DataFrame, 窗口: int) -> pd.DataFrame:
    """返回最小值在过去窗口内的位置，最旧为1、最新为窗口长度。"""
    return 数据.rolling(窗口, min_periods=窗口).apply(_最小值位置, raw=True)


def 横截面排名(数据: pd.DataFrame) -> pd.DataFrame:
    """在同一交易日的股票之间计算百分位排名。"""
    return 数据.rank(axis=1, method="average", pct=True, na_option="keep")


def 横截面缩放(数据: pd.DataFrame, 目标绝对值和: float = 1.0) -> pd.DataFrame:
    """使每个交易日所有股票数值绝对值之和等于目标值。"""
    绝对值和 = 数据.abs().sum(axis=1, min_count=1).replace(0, np.nan)
    return 数据.div(绝对值和, axis=0) * 目标绝对值和


def 符号(数据: pd.DataFrame) -> pd.DataFrame:
    """逐元素返回-1、0或1，并保留缺失值。"""
    return np.sign(数据)


def 带符号幂(数据: pd.DataFrame, 指数: float) -> pd.DataFrame:
    """计算 sign(x)×|x|^a，避免负数分数次幂产生复数。"""
    return 符号(数据) * 数据.abs().pow(指数)


def 日收益率(收盘价: pd.DataFrame) -> pd.DataFrame:
    """使用前复权收盘价计算逐股票日收益率。"""
    return 收盘价 / 延迟(收盘价, 1) - 1

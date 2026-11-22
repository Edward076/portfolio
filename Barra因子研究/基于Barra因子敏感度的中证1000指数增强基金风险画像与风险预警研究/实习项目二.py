"""
项目:
基于Barra因子敏感度的中证1000指数增强基金风险画像与风险预警研究

使用说明:
1) 按模块逐步完成各个 TODO.
2) 统一数据频率为周频(W-FRI).
3) 核心因子筛选放到滚动回归阶段再做.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# ============================================================================
# 0) 全局配置
# ============================================================================


@dataclass
class 配置:
    项目目录: Path
    barra文件: str = "Barra因子202672.xlsx"
    指数文件: str = "K线导出_000852_周线数据7_2.xlsx"
    基金文件列表: Tuple[str, ...] = (
        "产品净值-1.xlsx",
        "产品净值-2.xlsx",
        "产品净值-3.xlsx",
        "产品净值-4.xlsx",
        "产品净值-5.xlsx",
        "产品净值-6（1号项目）.xlsx",
    )
    周频规则: str = "W-FRI"
    滚动窗口: int = 12
    前瞻窗口: int = 4


# ============================================================================
# 1) 数据读取
# ============================================================================


def 读取_barra数据(参数: 配置) -> pd.DataFrame:
    """读取原始 Barra 因子水平序列."""
    文件路径 = 参数.项目目录 / 参数.barra文件
    数据 = pd.read_excel(文件路径)
    # TODO: 解析日期, 按时间升序, 做基础校验
    return 数据


def 读取_指数数据(参数: 配置) -> pd.DataFrame:
    """读取中证1000周线数据."""
    文件路径 = 参数.项目目录 / 参数.指数文件
    数据 = pd.read_excel(文件路径)
    # TODO: 移除尾部无效行, 解析日期, 保留收盘价
    return 数据


def 读取_基金数据(参数: 配置) -> Dict[str, pd.DataFrame]:
    """读取全部基金净值文件, 返回字典: 基金ID -> 数据表."""
    输出: Dict[str, pd.DataFrame] = {}
    for 序号, 文件名 in enumerate(参数.基金文件列表, start=1):
        文件路径 = 参数.项目目录 / 文件名
        输出[f"基金_{序号}"] = pd.read_excel(文件路径)
    # TODO: 补充文件名与真实基金名称/代码的映射表
    return 输出


# ============================================================================
# 2) 数据清洗与周频对齐
# ============================================================================


def 净值转周收益(
    数据: pd.DataFrame, 日期列: str, 净值列: str, 周频规则: str
) -> pd.Series:
    """按统一周频规则, 将净值序列转换为周收益率."""
    临时表 = 数据.copy()
    临时表[日期列] = pd.to_datetime(临时表[日期列], errors="coerce")
    临时表 = 临时表.dropna(subset=[日期列, 净值列]).sort_values(日期列)
    周净值 = 临时表.set_index(日期列)[净值列].resample(周频规则).last().dropna()
    周收益 = 周净值.pct_change().dropna()
    return 周收益


def 准备_barra周收益(barra原始: pd.DataFrame, 周频规则: str) -> pd.DataFrame:
    """将 Barra 因子水平序列转换为周频因子收益."""
    数据 = barra原始.copy()
    if "日期" not in 数据.columns:
        raise ValueError("Barra数据缺少'日期'列, 无法转换为周收益.")

    数据["日期"] = pd.to_datetime(数据["日期"].astype(str), format="%Y%m%d", errors="coerce")
    数据 = 数据.dropna(subset=["日期"]).sort_values("日期")

    排除列 = {"日期", "中证500", "沪深300", "上证50"}
    候选因子列 = [列 for 列 in 数据.columns if 列 not in 排除列]
    if not 候选因子列:
        raise ValueError("未识别到可用Barra风格因子列.")

    数据[候选因子列] = 数据[候选因子列].apply(pd.to_numeric, errors="coerce")
    数据 = 数据.dropna(subset=候选因子列, how="all")

    周度水平 = 数据.set_index("日期")[候选因子列].resample(周频规则).last().dropna(how="all")
    周度收益 = 周度水平.pct_change().dropna(how="all")
    周度收益.index.name = "日期"
    return 周度收益


def 准备_指数周收益(指数原始: pd.DataFrame) -> pd.DataFrame:
    """生成中证1000周收益率."""
    数据 = 指数原始.copy()
    必要列 = {"交易时间", "收盘价"}
    if not 必要列.issubset(set(数据.columns)):
        raise ValueError("指数数据缺少'交易时间'或'收盘价'列.")

    数据["交易时间"] = pd.to_datetime(数据["交易时间"], errors="coerce")
    数据["收盘价"] = pd.to_numeric(数据["收盘价"], errors="coerce")
    数据 = 数据.dropna(subset=["交易时间", "收盘价"]).sort_values("交易时间")

    周收盘 = (
        数据.set_index("交易时间")["收盘价"]
        .resample("W-FRI")
        .last()
        .dropna()
    )
    指数周收益 = 周收盘.pct_change().dropna().to_frame(name="指数收益率")
    指数周收益.index.name = "日期"
    return 指数周收益


def 构建_基础面板(
    基金收益字典: Dict[str, pd.Series],
    指数周收益: pd.DataFrame,
    barra周收益: pd.DataFrame,
) -> pd.DataFrame:
    """
    构建非平衡面板:
    预期字段: 日期, 基金, 基金收益率, 指数收益率, 超额收益, 各因子...
    """
    合并结果列表: List[pd.DataFrame] = []

    for 基金ID, 基金周收益 in 基金收益字典.items():
        单基金 = 基金周收益.to_frame(name="基金收益率")
        单基金.index.name = "日期"
        单基金 = 单基金.join(指数周收益, how="inner")
        单基金 = 单基金.join(barra周收益, how="inner")
        单基金["超额收益"] = 单基金["基金收益率"] - 单基金["指数收益率"]
        单基金["基金"] = 基金ID
        合并结果列表.append(单基金.reset_index())

    if not 合并结果列表:
        return pd.DataFrame()

    面板 = pd.concat(合并结果列表, ignore_index=True)
    因子列 = [列 for 列 in barra周收益.columns if 列 in 面板.columns]
    列顺序 = ["日期", "基金", "基金收益率", "指数收益率", "超额收益"] + 因子列
    面板 = 面板[列顺序].sort_values(["基金", "日期"]).reset_index(drop=True)
    return 面板


# ============================================================================
# 3) 基础风险指标
# ============================================================================


def 添加_基础风险特征(面板: pd.DataFrame, 滚动窗口: int) -> pd.DataFrame:
    """
    增加特征:
    - 跟踪误差_12周
    - 信息比率_12周
    - 最大回撤_12周
    - 胜率_12周
    - 残差波动_占位(占位, 回归后再补)
    """
    数据 = 面板.copy()
    必要列 = {"日期", "基金", "基金收益率", "超额收益"}
    if not 必要列.issubset(set(数据.columns)):
        raise ValueError(f"面板缺少必要列: {必要列 - set(数据.columns)}")

    数据["日期"] = pd.to_datetime(数据["日期"], errors="coerce")
    数据 = 数据.dropna(subset=["日期"]).sort_values(["基金", "日期"]).reset_index(drop=True)

    def 计算窗口最大回撤(窗口收益: pd.Series) -> float:
        if 窗口收益.isna().any():
            return np.nan
        累计净值 = (1 + 窗口收益).cumprod()
        回撤序列 = 累计净值 / 累计净值.cummax() - 1
        return float(回撤序列.min())

    分组 = 数据.groupby("基金", group_keys=False)

    数据["跟踪误差_12周"] = 分组["超额收益"].transform(
        lambda x: x.rolling(window=滚动窗口, min_periods=滚动窗口).std()
    )
    数据["信息比率_12周"] = 分组["超额收益"].transform(
        lambda x: x.rolling(window=滚动窗口, min_periods=滚动窗口).mean()
        / x.rolling(window=滚动窗口, min_periods=滚动窗口).std()
    )
    数据["最大回撤_12周"] = 分组["基金收益率"].transform(
        lambda x: x.rolling(window=滚动窗口, min_periods=滚动窗口).apply(计算窗口最大回撤, raw=False)
    )
    数据["胜率_12周"] = 分组["超额收益"].transform(
        lambda x: (x > 0).rolling(window=滚动窗口, min_periods=滚动窗口).mean()
    )
    数据["残差波动_占位"] = np.nan
    return 数据


# ============================================================================
# 4) 滚动 Barra 回归(核心)
# ============================================================================


def 稍后筛选_核心因子(barra周收益: pd.DataFrame) -> List[str]:
    """
    设计为后置占位:
    因子筛选在做到本模块时再执行.
    可选示例:
    ['size', 'momentum', 'liquidity', 'volatility', 'book_to_price']
    """
    默认候选 = ["size", "momentum", "liquidity", "volatility", "book_to_price"]
    可用因子 = [列 for 列 in 默认候选 if 列 in barra周收益.columns]
    if 可用因子:
        return 可用因子
    return list(barra周收益.columns[:5])


def 执行_滚动因子回归(
    面板: pd.DataFrame,
    因子列: List[str],
    滚动窗口: int,
) -> pd.DataFrame:
    """
    滚动模型(按基金, 按日期):
    超额收益 = alpha + b1*f1 + b2*f2 + ... + eps
    输出:
    alpha, betas, r2, 残差波动
    """
    数据 = 面板.copy()
    必要列 = {"日期", "基金", "超额收益"}
    缺失列 = 必要列 - set(数据.columns)
    if 缺失列:
        raise ValueError(f"执行回归缺少必要列: {缺失列}")
    因子缺失列 = [列 for 列 in 因子列 if 列 not in 数据.columns]
    if 因子缺失列:
        raise ValueError(f"执行回归缺少因子列: {因子缺失列}")

    数据["日期"] = pd.to_datetime(数据["日期"], errors="coerce")
    数据 = 数据.dropna(subset=["日期"]).sort_values(["基金", "日期"]).reset_index(drop=True)

    数据["超额收益"] = pd.to_numeric(数据["超额收益"], errors="coerce")
    for 列 in 因子列:
        数据[列] = pd.to_numeric(数据[列], errors="coerce")

    因子中文名映射 = {
        "size": "规模",
        "momentum": "动量",
        "liquidity": "流动性",
        "volatility": "波动率",
        "book_to_price": "账面市值比",
    }

    数据["超额收益截距_12周"] = np.nan
    数据["拟合优度_12周"] = np.nan
    数据["残差波动_12周"] = np.nan
    因子暴露列映射: Dict[str, str] = {}
    for 因子 in 因子列:
        中文因子名 = 因子中文名映射.get(因子, 因子)
        列名 = f"因子暴露_{中文因子名}_12周"
        因子暴露列映射[因子] = 列名
        数据[列名] = np.nan

    for 基金ID, 子表 in 数据.groupby("基金", group_keys=False):
        子表索引 = 子表.index.to_list()
        if len(子表索引) < 滚动窗口:
            continue

        for 结束位置 in range(滚动窗口 - 1, len(子表索引)):
            窗口索引 = 子表索引[结束位置 - 滚动窗口 + 1 : 结束位置 + 1]
            窗口数据 = 数据.loc[窗口索引, ["超额收益"] + 因子列].dropna()
            if len(窗口数据) < 滚动窗口:
                continue
            if len(窗口数据) <= len(因子列) + 1:
                continue

            y = 窗口数据["超额收益"].to_numpy(dtype=float)
            X = 窗口数据[因子列].to_numpy(dtype=float)
            X带截距 = np.column_stack([np.ones(len(X)), X])
            系数, *_ = np.linalg.lstsq(X带截距, y, rcond=None)

            预测值 = X带截距 @ 系数
            残差 = y - 预测值
            残差平方和 = float(np.sum(残差**2))
            总离差平方和 = float(np.sum((y - y.mean()) ** 2))
            r2 = np.nan if 总离差平方和 == 0 else 1 - 残差平方和 / 总离差平方和
            残差波动 = np.nan if len(残差) <= 1 else float(np.std(残差, ddof=1))

            目标索引 = 子表索引[结束位置]
            数据.loc[目标索引, "超额收益截距_12周"] = float(系数[0])
            数据.loc[目标索引, "拟合优度_12周"] = r2
            数据.loc[目标索引, "残差波动_12周"] = 残差波动
            for i, 因子 in enumerate(因子列, start=1):
                数据.loc[目标索引, 因子暴露列映射[因子]] = float(系数[i])

    if "残差波动_占位" in 数据.columns:
        数据["残差波动_占位"] = 数据["残差波动_12周"]
    return 数据


# ============================================================================
# 5) 风格漂移
# ============================================================================


def 添加_风格漂移得分(
    数据: pd.DataFrame, beta列: List[str], 长窗口: int = 52
) -> pd.DataFrame:
    """
    风格漂移得分:
    sum(abs(beta_now - rolling_mean_beta_long))
    """
    输出 = 数据.copy()
    必要列 = {"日期", "基金"}
    缺失列 = 必要列 - set(输出.columns)
    if 缺失列:
        raise ValueError(f"风格漂移计算缺少必要列: {缺失列}")
    缺失beta列 = [列 for 列 in beta列 if 列 not in 输出.columns]
    if 缺失beta列:
        raise ValueError(f"风格漂移计算缺少因子暴露列: {缺失beta列}")

    输出["日期"] = pd.to_datetime(输出["日期"], errors="coerce")
    输出 = 输出.dropna(subset=["日期"]).sort_values(["基金", "日期"]).reset_index(drop=True)

    for 列 in beta列:
        输出[列] = pd.to_numeric(输出[列], errors="coerce")
        漂移列 = f"漂移幅度_{列}"
        输出[漂移列] = 输出.groupby("基金")[列].transform(
            lambda x: (x - x.rolling(window=长窗口, min_periods=长窗口).mean()).abs()
        )

    漂移列列表 = [f"漂移幅度_{列}" for 列 in beta列]
    输出[f"风格漂移得分_{长窗口}周"] = 输出[漂移列列表].sum(axis=1, min_count=1)
    输出[f"风格漂移得分_{长窗口}周_4周均值"] = 输出.groupby("基金")[f"风格漂移得分_{长窗口}周"].transform(
        lambda x: x.rolling(window=4, min_periods=4).mean()
    )
    return 输出


# ============================================================================
# 6) 风险画像
# ============================================================================


def 构建_基金风险画像(数据: pd.DataFrame) -> pd.DataFrame:
    """
    构建基金级风险画像表:
    - Barra 敏感度等级
    - 风格稳定性
    - 跟踪误差水平
    - 画像标签
    """
    输入 = 数据.copy()
    必要列 = {"日期", "基金", "超额收益", "跟踪误差_12周", "残差波动_12周"}
    缺失列 = 必要列 - set(输入.columns)
    if 缺失列:
        raise ValueError(f"风险画像构建缺少必要列: {缺失列}")

    输入["日期"] = pd.to_datetime(输入["日期"], errors="coerce")
    输入 = 输入.dropna(subset=["日期"]).sort_values(["基金", "日期"]).reset_index(drop=True)

    因子暴露列 = [列 for 列 in 输入.columns if 列.startswith("因子暴露_") and 列.endswith("_12周")]
    if not 因子暴露列:
        raise ValueError("风险画像构建未找到因子暴露列.")

    def 最后一个有效值(序列: pd.Series) -> float:
        非空 = 序列.dropna()
        if 非空.empty:
            return np.nan
        return float(非空.iloc[-1])

    画像明细 = []
    for 基金ID, 子表 in 输入.groupby("基金"):
        最新日期 = 子表["日期"].max()
        最新跟踪误差 = 最后一个有效值(子表["跟踪误差_12周"])
        最新风格漂移 = (
            最后一个有效值(子表["风格漂移得分_52周_4周均值"])
            if "风格漂移得分_52周_4周均值" in 子表.columns
            else np.nan
        )
        if np.isnan(最新风格漂移) and "风格漂移得分_52周" in 子表.columns:
            最新风格漂移 = 最后一个有效值(子表["风格漂移得分_52周"])

        平均绝对因子暴露 = float(子表[因子暴露列].abs().mean(axis=1, skipna=True).mean(skipna=True))

        画像明细.append(
            {
                "基金": 基金ID,
                "最新日期": 最新日期,
                "平均超额收益": float(子表["超额收益"].mean(skipna=True)),
                "最新跟踪误差_12周": 最新跟踪误差,
                "最新信息比率_12周": 最后一个有效值(子表["信息比率_12周"]) if "信息比率_12周" in 子表.columns else np.nan,
                "最新最大回撤_12周": 最后一个有效值(子表["最大回撤_12周"]) if "最大回撤_12周" in 子表.columns else np.nan,
                "最新胜率_12周": 最后一个有效值(子表["胜率_12周"]) if "胜率_12周" in 子表.columns else np.nan,
                "最新残差波动_12周": 最后一个有效值(子表["残差波动_12周"]),
                "最新风格漂移得分": 最新风格漂移,
                "平均绝对因子暴露": 平均绝对因子暴露,
            }
        )

    画像表 = pd.DataFrame(画像明细)

    def 分位分级(序列: pd.Series, 高标签: str, 中标签: str, 低标签: str) -> pd.Series:
        q33 = 序列.quantile(1 / 3)
        q67 = 序列.quantile(2 / 3)
        return pd.Series(
            np.where(序列 >= q67, 高标签, np.where(序列 <= q33, 低标签, 中标签)),
            index=序列.index,
        )

    画像表["Barra敏感度等级"] = 分位分级(
        画像表["平均绝对因子暴露"], 高标签="高", 中标签="中", 低标签="低"
    )
    画像表["风格稳定性等级"] = 分位分级(
        画像表["最新风格漂移得分"], 高标签="不稳定", 中标签="中等", 低标签="稳定"
    )
    画像表["主动风险等级"] = 分位分级(
        画像表["最新跟踪误差_12周"], 高标签="高", 中标签="中", 低标签="低"
    )

    def 生成画像标签(行: pd.Series) -> str:
        if 行["风格稳定性等级"] == "不稳定" and 行["主动风险等级"] == "高":
            return "风格漂移高风险型"
        if 行["Barra敏感度等级"] == "高" and 行["风格稳定性等级"] == "稳定":
            return "稳定高Barra暴露型"
        if 行["主动风险等级"] == "低" and 行["风格稳定性等级"] == "稳定":
            return "稳健低主动风险型"
        return "均衡过渡型"

    画像表["风险画像标签"] = 画像表.apply(生成画像标签, axis=1)
    画像表 = 画像表.sort_values("基金").reset_index(drop=True)
    return 画像表


# ============================================================================
# 7) 未来风险标签
# ============================================================================


def 添加_未来风险标签(数据: pd.DataFrame, 前瞻窗口: int) -> pd.DataFrame:
    """
    建议标签:
    - future_high_te
    - future_drawdown_risk
    - future_high_residual_vol(未来高残差波动)
    """
    输出 = 数据.copy()
    必要列 = {"日期", "基金", "超额收益", "基金收益率", "残差波动_12周"}
    缺失列 = 必要列 - set(输出.columns)
    if 缺失列:
        raise ValueError(f"未来风险标签构建缺少必要列: {缺失列}")

    输出["日期"] = pd.to_datetime(输出["日期"], errors="coerce")
    输出 = 输出.dropna(subset=["日期"]).sort_values(["基金", "日期"]).reset_index(drop=True)

    输出["超额收益"] = pd.to_numeric(输出["超额收益"], errors="coerce")
    输出["基金收益率"] = pd.to_numeric(输出["基金收益率"], errors="coerce")
    输出["残差波动_12周"] = pd.to_numeric(输出["残差波动_12周"], errors="coerce")

    def 计算窗口最大回撤(窗口收益: pd.Series) -> float:
        if 窗口收益.isna().any():
            return np.nan
        累计净值 = (1 + 窗口收益).cumprod()
        回撤序列 = 累计净值 / 累计净值.cummax() - 1
        return float(回撤序列.min())

    分组 = 输出.groupby("基金", group_keys=False)

    # 未来4周跟踪误差: 基于未来4周超额收益的标准差
    输出["未来跟踪误差_4周"] = 分组["超额收益"].transform(
        lambda x: x.shift(-1)[::-1].rolling(window=前瞻窗口, min_periods=前瞻窗口).std()[::-1]
    )
    输出["未来高跟踪误差标签_4周"] = np.nan

    # 未来4周最大回撤: 基于未来4周基金收益率
    输出["未来最大回撤_4周"] = 分组["基金收益率"].transform(
        lambda x: x.shift(-1)[::-1]
        .rolling(window=前瞻窗口, min_periods=前瞻窗口)
        .apply(计算窗口最大回撤, raw=False)[::-1]
    )
    输出["未来回撤风险标签_4周"] = np.nan

    # 未来4周残差波动: 未来4周残差波动_12周的均值
    输出["未来残差波动均值_4周"] = 分组["残差波动_12周"].transform(
        lambda x: x.shift(-1)[::-1].rolling(window=前瞻窗口, min_periods=前瞻窗口).mean()[::-1]
    )
    输出["未来高残差波动标签_4周"] = np.nan

    # 使用“每只基金自身历史”75%分位数作为阈值
    def 打标签(子表: pd.DataFrame) -> pd.DataFrame:
        结果 = 子表.copy()
        te阈值 = 结果["未来跟踪误差_4周"].quantile(0.75)
        残差阈值 = 结果["未来残差波动均值_4周"].quantile(0.75)
        回撤阈值 = -0.03

        结果["未来高跟踪误差标签_4周"] = np.where(
            结果["未来跟踪误差_4周"] > te阈值, 1, 0
        )
        结果["未来回撤风险标签_4周"] = np.where(
            结果["未来最大回撤_4周"] < 回撤阈值, 1, 0
        )
        结果["未来高残差波动标签_4周"] = np.where(
            结果["未来残差波动均值_4周"] > 残差阈值, 1, 0
        )

        # 对未来窗口不足导致的NaN位置, 统一置空而非0/1
        无效te = 结果["未来跟踪误差_4周"].isna()
        无效回撤 = 结果["未来最大回撤_4周"].isna()
        无效残差 = 结果["未来残差波动均值_4周"].isna()
        结果.loc[无效te, "未来高跟踪误差标签_4周"] = np.nan
        结果.loc[无效回撤, "未来回撤风险标签_4周"] = np.nan
        结果.loc[无效残差, "未来高残差波动标签_4周"] = np.nan
        return 结果

    输出 = 分组.apply(打标签)
    if "基金" not in 输出.columns:
        输出 = 输出.reset_index()
        if "level_0" in 输出.columns and "基金" not in 输出.columns:
            输出 = 输出.rename(columns={"level_0": "基金"})
        if "level_1" in 输出.columns:
            输出 = 输出.drop(columns=["level_1"])
    输出 = 输出.reset_index(drop=True)
    return 输出


# ============================================================================
# 8) 轻量机器学习预警(Logistic / LASSO)
# ============================================================================


def 准备_机器学习数据集(数据: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """选择特征与目标变量, 并移除泄露字段."""
    输入 = 数据.copy()
    目标列 = "未来高跟踪误差标签_4周"
    if 目标列 not in 输入.columns:
        raise ValueError(f"机器学习数据集中缺少目标列: {目标列}")

    候选特征列 = [
        "基金",
        "日期",
        "因子暴露_规模_12周",
        "因子暴露_动量_12周",
        "因子暴露_流动性_12周",
        "因子暴露_波动率_12周",
        "因子暴露_账面市值比_12周",
        "超额收益截距_12周",
        "拟合优度_12周",
        "残差波动_12周",
        "跟踪误差_12周",
        "信息比率_12周",
        "最大回撤_12周",
        "胜率_12周",
        "风格漂移得分_52周_4周均值",
    ]
    可用特征列 = [列 for 列 in 候选特征列 if 列 in 输入.columns]
    if len(可用特征列) < 5:
        raise ValueError("可用特征列过少, 请先确认前序步骤输出是否完整.")

    训练表 = 输入[可用特征列 + [目标列]].copy()
    if "日期" in 训练表.columns:
        训练表["日期"] = pd.to_datetime(训练表["日期"], errors="coerce")

    for 列 in 训练表.columns:
        if 列 not in {"基金", "日期"}:
            训练表[列] = pd.to_numeric(训练表[列], errors="coerce")

    训练表 = 训练表.dropna(subset=可用特征列 + [目标列]).sort_values("日期").reset_index(drop=True)
    标签 = 训练表[目标列].astype(int)
    特征 = 训练表[可用特征列]
    return 特征, 标签


def 训练_logistic与lasso(特征: pd.DataFrame, 标签: pd.Series) -> Dict[str, object]:
    """
    - 按时间切分训练/测试集
    - Logistic Regression
    - LASSO Logistic Regression(变量选择)
    - LightGBM / XGBoost(非线性增强)
    """
    数据X = 特征.copy()
    数据y = 标签.copy().astype(int)

    if "日期" not in 数据X.columns:
        raise ValueError("训练数据缺少'日期'列, 无法进行时间切分.")

    数据X["日期"] = pd.to_datetime(数据X["日期"], errors="coerce")
    有效掩码 = 数据X["日期"].notna() & 数据y.notna()
    数据X = 数据X.loc[有效掩码].reset_index(drop=True)
    数据y = 数据y.loc[有效掩码].reset_index(drop=True)

    数据X = 数据X.sort_values("日期").reset_index(drop=True)
    数据y = 数据y.loc[数据X.index].reset_index(drop=True)

    日期序列 = 数据X["日期"].sort_values().unique()
    if len(日期序列) < 10:
        raise ValueError("有效日期点太少, 暂不适合做时间切分建模.")
    切分位置 = int(len(日期序列) * 0.7)
    切分日期 = 日期序列[切分位置]

    训练掩码 = 数据X["日期"] <= 切分日期
    测试掩码 = 数据X["日期"] > 切分日期
    if 训练掩码.sum() < 30 or 测试掩码.sum() < 20:
        raise ValueError("训练/测试样本过少, 请检查数据长度或切分方式.")

    建模X = 数据X.drop(columns=["日期"]).copy()
    if "基金" in 建模X.columns:
        建模X = pd.get_dummies(建模X, columns=["基金"], prefix="基金", drop_first=False)

    for 列 in 建模X.columns:
        建模X[列] = pd.to_numeric(建模X[列], errors="coerce")
    建模X = 建模X.fillna(0.0)

    训练X = 建模X.loc[训练掩码].to_numpy(dtype=float)
    测试X = 建模X.loc[测试掩码].to_numpy(dtype=float)
    训练y = 数据y.loc[训练掩码].to_numpy(dtype=int)
    测试y = 数据y.loc[测试掩码].to_numpy(dtype=int)

    if len(np.unique(训练y)) < 2:
        raise ValueError("训练集仅有单一类别, 无法进行分类建模.")

    # 在训练集内部再按时间切出验证集, 用于阈值优化(避免直接在测试集调阈值).
    训练日期序列 = 数据X.loc[训练掩码, "日期"].sort_values().unique()
    验证切分位置 = max(1, int(len(训练日期序列) * 0.8))
    验证切分日期 = 训练日期序列[验证切分位置 - 1]
    子训练掩码 = 数据X["日期"] <= 验证切分日期
    验证掩码 = (数据X["日期"] > 验证切分日期) & 训练掩码
    if 验证掩码.sum() < 10:
        子训练掩码 = 训练掩码.copy()
        验证掩码 = 训练掩码.copy()

    子训练X = 建模X.loc[子训练掩码].to_numpy(dtype=float)
    子训练y = 数据y.loc[子训练掩码].to_numpy(dtype=int)
    验证X = 建模X.loc[验证掩码].to_numpy(dtype=float)
    验证y = 数据y.loc[验证掩码].to_numpy(dtype=int)
    训练X_df = 建模X.loc[训练掩码].copy()
    测试X_df = 建模X.loc[测试掩码].copy()
    子训练X_df = 建模X.loc[子训练掩码].copy()
    验证X_df = 建模X.loc[验证掩码].copy()

    标准化器_子训练 = StandardScaler()
    子训练X标准化 = 标准化器_子训练.fit_transform(子训练X)
    验证X标准化 = 标准化器_子训练.transform(验证X)

    def 选择最优阈值(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return 0.5
        候选阈值 = np.linspace(0.2, 0.8, 61)
        最优阈值 = 0.5
        最优f1 = -1.0
        最优recall = -1.0
        for 阈值 in 候选阈值:
            pred = (y_prob >= 阈值).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred, zero_division=0)
            if (f1 > 最优f1) or (np.isclose(f1, 最优f1) and rec > 最优recall):
                最优f1 = f1
                最优recall = rec
                最优阈值 = float(阈值)
        return 最优阈值

    # Logistic: 先在子训练+验证上选阈值, 再用全训练重训并评估测试集.
    logistic模型_子训练 = LogisticRegression(max_iter=2000, class_weight="balanced")
    logistic模型_子训练.fit(子训练X标准化, 子训练y)
    logistic验证概率 = logistic模型_子训练.predict_proba(验证X标准化)[:, 1]
    logistic最优阈值 = 选择最优阈值(验证y, logistic验证概率)

    标准化器 = StandardScaler()
    训练X标准化 = 标准化器.fit_transform(训练X)
    测试X标准化 = 标准化器.transform(测试X)
    logistic模型 = LogisticRegression(max_iter=2000, class_weight="balanced")
    logistic模型.fit(训练X标准化, 训练y)
    logistic概率 = logistic模型.predict_proba(测试X标准化)[:, 1]
    logistic预测 = (logistic概率 >= logistic最优阈值).astype(int)

    # LASSO Logistic: 同样流程.
    lasso模型_子训练 = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        C=0.5,
        max_iter=4000,
        class_weight="balanced",
    )
    lasso模型_子训练.fit(子训练X标准化, 子训练y)
    lasso验证概率 = lasso模型_子训练.predict_proba(验证X标准化)[:, 1]
    lasso最优阈值 = 选择最优阈值(验证y, lasso验证概率)

    lasso模型 = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        C=0.5,
        max_iter=4000,
        class_weight="balanced",
    )
    lasso模型.fit(训练X标准化, 训练y)
    lasso概率 = lasso模型.predict_proba(测试X标准化)[:, 1]
    lasso预测 = (lasso概率 >= lasso最优阈值).astype(int)

    非零系数掩码 = np.abs(lasso模型.coef_[0]) > 1e-8
    lasso入选特征 = list(np.array(建模X.columns)[非零系数掩码])

    # Tree模型: 对非线性关系更友好, 并通过scale_pos_weight缓解样本不平衡.
    子训练正样本 = int(np.sum(子训练y == 1))
    子训练负样本 = int(np.sum(子训练y == 0))
    if 子训练正样本 == 0:
        类别权重比例 = 1.0
    else:
        类别权重比例 = float(子训练负样本 / max(1, 子训练正样本))

    try:
        from lightgbm import LGBMClassifier

        lightgbm模型_子训练 = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            scale_pos_weight=类别权重比例,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        lightgbm模型_子训练.fit(子训练X_df, 子训练y)
        lightgbm验证概率 = lightgbm模型_子训练.predict_proba(验证X_df)[:, 1]
        lightgbm最优阈值 = 选择最优阈值(验证y, lightgbm验证概率)

        lightgbm模型 = LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            scale_pos_weight=类别权重比例,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        lightgbm模型.fit(训练X_df, 训练y)
        lightgbm概率 = lightgbm模型.predict_proba(测试X_df)[:, 1]
        lightgbm预测 = (lightgbm概率 >= lightgbm最优阈值).astype(int)
    except Exception as 异常:
        print(f"  - LightGBM训练失败, 已跳过: {异常}")
        lightgbm最优阈值 = np.nan
        lightgbm概率 = np.array([])
        lightgbm预测 = np.array([])

    try:
        from xgboost import XGBClassifier

        xgboost模型_子训练 = XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=类别权重比例,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgboost模型_子训练.fit(子训练X_df, 子训练y)
        xgboost验证概率 = xgboost模型_子训练.predict_proba(验证X_df)[:, 1]
        xgboost最优阈值 = 选择最优阈值(验证y, xgboost验证概率)

        xgboost模型 = XGBClassifier(
            objective="binary:logistic",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=类别权重比例,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgboost模型.fit(训练X_df, 训练y)
        xgboost概率 = xgboost模型.predict_proba(测试X_df)[:, 1]
        xgboost预测 = (xgboost概率 >= xgboost最优阈值).astype(int)
    except Exception as 异常:
        print(f"  - XGBoost训练失败, 已跳过: {异常}")
        xgboost最优阈值 = np.nan
        xgboost概率 = np.array([])
        xgboost预测 = np.array([])

    return {
        "切分日期": pd.to_datetime(切分日期),
        "阈值验证切分日期": pd.to_datetime(验证切分日期),
        "训练样本数": int(len(训练y)),
        "测试样本数": int(len(测试y)),
        "测试y": 测试y,
        "logistic概率": logistic概率,
        "logistic预测": logistic预测,
        "logistic最优阈值": logistic最优阈值,
        "lasso概率": lasso概率,
        "lasso预测": lasso预测,
        "lasso最优阈值": lasso最优阈值,
        "lightgbm概率": lightgbm概率,
        "lightgbm预测": lightgbm预测,
        "lightgbm最优阈值": lightgbm最优阈值,
        "xgboost概率": xgboost概率,
        "xgboost预测": xgboost预测,
        "xgboost最优阈值": xgboost最优阈值,
        "建模特征列": list(建模X.columns),
        "lasso入选特征": lasso入选特征,
    }


def 评估_分类结果(结果: Dict[str, object]) -> pd.DataFrame:
    """输出 AUC / Recall / 混淆矩阵等评价摘要."""
    测试y = np.array(结果["测试y"])

    def 安全auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_prob))

    def 安全pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_prob))

    def 单模型评估(模型名: str, 概率: np.ndarray, 预测: np.ndarray, 阈值: float) -> Dict[str, object]:
        tn, fp, fn, tp = confusion_matrix(测试y, 预测, labels=[0, 1]).ravel()
        return {
            "模型": 模型名,
            "分类阈值": float(阈值),
            "AUC": 安全auc(测试y, 概率),
            "PR_AUC": 安全pr_auc(测试y, 概率),
            "Recall": float(recall_score(测试y, 预测, zero_division=0)),
            "Precision": float(precision_score(测试y, 预测, zero_division=0)),
            "F1": float(f1_score(测试y, 预测, zero_division=0)),
            "Accuracy": float(accuracy_score(测试y, 预测)),
            "混淆矩阵_TN": int(tn),
            "混淆矩阵_FP": int(fp),
            "混淆矩阵_FN": int(fn),
            "混淆矩阵_TP": int(tp),
        }

    评估结果 = [
        单模型评估(
            "Logistic",
            np.array(结果["logistic概率"]),
            np.array(结果["logistic预测"]),
            float(结果.get("logistic最优阈值", 0.5)),
        ),
        单模型评估(
            "LASSO_Logistic",
            np.array(结果["lasso概率"]),
            np.array(结果["lasso预测"]),
            float(结果.get("lasso最优阈值", 0.5)),
        ),
    ]

    if len(np.array(结果.get("lightgbm概率", []))) == len(测试y):
        评估结果.append(
            单模型评估(
                "LightGBM",
                np.array(结果["lightgbm概率"]),
                np.array(结果["lightgbm预测"]),
                float(结果.get("lightgbm最优阈值", 0.5)),
            )
        )
    if len(np.array(结果.get("xgboost概率", []))) == len(测试y):
        评估结果.append(
            单模型评估(
                "XGBoost",
                np.array(结果["xgboost概率"]),
                np.array(结果["xgboost预测"]),
                float(结果.get("xgboost最优阈值", 0.5)),
            )
        )
    return pd.DataFrame(评估结果)


# ============================================================================
# 9) 可视化与结果导出
# ============================================================================


def 绘制_关键图表(数据: pd.DataFrame, 画像: pd.DataFrame, 输出目录: Path | None = None) -> None:
    """
    计划图表:
    1) Barra beta 动态变化图
    2) 风格漂移得分曲线
    3) 跟踪误差 / 回撤变化图
    4) 风险预警概率时间线
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  - 未安装matplotlib, 跳过第9步图表绘制.")
        return None

    输出路径 = 输出目录 if 输出目录 is not None else Path.cwd()
    输出路径.mkdir(parents=True, exist_ok=True)

    # 图表全局样式优化: 中文显示、网格、字号、分辨率.
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10

    图表数据 = 数据.copy()
    if "日期" in 图表数据.columns:
        图表数据["日期"] = pd.to_datetime(图表数据["日期"], errors="coerce")
    if "基金" not in 图表数据.columns and "level_0" in 图表数据.columns:
        图表数据 = 图表数据.rename(columns={"level_0": "基金"})
    图表数据 = 图表数据.dropna(subset=["日期"]).copy()
    if "基金" in 图表数据.columns:
        图表数据 = 图表数据.sort_values(["基金", "日期"])
    else:
        图表数据 = 图表数据.sort_values(["日期"])

    # 图1: 风格漂移得分时间序列
    if {"基金", "日期", "风格漂移得分_52周_4周均值"}.issubset(图表数据.columns):
        plt.figure(figsize=(12, 6))
        for 基金ID, 子表 in 图表数据.groupby("基金"):
            plt.plot(子表["日期"], 子表["风格漂移得分_52周_4周均值"], label=基金ID, linewidth=1.8, alpha=0.9)
        plt.title("风格漂移得分(52周, 4周均值)")
        plt.xlabel("日期")
        plt.ylabel("得分")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=20, ha="right")
        plt.legend(loc="upper left", ncols=2, frameon=True)
        plt.tight_layout()
        plt.savefig(输出路径 / "step9_图1_风格漂移得分.png", bbox_inches="tight")
        plt.close()

    # 图2: 跟踪误差时间序列
    if {"基金", "日期", "跟踪误差_12周"}.issubset(图表数据.columns):
        plt.figure(figsize=(12, 6))
        for 基金ID, 子表 in 图表数据.groupby("基金"):
            plt.plot(子表["日期"], 子表["跟踪误差_12周"], label=基金ID, linewidth=1.8, alpha=0.9)
        plt.title("跟踪误差(12周滚动)")
        plt.xlabel("日期")
        plt.ylabel("跟踪误差")
        plt.grid(True, linestyle="--", alpha=0.25)
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=20, ha="right")
        plt.legend(loc="upper right", ncols=2, frameon=True)
        plt.tight_layout()
        plt.savefig(输出路径 / "step9_图2_跟踪误差.png", bbox_inches="tight")
        plt.close()

    # 图3: 风险画像标签分布
    if {"基金", "风险画像标签"}.issubset(画像.columns):
        标签计数 = 画像["风险画像标签"].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        颜色 = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
        柱图对象 = 标签计数.plot(kind="bar", color=颜色[: len(标签计数)], width=0.6)
        plt.title("风险画像标签分布")
        plt.xlabel("标签")
        plt.ylabel("基金数量")
        plt.grid(axis="y", linestyle="--", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        # 在柱顶标注数量值
        for 柱 in 柱图对象.patches:
            高度 = 柱.get_height()
            柱图对象.annotate(
                f"{int(高度)}",
                (柱.get_x() + 柱.get_width() / 2, 高度),
                ha="center",
                va="bottom",
                xytext=(0, 3),
                textcoords="offset points",
            )
        plt.tight_layout()
        plt.savefig(输出路径 / "step9_图3_风险画像标签分布.png", bbox_inches="tight")
        plt.close()

    print(f"  - 第9步图表已导出目录: {输出路径}")
    return None


def 导出_结果文件(
    数据: pd.DataFrame, 画像: pd.DataFrame, 指标: pd.DataFrame, 参数: 配置
) -> None:
    """导出中间结果与最终结果表."""
    最终结果路径_xlsx = 参数.项目目录 / "项目总览_step9.xlsx"
    try:
        with pd.ExcelWriter(最终结果路径_xlsx, engine="openpyxl") as writer:
            数据导出 = 数据.copy()
            if "日期" in 数据导出.columns:
                数据导出["日期"] = pd.to_datetime(数据导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
            数据导出.to_excel(writer, sheet_name="明细数据_到step7", index=False)

            画像导出 = 画像.copy()
            if "最新日期" in 画像导出.columns:
                画像导出["最新日期"] = pd.to_datetime(画像导出["最新日期"], errors="coerce").dt.strftime("%Y-%m-%d")
            画像导出.to_excel(writer, sheet_name="风险画像_step6", index=False)
            指标.to_excel(writer, sheet_name="模型评估_step8", index=False)
        print(f"  - 第9步总览已导出(xlsx): {最终结果路径_xlsx}")
    except PermissionError:
        print(f"  - 第9步总览xlsx导出失败(文件可能被占用): {最终结果路径_xlsx}")

    数据.to_csv(参数.项目目录 / "项目总览_明细_step9.csv", index=False, encoding="utf-8-sig")
    画像.to_csv(参数.项目目录 / "项目总览_画像_step9.csv", index=False, encoding="utf-8-sig")
    指标.to_csv(参数.项目目录 / "项目总览_模型评估_step9.csv", index=False, encoding="utf-8-sig")
    return None


# ============================================================================
# 10) 主流程
# ============================================================================


def 主流程() -> None:
    """
    项目执行总入口.

    阅读方式:
    - 可以把它当作"任务清单".
    - 每一段注释都说明了"当前在做什么"和"下一步要做什么".
    - 当前版本先完成第1步读取数据, 第2步及后续模块保留TODO.
    """
    print("\n================ 项目主流程开始 ================\n")

    # [0/10] 先配置本次运行要使用的文件路径和文件名.
    # 这一段的作用是"告诉程序去哪里找数据".
    参数 = 配置(
        项目目录=Path(
            r"d:\OneDrive\文档\武宇翔国泰君安期货实习任务\基于Barra因子敏感度的中证1000指数增强基金风险画像与风险预警研究"
        ),
        barra文件="Barra因子202672.xlsx",
        指数文件="K线导出_000852_周线数据7_2.xlsx",
        基金文件列表=(
            "产品净值-1.xlsx",
            "产品净值-2.xlsx",
            "产品净值-3.xlsx",
            "产品净值-4.xlsx",
            "产品净值-5.xlsx",
            "产品净值-6（1号项目）.xlsx",
        ),
    )
    print("[配置完成] 已写入项目目录和8个原始数据文件名.")

    # [1/10] 读取原始数据.
    # 目标: 把"Barra + 指数 + 六只基金"读入内存, 暂不做清洗.
    print("\n[第1步] 开始读取原始数据...")

    # 1.1 读取Barra原始数据.
    # 文件: Barra因子202672.xlsx
    barra原始 = 读取_barra数据(参数)
    print(f"  - Barra数据读取完成, 行列: {barra原始.shape}")

    # 1.2 读取中证1000指数数据.
    # 文件: K线导出_000852_周线数据7_2.xlsx
    指数原始 = 读取_指数数据(参数)
    print(f"  - 指数数据读取完成, 行列: {指数原始.shape}")

    # 1.3 读取六只基金原始净值数据.
    # 文件: 产品净值-1.xlsx ~ 产品净值-6（1号项目）.xlsx
    基金原始 = 读取_基金数据(参数)
    print(f"  - 基金文件读取完成, 共读取: {len(基金原始)} 只基金")
    print(f"  - 基金键名列表: {list(基金原始.keys())}")

    # 快速检查: 打印各数据源前2行, 确认读取内容和字段是否符合预期.
    print("\n  [快速检查] Barra原始数据前2行:")
    print(barra原始.head(2))
    print("\n  [快速检查] 指数原始数据前2行:")
    print(指数原始.head(2))
    print('\n  [快速检查] 基金原始数据(基金_1)前2行:')
    print(基金原始["基金_1"].head(2))

    print("[第1步完成] 原始数据已全部读取.\n")

    # [2/10] 清洗并对齐.
    # 目标: 将各来源数据统一到周频, 然后按日期内连接成非平衡面板.
    print("[第2步] 开始清洗并对齐数据...")

    # 2.1 基金净值 -> 周收益(统一用复权净值).
    基金周收益字典: Dict[str, pd.Series] = {}
    for 基金ID, 基金表 in 基金原始.items():
        周收益 = 净值转周收益(
            数据=基金表,
            日期列="日期",
            净值列="复权净值",
            周频规则=参数.周频规则,
        )
        基金周收益字典[基金ID] = 周收益
        print(f"  - {基金ID} 周收益样本数: {len(周收益)}")

    # 2.2 指数数据 -> 周收益.
    指数周收益 = 准备_指数周收益(指数原始)
    print(f"  - 指数周收益样本数: {len(指数周收益)}")

    # 2.3 Barra因子水平 -> 周频因子收益.
    barra周收益 = 准备_barra周收益(barra原始, 参数.周频规则)
    print(f"  - Barra周收益样本数: {len(barra周收益)}, 因子数: {barra周收益.shape[1]}")

    # 2.4 构建非平衡面板(日期, 基金).
    基础面板 = 构建_基础面板(基金周收益字典, 指数周收益, barra周收益)
    print(f"  - 基础面板行列: {基础面板.shape}")
    if not 基础面板.empty:
        print("  - 基础面板前3行:")
        print(基础面板.head(3))

    # 2.5 导出第2步结果, 便于直接打开检查.
    # 说明: 优先导出xlsx供Excel直接查看, 同时保留csv备份.
    第2步输出路径_xlsx = 参数.项目目录 / "基础面板_step2.xlsx"
    第2步输出路径_csv = 参数.项目目录 / "基础面板_step2.csv"
    基础面板导出 = 基础面板.copy()
    if "日期" in 基础面板导出.columns:
        基础面板导出["日期"] = pd.to_datetime(基础面板导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        基础面板导出.to_excel(第2步输出路径_xlsx, index=False)
        print(f"  - 第2步结果已导出(xlsx): {第2步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第2步xlsx导出失败(文件可能被占用): {第2步输出路径_xlsx}")
    基础面板导出.to_csv(第2步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第2步结果已导出(csv):  {第2步输出路径_csv}")

    print("[第2步完成] 数据清洗与对齐已完成.\n")

    # [3/10] 基础风险指标.
    print("[第3步] 开始构建基础风险指标...")
    第3步结果 = 添加_基础风险特征(基础面板, 参数.滚动窗口)
    print(f"  - 第3步结果行列: {第3步结果.shape}")

    预览列 = [
        "日期",
        "基金",
        "基金收益率",
        "超额收益",
        "跟踪误差_12周",
        "信息比率_12周",
        "最大回撤_12周",
        "胜率_12周",
    ]
    存在预览列 = [列 for 列 in 预览列 if 列 in 第3步结果.columns]
    print("  - 第3步结果前3行(关键列):")
    print(第3步结果[存在预览列].head(3))

    # 导出第3步结果.
    第3步输出路径_xlsx = 参数.项目目录 / "基础风险特征_step3.xlsx"
    第3步输出路径_csv = 参数.项目目录 / "基础风险特征_step3.csv"
    第3步导出 = 第3步结果.copy()
    if "日期" in 第3步导出.columns:
        第3步导出["日期"] = pd.to_datetime(第3步导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        第3步导出.to_excel(第3步输出路径_xlsx, index=False)
        print(f"  - 第3步结果已导出(xlsx): {第3步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第3步xlsx导出失败(文件可能被占用): {第3步输出路径_xlsx}")
    第3步导出.to_csv(第3步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第3步结果已导出(csv):  {第3步输出路径_csv}")
    print("[第3步完成] 基础风险指标已构建.\n")

    # [4/10] 滚动Barra回归(核心).
    print("[第4步] 开始执行滚动Barra回归...")
    核心因子列 = 稍后筛选_核心因子(barra周收益)
    print(f"  - 当前用于回归的核心因子: {核心因子列}")

    第4步结果 = 执行_滚动因子回归(第3步结果, 核心因子列, 参数.滚动窗口)
    print(f"  - 第4步结果行列: {第4步结果.shape}")

    因子中文名映射 = {
        "size": "规模",
        "momentum": "动量",
        "liquidity": "流动性",
        "volatility": "波动率",
        "book_to_price": "账面市值比",
    }
    回归预览列 = ["日期", "基金", "超额收益", "超额收益截距_12周", "拟合优度_12周", "残差波动_12周"]
    回归预览列 += [f"因子暴露_{因子中文名映射.get(因子, 因子)}_12周" for 因子 in 核心因子列]
    存在回归预览列 = [列 for 列 in 回归预览列 if 列 in 第4步结果.columns]
    print("  - 第4步结果前3行(关键列):")
    print(第4步结果[存在回归预览列].head(3))

    第4步输出路径_xlsx = 参数.项目目录 / "滚动回归结果_step4.xlsx"
    第4步输出路径_csv = 参数.项目目录 / "滚动回归结果_step4.csv"
    第4步导出 = 第4步结果.copy()
    if "日期" in 第4步导出.columns:
        第4步导出["日期"] = pd.to_datetime(第4步导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        第4步导出.to_excel(第4步输出路径_xlsx, index=False)
        print(f"  - 第4步结果已导出(xlsx): {第4步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第4步xlsx导出失败(文件可能被占用): {第4步输出路径_xlsx}")
    第4步导出.to_csv(第4步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第4步结果已导出(csv):  {第4步输出路径_csv}")
    print("[第4步完成] 滚动Barra回归已完成.\n")

    # [5/10] 风格漂移得分.
    print("[第5步] 开始构建风格漂移得分...")
    beta列 = [f"因子暴露_{因子中文名映射.get(因子, 因子)}_12周" for 因子 in 核心因子列]
    第5步结果 = 添加_风格漂移得分(第4步结果, beta列=beta列, 长窗口=52)
    print(f"  - 第5步结果行列: {第5步结果.shape}")

    风格漂移预览列 = ["日期", "基金"] + beta列 + ["风格漂移得分_52周", "风格漂移得分_52周_4周均值"]
    存在风格漂移预览列 = [列 for 列 in 风格漂移预览列 if 列 in 第5步结果.columns]
    print("  - 第5步结果前3行(关键列):")
    print(第5步结果[存在风格漂移预览列].head(3))

    第5步输出路径_xlsx = 参数.项目目录 / "风格漂移结果_step5.xlsx"
    第5步输出路径_csv = 参数.项目目录 / "风格漂移结果_step5.csv"
    第5步导出 = 第5步结果.copy()
    if "日期" in 第5步导出.columns:
        第5步导出["日期"] = pd.to_datetime(第5步导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        第5步导出.to_excel(第5步输出路径_xlsx, index=False)
        print(f"  - 第5步结果已导出(xlsx): {第5步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第5步xlsx导出失败(文件可能被占用): {第5步输出路径_xlsx}")
    第5步导出.to_csv(第5步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第5步结果已导出(csv):  {第5步输出路径_csv}")
    print("[第5步完成] 风格漂移得分已构建.\n")

    # [6/10] 基金风险画像.
    print("[第6步] 开始构建基金风险画像...")
    基金风险画像表 = 构建_基金风险画像(第5步结果)
    print(f"  - 基金风险画像表行列: {基金风险画像表.shape}")
    print("  - 基金风险画像预览:")
    print(基金风险画像表.head(6))

    第6步输出路径_xlsx = 参数.项目目录 / "基金风险画像_step6.xlsx"
    第6步输出路径_csv = 参数.项目目录 / "基金风险画像_step6.csv"
    第6步导出 = 基金风险画像表.copy()
    for 日期列名 in ["最新日期"]:
        if 日期列名 in 第6步导出.columns:
            第6步导出[日期列名] = pd.to_datetime(第6步导出[日期列名], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        第6步导出.to_excel(第6步输出路径_xlsx, index=False)
        print(f"  - 第6步结果已导出(xlsx): {第6步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第6步xlsx导出失败(文件可能被占用): {第6步输出路径_xlsx}")
    第6步导出.to_csv(第6步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第6步结果已导出(csv):  {第6步输出路径_csv}")
    print("[第6步完成] 基金风险画像已构建.\n")

    # [7/10] 未来风险标签.
    print("[第7步] 开始构建未来风险标签...")
    第7步结果 = 添加_未来风险标签(第5步结果, 参数.前瞻窗口)
    print(f"  - 第7步结果行列: {第7步结果.shape}")

    标签预览列 = [
        "日期",
        "基金",
        "未来跟踪误差_4周",
        "未来高跟踪误差标签_4周",
        "未来最大回撤_4周",
        "未来回撤风险标签_4周",
        "未来残差波动均值_4周",
        "未来高残差波动标签_4周",
    ]
    存在标签预览列 = [列 for 列 in 标签预览列 if 列 in 第7步结果.columns]
    print("  - 第7步结果前3行(关键列):")
    print(第7步结果[存在标签预览列].head(3))

    第7步输出路径_xlsx = 参数.项目目录 / "未来风险标签_step7.xlsx"
    第7步输出路径_csv = 参数.项目目录 / "未来风险标签_step7.csv"
    第7步导出 = 第7步结果.copy()
    if "日期" in 第7步导出.columns:
        第7步导出["日期"] = pd.to_datetime(第7步导出["日期"], errors="coerce").dt.strftime("%Y-%m-%d")
    try:
        第7步导出.to_excel(第7步输出路径_xlsx, index=False)
        print(f"  - 第7步结果已导出(xlsx): {第7步输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第7步xlsx导出失败(文件可能被占用): {第7步输出路径_xlsx}")
    第7步导出.to_csv(第7步输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第7步结果已导出(csv):  {第7步输出路径_csv}")
    print("[第7步完成] 未来风险标签已构建.\n")

    # [8/10] 机器学习预警模型.
    print("[第8步] 开始机器学习预警建模...")
    模型特征, 模型标签 = 准备_机器学习数据集(第7步结果)
    print(f"  - 建模数据行列: X={模型特征.shape}, y={模型标签.shape}")

    模型结果 = 训练_logistic与lasso(模型特征, 模型标签)
    print(f"  - 时间切分日期: {模型结果['切分日期'].date()}")
    print(f"  - 阈值验证切分日期: {模型结果['阈值验证切分日期'].date()}")
    print(f"  - 训练样本数: {模型结果['训练样本数']}, 测试样本数: {模型结果['测试样本数']}")
    print(f"  - Logistic最优阈值: {模型结果['logistic最优阈值']:.3f}")
    print(f"  - LASSO最优阈值: {模型结果['lasso最优阈值']:.3f}")
    if np.isfinite(模型结果.get("lightgbm最优阈值", np.nan)):
        print(f"  - LightGBM最优阈值: {模型结果['lightgbm最优阈值']:.3f}")
    if np.isfinite(模型结果.get("xgboost最优阈值", np.nan)):
        print(f"  - XGBoost最优阈值: {模型结果['xgboost最优阈值']:.3f}")
    print(f"  - LASSO入选特征数: {len(模型结果['lasso入选特征'])}")

    模型评估表 = 评估_分类结果(模型结果)
    print("  - 模型评估结果:")
    print(模型评估表)

    第8步评估输出路径_xlsx = 参数.项目目录 / "模型评估_step8.xlsx"
    第8步评估输出路径_csv = 参数.项目目录 / "模型评估_step8.csv"
    try:
        模型评估表.to_excel(第8步评估输出路径_xlsx, index=False)
        print(f"  - 第8步评估已导出(xlsx): {第8步评估输出路径_xlsx}")
    except PermissionError:
        print(f"  - 第8步评估xlsx导出失败(文件可能被占用): {第8步评估输出路径_xlsx}")
    模型评估表.to_csv(第8步评估输出路径_csv, index=False, encoding="utf-8-sig")
    print(f"  - 第8步评估已导出(csv):  {第8步评估输出路径_csv}")

    第8步特征输出路径 = 参数.项目目录 / "LASSO入选特征_step8.csv"
    pd.DataFrame({"入选特征": 模型结果["lasso入选特征"]}).to_csv(
        第8步特征输出路径, index=False, encoding="utf-8-sig"
    )
    print(f"  - 第8步LASSO入选特征已导出: {第8步特征输出路径}")
    print("[第8步完成] 机器学习预警建模已完成.\n")

    # [9/10] 可视化与总览导出.
    print("[第9步] 开始可视化与总览导出...")
    绘制_关键图表(第7步结果, 基金风险画像表, 参数.项目目录)
    导出_结果文件(第7步结果, 基金风险画像表, 模型评估表, 参数)

    结论摘要 = pd.DataFrame(
        [
            {
                "项目": "机器学习体现",
                "说明": "第8步已完成Logistic、LASSO Logistic、LightGBM、XGBoost建模, 并输出AUC/PR_AUC/Recall/混淆矩阵.",
            },
            {"项目": "风险画像结果", "说明": f"共形成{len(基金风险画像表)}只基金画像标签."},
            {"项目": "最终阶段", "说明": "第1~9步已全部完成, 可进入报告撰写与图表解读."},
        ]
    )
    第9步结论路径 = 参数.项目目录 / "结论摘要_step9.csv"
    结论摘要.to_csv(第9步结论路径, index=False, encoding="utf-8-sig")
    print(f"  - 第9步结论摘要已导出: {第9步结论路径}")
    print("[第9步完成] 可视化与总览导出已完成.\n")

    print("当前进度: 第1~9步全部完成. 你可以直接开始整理报告正文.")
    print("\n================ 项目主流程结束 ================\n")


if __name__ == "__main__":
    主流程()

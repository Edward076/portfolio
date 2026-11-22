"""Choice 行情宽表转长表、清洗和动态成分标记。"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .data_audit import _规范日期, _规范证券代码


价格列 = ["开盘价", "收盘价", "最高价", "最低价"]
数值列 = 价格列 + ["成交量", "成交金额"]
核心列 = ["日期", "证券代码", "证券名称"] + 数值列


def _创建空批次() -> dict[str, list[Any]]:
    """创建宽表解析所需的列式缓存。"""
    return {列名: [] for 列名 in 核心列}


def _批次转数据表(批次: dict[str, list[Any]]) -> pd.DataFrame:
    """将列式缓存转换为类型统一的 DataFrame。"""
    数据表 = pd.DataFrame(批次)
    数据表["日期"] = pd.to_datetime(数据表["日期"]).dt.normalize()
    数据表["证券代码"] = 数据表["证券代码"].astype("string")
    数据表["证券名称"] = 数据表["证券名称"].astype("string")
    for 列名 in 数值列:
        数据表[列名] = pd.to_numeric(数据表[列名], errors="coerce")
    return 数据表


def 遍历行情长表分块(
    文件路径: Path,
    每块交易日数: int = 60,
) -> Iterator[pd.DataFrame]:
    """流式读取 Choice 宽表，并按若干交易日输出长表分块。"""
    if 每块交易日数 <= 0:
        raise ValueError("每块交易日数必须为正整数")

    工作簿 = load_workbook(文件路径, read_only=True, data_only=True)
    try:
        for 工作表 in 工作簿.worksheets:
            工作表.reset_dimensions()
            数据行 = 工作表.iter_rows(values_only=True)
            表头行: list[tuple[Any, ...]] = []
            for _ in range(4):
                try:
                    表头行.append(next(数据行))
                except StopIteration as 异常:
                    raise ValueError(
                        f"工作表 {工作表.title} 不足4行，无法读取行情"
                    ) from 异常

            代码行, 名称行, 字段行 = 表头行[1], 表头行[2], 表头行[3]
            证券代码: list[str] = []
            证券名称: list[str] = []
            起始列: list[int] = []

            for 列号 in range(1, len(代码行), 6):
                代码 = _规范证券代码(代码行[列号])
                if not 代码:
                    continue
                字段 = tuple(字段行[列号 : 列号 + 6])
                if 字段 != tuple(数值列):
                    raise ValueError(
                        f"{工作表.title} 中 {代码} 字段顺序异常: {字段}"
                    )
                证券代码.append(代码)
                证券名称.append(str(名称行[列号]).strip())
                起始列.append(列号)

            批次 = _创建空批次()
            批次日期数 = 0
            for 行 in 数据行:
                日期 = _规范日期(行[0] if 行 else None)
                if 日期 is None:
                    continue

                股票数量 = len(证券代码)
                批次["日期"].extend([日期] * 股票数量)
                批次["证券代码"].extend(证券代码)
                批次["证券名称"].extend(证券名称)
                for 列号 in 起始列:
                    六项数值 = 行[列号 : 列号 + 6]
                    for 字段序号, 列名 in enumerate(数值列):
                        批次[列名].append(六项数值[字段序号])

                批次日期数 += 1
                if 批次日期数 >= 每块交易日数:
                    yield _批次转数据表(批次)
                    批次 = _创建空批次()
                    批次日期数 = 0

            if 批次日期数:
                yield _批次转数据表(批次)
    finally:
        工作簿.close()


def 读取行情长表(文件路径: Path) -> pd.DataFrame:
    """读取并合并工作簿中的全部行情工作表。"""
    分块列表 = list(遍历行情长表分块(文件路径))
    if not 分块列表:
        raise ValueError(f"{文件路径.name} 中没有可用行情")

    行情 = pd.concat(分块列表, ignore_index=True)
    重复记录 = 行情.duplicated(subset=["日期", "证券代码"], keep=False)
    if 重复记录.any():
        重复示例 = 行情.loc[重复记录, ["日期", "证券代码"]].head()
        raise ValueError(f"存在重复的股票—日期记录:\n{重复示例}")
    return 行情.sort_values(["证券代码", "日期"]).reset_index(drop=True)


def 清洗行情(原始行情: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """清洗价格和成交数据，保留停牌及上市前日期但不进行填充。"""
    行情 = 原始行情.copy()

    价格零值 = 行情[价格列].eq(0).any(axis=1)
    成交量非正 = 行情["成交量"].fillna(0).le(0)
    成交金额非正 = 行情["成交金额"].fillna(0).le(0)

    行情[价格列] = 行情[价格列].replace(0, np.nan)
    价格不完整 = 行情[价格列].isna().any(axis=1)
    价格逻辑异常 = (
        行情["最高价"].lt(行情[["开盘价", "收盘价"]].max(axis=1))
        | 行情["最低价"].gt(行情[["开盘价", "收盘价"]].min(axis=1))
        | 行情["最高价"].lt(行情["最低价"])
    ).fillna(False)

    行情["价格零值"] = 价格零值
    行情["成交量非正"] = 成交量非正
    行情["成交金额非正"] = 成交金额非正
    行情["价格逻辑异常"] = 价格逻辑异常
    行情["是否可交易"] = ~(
        价格不完整 | 成交量非正 | 成交金额非正 | 价格逻辑异常
    )

    质量统计 = {
        "原始行数": len(行情),
        "价格含零行数": int(价格零值.sum()),
        "价格不完整行数": int(价格不完整.sum()),
        "成交量非正行数": int(成交量非正.sum()),
        "成交金额非正行数": int(成交金额非正.sum()),
        "价格逻辑异常行数": int(价格逻辑异常.sum()),
        "可交易行数": int(行情["是否可交易"].sum()),
    }
    return 行情, 质量统计


def 构造每日成分集合(
    交易日期: pd.Series,
    初始代码: set[str],
    调整记录: pd.DataFrame,
) -> dict[pd.Timestamp, frozenset[str]]:
    """根据生效日生成每个市场交易日的点时成分股集合。"""
    日期列表 = sorted(pd.to_datetime(交易日期).dt.normalize().unique())
    按日调整 = {
        pd.Timestamp(日期): 当日记录
        for 日期, 当日记录 in 调整记录.groupby("调整日期", sort=True)
    }
    当前成分 = set(初始代码)
    每日成分: dict[pd.Timestamp, frozenset[str]] = {}

    for 日期值 in 日期列表:
        日期 = pd.Timestamp(日期值)
        if 日期 in 按日调整:
            当日记录 = 按日调整[日期]
            剔除代码 = set(
                当日记录.loc[当日记录["调整类型"] == "剔除", "证券代码"]
            )
            纳入代码 = set(
                当日记录.loc[当日记录["调整类型"] == "纳入", "证券代码"]
            )
            当前成分.difference_update(剔除代码)
            当前成分.update(纳入代码)
        每日成分[日期] = frozenset(当前成分)

    return 每日成分


def 添加动态成分标记(
    行情: pd.DataFrame,
    初始代码: set[str],
    调整记录: pd.DataFrame,
) -> pd.DataFrame:
    """为每条股票行情添加当日沪深300成员标记。"""
    结果 = 行情.copy()
    每日成分 = 构造每日成分集合(结果["日期"], 初始代码, 调整记录)
    结果["是否沪深300成分"] = [
        代码 in 每日成分[pd.Timestamp(日期)]
        for 日期, 代码 in zip(结果["日期"], 结果["证券代码"], strict=True)
    ]
    return 结果

"""Choice 数据读取与沪深300历史股票池审计工具。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import load_workbook


行情字段 = ("开盘价", "收盘价", "最高价", "最低价", "成交量", "成交金额")
有效调整类型 = {"纳入", "剔除"}


@dataclass(frozen=True)
class 行情工作表信息:
    """一张 Choice 宽表行情工作表的元数据。"""

    工作表名称: str
    股票数量: int
    交易日数量: int
    开始日期: str
    结束日期: str
    证券代码: tuple[str, ...]


def _规范证券代码(值: Any) -> str:
    """将证券代码规范为 Choice 的 000001.SZ/600000.SH 格式。"""
    if 值 is None:
        return ""
    return str(值).strip().upper()


def _规范日期(值: Any) -> pd.Timestamp | None:
    """解析 Excel 日期；无法解析时返回 None。"""
    if 值 is None:
        return None
    解析结果 = pd.to_datetime(值, errors="coerce")
    if pd.isna(解析结果):
        return None
    return pd.Timestamp(解析结果).normalize()


def 读取成分快照(文件路径: Path) -> pd.DataFrame:
    """读取 Choice 成分详情，过滤空行和数据来源页脚。"""
    数据表 = pd.read_excel(
        文件路径,
        sheet_name="成分详情",
        dtype={"证券代码": "string"},
    )
    必需字段 = {"证券代码", "证券简称"}
    缺失字段 = 必需字段.difference(数据表.columns)
    if 缺失字段:
        raise ValueError(f"{文件路径.name} 缺少字段: {sorted(缺失字段)}")

    数据表 = 数据表.loc[数据表["证券代码"].notna()].copy()
    数据表["证券代码"] = 数据表["证券代码"].map(_规范证券代码)
    数据表 = 数据表.loc[
        数据表["证券代码"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)
    ]
    数据表 = 数据表.drop_duplicates(subset=["证券代码"], keep=False)
    return 数据表.reset_index(drop=True)


def 读取成分调整(文件路径: Path) -> pd.DataFrame:
    """读取成分进出记录，仅保留有效的纳入和剔除事件。"""
    数据表 = pd.read_excel(
        文件路径,
        sheet_name="成分进出记录",
        dtype={"证券代码": "string"},
    )
    必需字段 = {"调整日期", "证券代码", "调整类型"}
    缺失字段 = 必需字段.difference(数据表.columns)
    if 缺失字段:
        raise ValueError(f"{文件路径.name} 缺少字段: {sorted(缺失字段)}")

    数据表 = 数据表.loc[数据表["调整类型"].isin(有效调整类型)].copy()
    数据表["证券代码"] = 数据表["证券代码"].map(_规范证券代码)
    数据表["调整日期"] = pd.to_datetime(
        数据表["调整日期"],
        errors="coerce",
    ).dt.normalize()
    数据表 = 数据表.dropna(subset=["调整日期"])
    数据表 = 数据表.loc[
        数据表["证券代码"].str.match(r"^\d{6}\.(SH|SZ)$", na=False)
    ]

    重复记录 = 数据表.duplicated(subset=["调整日期", "证券代码", "调整类型"])
    if 重复记录.any():
        示例 = 数据表.loc[重复记录, ["调整日期", "证券代码", "调整类型"]]
        raise ValueError(f"成分进出记录存在重复事件:\n{示例.head()}")

    return 数据表.sort_values(
        ["调整日期", "调整类型", "证券代码"]
    ).reset_index(drop=True)


def 重建成分股链条(
    初始代码: set[str],
    调整记录: pd.DataFrame,
) -> tuple[set[str], pd.DataFrame]:
    """按生效日应用成分变化，并检查每次调整链条的一致性。"""
    当前成分 = set(初始代码)
    审计记录: list[dict[str, Any]] = []

    for 生效日期, 当日调整 in 调整记录.groupby("调整日期", sort=True):
        剔除代码 = set(
            当日调整.loc[当日调整["调整类型"] == "剔除", "证券代码"]
        )
        纳入代码 = set(
            当日调整.loc[当日调整["调整类型"] == "纳入", "证券代码"]
        )
        无效剔除 = sorted(剔除代码.difference(当前成分))
        重复纳入 = sorted(纳入代码.intersection(当前成分))

        当前成分.difference_update(剔除代码)
        当前成分.update(纳入代码)
        审计记录.append(
            {
                "调整日期": 生效日期,
                "剔除数量": len(剔除代码),
                "纳入数量": len(纳入代码),
                "调整后成分数量": len(当前成分),
                "无效剔除代码": 无效剔除,
                "重复纳入代码": 重复纳入,
            }
        )

    return 当前成分, pd.DataFrame(审计记录)


def 检查行情工作簿(文件路径: Path) -> list[行情工作表信息]:
    """流式读取 Choice 行情宽表，只提取表头、日期范围和交易日数量。"""
    工作簿 = load_workbook(文件路径, read_only=True, data_only=True)
    检查结果: list[行情工作表信息] = []

    try:
        for 工作表 in 工作簿.worksheets:
            # Choice 导出的 dimension 常错误地写为 A1，必须重置后再流式遍历。
            工作表.reset_dimensions()
            数据行 = 工作表.iter_rows(values_only=True)
            表头行 = []
            for _ in range(4):
                try:
                    表头行.append(next(数据行))
                except StopIteration as 异常:
                    raise ValueError(
                        f"工作表 {工作表.title} 不足4行，无法识别表头"
                    ) from 异常

            代码行 = 表头行[1]
            名称行 = 表头行[2]
            字段行 = 表头行[3]
            代码列表: list[str] = []

            for 列号 in range(1, len(代码行), 6):
                代码 = _规范证券代码(代码行[列号])
                字段 = tuple(字段行[列号 : 列号 + 6])
                if not 代码:
                    continue
                if 字段 != 行情字段:
                    raise ValueError(
                        f"{工作表.title} 中 {代码} 的字段顺序异常: {字段}"
                    )
                if not 名称行[列号]:
                    raise ValueError(f"{工作表.title} 中 {代码} 缺少证券名称")
                代码列表.append(代码)

            有效日期: list[pd.Timestamp] = []
            for 行 in 数据行:
                解析日期 = _规范日期(行[0] if 行 else None)
                if 解析日期 is not None:
                    有效日期.append(解析日期)

            if not 有效日期:
                raise ValueError(f"工作表 {工作表.title} 没有有效交易日期")
            if len(代码列表) != len(set(代码列表)):
                重复代码 = [
                    代码
                    for 代码, 数量 in Counter(代码列表).items()
                    if 数量 > 1
                ]
                raise ValueError(
                    f"工作表 {工作表.title} 存在重复证券代码: {重复代码}"
                )

            检查结果.append(
                行情工作表信息(
                    工作表名称=工作表.title,
                    股票数量=len(代码列表),
                    交易日数量=len(set(有效日期)),
                    开始日期=min(有效日期).date().isoformat(),
                    结束日期=max(有效日期).date().isoformat(),
                    证券代码=tuple(代码列表),
                )
            )
    finally:
        工作簿.close()

    return 检查结果


def 生成审计报告(
    初始快照路径: Path,
    调整记录路径: Path,
    行情文件路径: Path,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """生成股票池、调整链条和行情覆盖率审计结果。"""
    初始成分 = 读取成分快照(初始快照路径)
    调整记录 = 读取成分调整(调整记录路径)
    行情工作表 = 检查行情工作簿(行情文件路径)

    初始代码 = set(初始成分["证券代码"])
    历史成分并集 = 初始代码.union(set(调整记录["证券代码"]))
    最终代码, 调整链条 = 重建成分股链条(初始代码, 调整记录)

    各表代码集合 = [set(工作表.证券代码) for 工作表 in 行情工作表]
    行情代码 = set.union(*各表代码集合) if 各表代码集合 else set()
    各表股票一致 = all(
        代码集合 == 各表代码集合[0] for 代码集合 in 各表代码集合
    )

    缺失代码 = sorted(历史成分并集.difference(行情代码))
    额外代码 = sorted(行情代码.difference(历史成分并集))
    已覆盖代码 = sorted(历史成分并集.intersection(行情代码))
    调整链条有效 = bool(
        (调整链条["调整后成分数量"] == 300).all()
        and 调整链条["无效剔除代码"].map(len).eq(0).all()
        and 调整链条["重复纳入代码"].map(len).eq(0).all()
    )

    报告 = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "初始成分快照": {
            "股票数量": len(初始代码),
            "文件": 初始快照路径.name,
        },
        "成分调整": {
            "事件数量": len(调整记录),
            "调整日数量": int(调整记录["调整日期"].nunique()),
            "开始日期": 调整记录["调整日期"].min().date().isoformat(),
            "结束日期": 调整记录["调整日期"].max().date().isoformat(),
            "纳入数量": int((调整记录["调整类型"] == "纳入").sum()),
            "剔除数量": int((调整记录["调整类型"] == "剔除").sum()),
        },
        "成分股": {
            "历史并集数量": len(历史成分并集),
            "最终成分数量": len(最终代码),
            "调整链条有效": 调整链条有效,
        },
        "行情工作簿": {
            "文件": 行情文件路径.name,
            "工作表": [
                {
                    键: 值
                    for 键, 值 in asdict(工作表).items()
                    if 键 != "证券代码"
                }
                for 工作表 in 行情工作表
            ],
            "各工作表股票池一致": 各表股票一致,
            "唯一股票数量": len(行情代码),
        },
        "覆盖情况": {
            "已覆盖历史代码数": len(已覆盖代码),
            "缺失历史代码数": len(缺失代码),
            "额外行情代码数": len(额外代码),
            "覆盖率": len(已覆盖代码) / len(历史成分并集),
            "可开展完整动态研究": len(缺失代码) == 0 and 调整链条有效,
        },
    }

    缺失代码表 = pd.DataFrame({"证券代码": 缺失代码})
    额外代码表 = pd.DataFrame({"证券代码": 额外代码})
    已覆盖代码表 = pd.DataFrame({"证券代码": 已覆盖代码})
    return 报告, 调整链条, 缺失代码表, 额外代码表, 已覆盖代码表

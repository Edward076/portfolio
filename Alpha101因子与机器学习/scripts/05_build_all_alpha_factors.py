"""计算剩余6个因子并合并为完整18因子数据集。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.alpha_factors import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    全部因子名称,
    宽表因子转长表,
    第二批因子名称,
    计算第二批因子,
)


清洗面板文件 = 项目根目录 / "data" / "processed" / "clean_stock_panel.parquet"
第一批因子文件 = 项目根目录 / "data" / "processed" / "alpha_first_batch.parquet"
完整因子文件 = 项目根目录 / "data" / "processed" / "alpha_all_18.parquet"
摘要文件 = 项目根目录 / "outputs" / "alpha_factors" / "all_18_summary.json"


def _转换宽表(
    面板: pd.DataFrame,
    列名: str,
    股票代码: list[str],
) -> pd.DataFrame:
    """把面板字段转换为日期×股票矩阵。"""
    return (
        面板.loc[面板["证券代码"].isin(股票代码)]
        .pivot(index="日期", columns="证券代码", values=列名)
        .sort_index()
        .reindex(columns=股票代码)
    )


def 主程序() -> None:
    """计算第二批因子并与第一批进行一对一合并。"""
    面板 = pd.read_parquet(清洗面板文件)
    面板["日期"] = pd.to_datetime(面板["日期"]).dt.normalize()
    第一批因子 = pd.read_parquet(第一批因子文件)
    第一批因子["日期"] = pd.to_datetime(第一批因子["日期"]).dt.normalize()

    曾为成分代码 = set(
        面板.loc[面板["是否沪深300成分"], "证券代码"].unique()
    )
    价格异常代码 = set(
        面板.loc[面板["价格逻辑异常"], "证券代码"].unique()
    )
    可用代码 = sorted(曾为成分代码.difference(价格异常代码))

    开盘价 = _转换宽表(面板, "开盘价", 可用代码)
    收盘价 = _转换宽表(面板, "收盘价", 可用代码)
    最高价 = _转换宽表(面板, "最高价", 可用代码)
    最低价 = _转换宽表(面板, "最低价", 可用代码)
    成交量 = _转换宽表(面板, "成交量", 可用代码)
    成分掩码 = _转换宽表(
        面板,
        "是否沪深300成分",
        可用代码,
    ).fillna(False).astype(bool)
    可交易掩码 = _转换宽表(
        面板,
        "是否可交易",
        可用代码,
    ).fillna(False).astype(bool)

    第二批结果 = 计算第二批因子(
        开盘价=开盘价,
        收盘价=收盘价,
        最高价=最高价,
        最低价=最低价,
        成交量=成交量,
        横截面掩码=成分掩码,
    )
    第二批长表 = 宽表因子转长表(
        第二批结果,
        成分掩码 & 可交易掩码,
    )
    第二批长表[list(第二批因子名称)] = 第二批长表[
        list(第二批因子名称)
    ].replace([np.inf, -np.inf], np.nan)

    完整因子 = 第一批因子.merge(
        第二批长表,
        on=["日期", "证券代码"],
        how="inner",
        validate="one_to_one",
    )
    if "证券名称_x" in 完整因子:
        完整因子 = 完整因子.rename(columns={"证券名称_x": "证券名称"})
        完整因子 = 完整因子.drop(columns=["证券名称_y"], errors="ignore")

    完整因子 = 完整因子[
        ["日期", "证券代码", "证券名称", *全部因子名称]
    ].sort_values(["日期", "证券代码"])

    摘要 = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "样本行数": len(完整因子),
        "股票数量": int(完整因子["证券代码"].nunique()),
        "交易日数量": int(完整因子["日期"].nunique()),
        "因子数量": len(全部因子名称),
        "因子名称": list(全部因子名称),
        "因子缺失率": {
            因子名称: float(完整因子[因子名称].isna().mean())
            for 因子名称 in 全部因子名称
        },
        "无穷值数量": int(
            np.isinf(完整因子[list(全部因子名称)].to_numpy()).sum()
        ),
    }

    完整因子文件.parent.mkdir(parents=True, exist_ok=True)
    摘要文件.parent.mkdir(parents=True, exist_ok=True)
    完整因子.to_parquet(
        完整因子文件,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    摘要文件.write_text(
        json.dumps(摘要, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("完整18个Alpha101因子计算完成")
    print(f"样本行数: {摘要['样本行数']:,}")
    print(f"因子数量: {摘要['因子数量']}")
    print(f"输出文件: {完整因子文件}")


if __name__ == "__main__":
    主程序()

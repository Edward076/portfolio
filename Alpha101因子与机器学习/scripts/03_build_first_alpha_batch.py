"""第三阶段：计算第一批12个精选Alpha101因子。"""

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
    第一批因子名称,
    宽表因子转长表,
    计算第一批因子,
)


清洗面板文件 = 项目根目录 / "data" / "processed" / "clean_stock_panel.parquet"
因子输出文件 = 项目根目录 / "data" / "processed" / "alpha_first_batch.parquet"
摘要输出文件 = 项目根目录 / "outputs" / "alpha_factors" / "first_batch_summary.json"


def _转换宽表(
    面板: pd.DataFrame,
    列名: str,
    股票代码: list[str],
) -> pd.DataFrame:
    """把指定字段转换为日期×股票矩阵。"""
    return (
        面板.loc[面板["证券代码"].isin(股票代码)]
        .pivot(index="日期", columns="证券代码", values=列名)
        .sort_index()
        .reindex(columns=股票代码)
    )


def 主程序() -> None:
    """计算因子，过滤不可用样本并保存Parquet。"""
    面板 = pd.read_parquet(清洗面板文件)
    面板["日期"] = pd.to_datetime(面板["日期"]).dt.normalize()

    曾为成分代码 = set(
        面板.loc[面板["是否沪深300成分"], "证券代码"].unique()
    )
    价格异常代码 = set(
        面板.loc[面板["价格逻辑异常"], "证券代码"].unique()
    )
    可用代码 = sorted(曾为成分代码.difference(价格异常代码))
    if not 可用代码:
        raise ValueError("没有可用于因子计算的股票")

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
    最终样本掩码 = 成分掩码 & 可交易掩码

    因子结果 = 计算第一批因子(
        开盘价=开盘价,
        收盘价=收盘价,
        最高价=最高价,
        最低价=最低价,
        成交量=成交量,
        横截面掩码=成分掩码,
    )
    因子长表 = 宽表因子转长表(因子结果, 最终样本掩码)
    因子长表[list(第一批因子名称)] = 因子长表[
        list(第一批因子名称)
    ].replace([np.inf, -np.inf], np.nan)

    名称映射 = (
        面板[["证券代码", "证券名称"]]
        .drop_duplicates(subset=["证券代码"], keep="last")
        .set_index("证券代码")["证券名称"]
    )
    因子长表.insert(
        2,
        "证券名称",
        因子长表["证券代码"].map(名称映射).astype("string"),
    )

    缺失率 = {
        因子名称: float(因子长表[因子名称].isna().mean())
        for 因子名称 in 第一批因子名称
    }
    摘要 = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "因子数量": len(第一批因子名称),
        "因子名称": list(第一批因子名称),
        "样本行数": len(因子长表),
        "可用股票数量": len(可用代码),
        "排除OHLC异常股票数量": len(价格异常代码),
        "交易日数量": int(因子长表["日期"].nunique()),
        "因子缺失率": 缺失率,
        "样本方案": "现有动态成分子样本，排除OHLC异常股票。",
        "局限": "历史成分行情覆盖率为56.78%，存在样本选择偏差。",
    }

    因子输出文件.parent.mkdir(parents=True, exist_ok=True)
    摘要输出文件.parent.mkdir(parents=True, exist_ok=True)
    因子长表.to_parquet(
        因子输出文件,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    摘要输出文件.write_text(
        json.dumps(摘要, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("第一批Alpha101因子计算完成")
    print(f"可用股票数量: {摘要['可用股票数量']}")
    print(f"样本行数: {摘要['样本行数']:,}")
    print(f"因子数量: {摘要['因子数量']}")
    print(f"输出文件: {因子输出文件}")


if __name__ == "__main__":
    主程序()

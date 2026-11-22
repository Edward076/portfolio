"""构造18因子处理结果、量价特征、未来收益标签和时间数据集。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.alpha_factors import 全部因子名称  # pyright: ignore[reportMissingImports]  # noqa: E402
from csi300_ml.dataset_builder import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    普通量价特征名称,
    按日处理因子,
    构造普通量价特征,
    添加未来收益标签,
    添加时间数据集标记,
)


清洗面板文件 = 项目根目录 / "data" / "processed" / "clean_stock_panel.parquet"
完整因子文件 = 项目根目录 / "data" / "processed" / "alpha_all_18.parquet"
模型数据文件 = 项目根目录 / "data" / "processed" / "model_dataset.parquet"
摘要文件 = 项目根目录 / "outputs" / "dataset" / "model_dataset_summary.json"


def 主程序() -> None:
    """生成可供单因子检验和机器学习使用的统一数据集。"""
    面板 = pd.read_parquet(清洗面板文件)
    因子 = pd.read_parquet(完整因子文件)
    面板["日期"] = pd.to_datetime(面板["日期"]).dt.normalize()
    因子["日期"] = pd.to_datetime(因子["日期"]).dt.normalize()

    处理后因子 = 按日处理因子(因子, 全部因子名称)
    特征面板 = 构造普通量价特征(面板)

    一日标签 = 添加未来收益标签(特征面板, 预测周期=1)[
        ["日期", "证券代码", "未来1日收益", "未来1日相对收益"]
    ]
    十日标签 = 添加未来收益标签(特征面板, 预测周期=10)[
        ["日期", "证券代码", "未来10日收益", "未来10日相对收益"]
    ]
    主标签面板 = 添加未来收益标签(特征面板, 预测周期=5)
    主标签面板 = 添加时间数据集标记(主标签面板)
    主标签面板 = 主标签面板.merge(
        一日标签,
        on=["日期", "证券代码"],
        how="left",
        validate="one_to_one",
    ).merge(
        十日标签,
        on=["日期", "证券代码"],
        how="left",
        validate="one_to_one",
    )

    保留列 = [
        "日期",
        "证券代码",
        "建仓日期",
        "退出日期",
        "数据集",
        *普通量价特征名称,
        "未来1日收益",
        "未来1日相对收益",
        "未来5日收益",
        "未来5日相对收益",
        "未来5日分类标签",
        "未来10日收益",
        "未来10日相对收益",
    ]
    模型数据 = 处理后因子.merge(
        主标签面板[保留列],
        on=["日期", "证券代码"],
        how="left",
        validate="one_to_one",
    )
    模型数据 = 模型数据.loc[
        模型数据["数据集"].notna()
        & 模型数据["未来5日分类标签"].notna()
    ].copy()
    模型数据["未来5日分类标签"] = 模型数据[
        "未来5日分类标签"
    ].astype("int8")

    特征名称 = [*全部因子名称, *普通量价特征名称]
    摘要 = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "样本行数": len(模型数据),
        "特征数量": len(特征名称),
        "Alpha因子数量": len(全部因子名称),
        "普通量价特征数量": len(普通量价特征名称),
        "各数据集样本数": {
            str(名称): int(数量)
            for 名称, 数量 in 模型数据["数据集"].value_counts().items()
        },
        "各数据集正样本比例": {
            str(名称): float(分组["未来5日分类标签"].mean())
            for 名称, 分组 in 模型数据.groupby("数据集")
        },
        "特征缺失率": {
            特征: float(模型数据[特征].isna().mean())
            for 特征 in 特征名称
        },
        "标签口径": "t日收盘信号，t+1开盘建仓，t+6开盘退出；减去当日可得动态成分子样本中位数。",
    }

    模型数据文件.parent.mkdir(parents=True, exist_ok=True)
    摘要文件.parent.mkdir(parents=True, exist_ok=True)
    模型数据.to_parquet(
        模型数据文件,
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    摘要文件.write_text(
        json.dumps(摘要, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("模型数据集构造完成")
    print(f"样本行数: {摘要['样本行数']:,}")
    print(f"特征数量: {摘要['特征数量']}")
    print(f"数据集样本数: {摘要['各数据集样本数']}")
    print(f"输出文件: {模型数据文件}")


if __name__ == "__main__":
    主程序()

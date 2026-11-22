"""将第一批因子Parquet导出为可直接查看的Excel文件。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xlsxwriter


项目根目录 = Path(__file__).resolve().parents[1]
因子文件 = 项目根目录 / "data" / "processed" / "alpha_first_batch.parquet"
Excel文件 = 项目根目录 / "data" / "processed" / "alpha_first_batch.xlsx"


def 主程序() -> None:
    """使用低内存逐行方式导出完整Excel工作簿。"""
    因子数据 = pd.read_parquet(因子文件)
    因子列 = [列名 for 列名 in 因子数据.columns if 列名.startswith("Alpha")]

    工作簿 = xlsxwriter.Workbook(Excel文件, {"constant_memory": True})
    try:
        标题格式 = 工作簿.add_format(
            {
                "bold": True,
                "bg_color": "#D9EAF7",
                "border": 1,
                "align": "center",
            }
        )
        日期格式 = 工作簿.add_format({"num_format": "yyyy-mm-dd"})
        小数格式 = 工作簿.add_format({"num_format": "0.000000"})

        数据表 = 工作簿.add_worksheet("因子数据")
        数据表.freeze_panes(1, 3)
        数据表.autofilter(0, 0, len(因子数据), len(因子数据.columns) - 1)
        数据表.set_column(0, 0, 12)
        数据表.set_column(1, 1, 13)
        数据表.set_column(2, 2, 12)
        数据表.set_column(3, len(因子数据.columns) - 1, 13, 小数格式)

        for 列号, 列名 in enumerate(因子数据.columns):
            数据表.write(0, 列号, 列名, 标题格式)

        for 行号, 数据行 in enumerate(因子数据.itertuples(index=False, name=None), start=1):
            for 列号, 值 in enumerate(数据行):
                if pd.isna(值):
                    continue
                if 列号 == 0:
                    数据表.write_datetime(
                        行号,
                        列号,
                        pd.Timestamp(值).to_pydatetime(),
                        日期格式,
                    )
                else:
                    数据表.write(行号, 列号, 值)

        摘要表 = 工作簿.add_worksheet("数据摘要")
        摘要表.set_column(0, 0, 24)
        摘要表.set_column(1, 1, 18)
        摘要项目 = [
            ("样本行数", len(因子数据)),
            ("股票数量", 因子数据["证券代码"].nunique()),
            ("交易日数量", 因子数据["日期"].nunique()),
            ("因子数量", len(因子列)),
        ]
        for 行号, (名称, 数值) in enumerate(摘要项目):
            摘要表.write(行号, 0, 名称, 标题格式)
            摘要表.write(行号, 1, 数值)

        起始行 = len(摘要项目) + 2
        摘要表.write(起始行, 0, "因子名称", 标题格式)
        摘要表.write(起始行, 1, "缺失率", 标题格式)
        for 偏移, 因子名称 in enumerate(因子列, start=1):
            摘要表.write(起始行 + 偏移, 0, 因子名称)
            摘要表.write(
                起始行 + 偏移,
                1,
                float(因子数据[因子名称].isna().mean()),
            )
    finally:
        工作簿.close()

    print(f"Excel查看版已生成: {Excel文件}")
    print(f"数据行数: {len(因子数据):,}")


if __name__ == "__main__":
    主程序()

"""评价18个Alpha101因子的Rank IC和五分组收益。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.alpha_factors import 全部因子名称  # pyright: ignore[reportMissingImports]  # noqa: E402
from csi300_ml.factor_evaluation import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    计算HAC均值t值,
    计算五分组收益,
    计算每日RankIC,
    汇总五分组收益,
    汇总RankIC,
)


模型数据文件 = 项目根目录 / "data" / "processed" / "model_dataset.parquet"
结果目录 = 项目根目录 / "outputs" / "factor_evaluation"
每日IC文件 = 结果目录 / "daily_rank_ic.parquet"
每日分组文件 = 结果目录 / "daily_group_returns.parquet"
评价工作簿 = 结果目录 / "factor_evaluation.xlsx"


def _汇总年度IC(每日IC: pd.DataFrame) -> pd.DataFrame:
    """按自然年汇总各因子IC。"""
    结果: list[dict[str, object]] = []
    带年份数据 = 每日IC.copy()
    带年份数据["年份"] = pd.to_datetime(带年份数据["日期"]).dt.year
    for 年份, 年度数据 in 带年份数据.groupby("年份"):
        for 因子 in 全部因子名称:
            序列 = 年度数据[因子].dropna()
            标准差 = 序列.std(ddof=1)
            结果.append(
                {
                    "年份": int(年份),
                    "因子": 因子,
                    "IC均值": 序列.mean(),
                    "ICIR": (
                        序列.mean() / 标准差 if 标准差 > 0 else np.nan
                    ),
                    "正IC比例": (序列 > 0).mean(),
                    "HAC_t值": 计算HAC均值t值(序列, 最大滞后=5),
                    "有效交易日": len(序列),
                }
            )
    return pd.DataFrame(结果)


def 主程序() -> None:
    """生成单因子完整评价结果。"""
    数据 = pd.read_parquet(模型数据文件)
    数据["日期"] = pd.to_datetime(数据["日期"]).dt.normalize()
    日期数据集映射 = (
        数据[["日期", "数据集"]]
        .drop_duplicates("日期")
        .set_index("日期")["数据集"]
    )

    每日IC = 计算每日RankIC(数据, 全部因子名称)
    IC汇总 = 汇总RankIC(每日IC, 日期数据集映射, 全部因子名称)
    年度IC = _汇总年度IC(每日IC)
    每日分组 = 计算五分组收益(数据, 全部因子名称)
    分组汇总 = 汇总五分组收益(每日分组)

    测试集结果 = (
        IC汇总.loc[IC汇总["数据集"] == "测试集"]
        .merge(
            分组汇总.loc[分组汇总["数据集"] == "测试集"],
            on=["数据集", "因子"],
            how="left",
            validate="one_to_one",
        )
        .sort_values("IC均值", key=lambda 列: 列.abs(), ascending=False)
    )

    结果目录.mkdir(parents=True, exist_ok=True)
    每日IC.to_parquet(每日IC文件, index=False, compression="zstd")
    每日分组.to_parquet(每日分组文件, index=False, compression="zstd")
    with pd.ExcelWriter(评价工作簿, engine="xlsxwriter") as 写入器:
        测试集结果.to_excel(写入器, sheet_name="测试集总览", index=False)
        IC汇总.to_excel(写入器, sheet_name="分数据集IC", index=False)
        年度IC.to_excel(写入器, sheet_name="年度IC", index=False)
        分组汇总.to_excel(写入器, sheet_name="五分组收益", index=False)
        for 工作表 in 写入器.sheets.values():
            工作表.freeze_panes(1, 0)
            工作表.autofilter(0, 0, 工作表.dim_rowmax, 工作表.dim_colmax)
            工作表.set_column(0, 工作表.dim_colmax, 14)

    print("单因子评价完成")
    print(f"每日IC交易日数: {len(每日IC)}")
    print(f"评价因子数: {len(全部因子名称)}")
    print(f"输出文件: {评价工作簿}")


if __name__ == "__main__":
    主程序()

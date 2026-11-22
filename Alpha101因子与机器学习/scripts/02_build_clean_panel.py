"""第二阶段：将 Choice 行情宽表转换为清洗后的长表面板。"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


项目根目录 = Path(__file__).resolve().parents[1]
源码目录 = 项目根目录 / "src"
if str(源码目录) not in sys.path:
    sys.path.insert(0, str(源码目录))

from csi300_ml.data_audit import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    读取成分调整,
    读取成分快照,
)
from csi300_ml.panel_processing import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    添加动态成分标记,
    清洗行情,
    读取行情长表,
)


行情文件 = 项目根目录 / "沪深300选股数据导出19-25.xlsx"
初始成分文件 = 项目根目录 / "沪深300成分详情19 1 2.xlsx"
成分调整文件 = 项目根目录 / "沪深300成分进出记录(000300.SH).xlsx"
面板输出目录 = 项目根目录 / "data" / "processed"
报告输出目录 = 项目根目录 / "outputs" / "data_cleaning"


def 主程序() -> None:
    """生成清洗面板、异常样本和清洗摘要。"""
    行情长表 = 读取行情长表(行情文件)
    清洗后行情, 质量统计 = 清洗行情(行情长表)

    初始成分 = 读取成分快照(初始成分文件)
    调整记录 = 读取成分调整(成分调整文件)
    初始代码 = set(初始成分["证券代码"])
    研究面板 = 添加动态成分标记(清洗后行情, 初始代码, 调整记录)

    研究面板 = 研究面板.sort_values(["证券代码", "日期"]).reset_index(drop=True)
    日期成分数量 = (
        研究面板.loc[研究面板["是否沪深300成分"]]
        .groupby("日期")["证券代码"]
        .nunique()
    )
    价格异常股票统计 = (
        研究面板.loc[研究面板["价格逻辑异常"]]
        .groupby(["证券代码", "证券名称"], as_index=False)
        .agg(
            异常行数=("价格逻辑异常", "size"),
            最早异常日期=("日期", "min"),
            最晚异常日期=("日期", "max"),
        )
        .sort_values("异常行数", ascending=False)
    )

    摘要 = {
        "生成时间": datetime.now().isoformat(timespec="seconds"),
        "数据行数": len(研究面板),
        "股票数量": int(研究面板["证券代码"].nunique()),
        "交易日数量": int(研究面板["日期"].nunique()),
        "开始日期": 研究面板["日期"].min().date().isoformat(),
        "结束日期": 研究面板["日期"].max().date().isoformat(),
        "重复股票日期数": int(
            研究面板.duplicated(subset=["日期", "证券代码"]).sum()
        ),
        "现有行情中的每日成分数量最小值": int(日期成分数量.min()),
        "现有行情中的每日成分数量最大值": int(日期成分数量.max()),
        "现有行情中的每日成分数量中位数": float(日期成分数量.median()),
        "价格逻辑异常股票数": len(价格异常股票统计),
        "价格逻辑异常股票代码": 价格异常股票统计["证券代码"].tolist(),
        "清洗质量": 质量统计,
        "正式研究数据是否完整": bool(
            日期成分数量.eq(300).all()
            and 研究面板["证券代码"].nunique() == 509
        ),
        "后续研究方案": "使用现有动态成分子样本，并排除OHLC异常股票。",
        "局限说明": "历史成分行情覆盖率为56.78%，结果存在样本选择偏差。",
    }

    面板输出目录.mkdir(parents=True, exist_ok=True)
    报告输出目录.mkdir(parents=True, exist_ok=True)
    研究面板.to_parquet(
        面板输出目录 / "clean_stock_panel.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    (报告输出目录 / "cleaning_summary.json").write_text(
        json.dumps(摘要, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    异常记录 = 研究面板.loc[
        研究面板[
            ["价格零值", "成交量非正", "成交金额非正", "价格逻辑异常"]
        ].any(axis=1)
    ]
    异常记录.to_parquet(
        报告输出目录 / "invalid_market_rows.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    print("清洗面板生成完成")
    print(f"数据行数: {摘要['数据行数']:,}")
    print(f"股票数量: {摘要['股票数量']}")
    print(f"交易日数量: {摘要['交易日数量']}")
    print(
        "每日已覆盖成分数量: "
        f"{摘要['现有行情中的每日成分数量最小值']}"
        f"—{摘要['现有行情中的每日成分数量最大值']}"
    )
    print(f"正式研究数据是否完整: {摘要['正式研究数据是否完整']}")
    print(f"面板文件: {面板输出目录 / 'clean_stock_panel.parquet'}")


if __name__ == "__main__":
    主程序()
